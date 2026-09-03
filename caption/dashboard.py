"""A small web dashboard for the captioner pipeline. Decorative by design.

    python -m caption.dashboard --port 8765

Reads file counts, gen logs, the train log and nvidia-smi; serves one page
that polls /api/status. Read-only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "caption"

JOBS = [
    ("probe", "probe set", DATA / "probe", 600),
    ("latents", "training set", DATA / "latents", 6000),
]

_cache: dict[Path, dict] = {}
_lock = threading.Lock()


def _record(path: Path) -> dict:
    with _lock:
        if path in _cache:
            return _cache[path]
    try:
        r = torch.load(path, map_location="cpu")
        lat = r["latents"].float()  # (L, T, D)
        # A colour fingerprint: 24 numbers from the mid layer, mean over time.
        finger = lat[1].mean(0)[:24].tolist()
        info = {
            "id": path.stem,
            "attrs": r["attrs"],
            "prose": r["prose"],
            "text": r["text"],
            "seconds": round(float(r["seconds"]), 1),
            "mtime": path.stat().st_mtime,
            "finger": finger,
        }
    except (OSError, RuntimeError, KeyError, EOFError) as exc:  # partially written
        return {"id": path.stem, "error": str(exc), "mtime": 0, "seconds": 0}
    with _lock:
        _cache[path] = info
    return info


def job_status(key: str, name: str, d: Path, total: int) -> dict:
    files = sorted(d.glob("*.pt")) if d.exists() else []
    recent = [_record(p) for p in files[-40:]]
    ok = [r for r in recent if "error" not in r]
    now = time.time()
    # Rate from the last 30 files' mtimes, or zero if the newest is stale.
    rate = 0.0
    if len(ok) >= 3:
        span = ok[-1]["mtime"] - ok[max(0, len(ok) - 30)]["mtime"]
        n = min(30, len(ok)) - 1
        if span > 0 and now - ok[-1]["mtime"] < 120:
            rate = n / span * 3600
    seconds = (
        sum(_record(p).get("seconds", 0) for p in files) if len(files) < 20000 else 0
    )
    remaining = max(0, total - len(files))
    return {
        "key": key,
        "name": name,
        "done": len(files),
        "total": total,
        "rate_per_hour": round(rate),
        "eta_seconds": round(remaining / rate * 3600) if rate else None,
        "audio_seconds": round(seconds),
        "active": bool(rate),
        "recent": ok[-8:][::-1],
    }


def gpu_status() -> list[dict]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, name, util, used, total, temp, power = [x.strip() for x in line.split(",")]
        gpus.append(
            {
                "index": int(idx),
                "name": name.replace("NVIDIA ", ""),
                "util": int(util),
                "mem_used": int(used),
                "mem_total": int(total),
                "temp": int(temp),
                "power": float(power),
            }
        )
    return gpus


_STEP = re.compile(
    r"step (\d+)/(\d+) loss ([\d.]+) (?:aux [\d.]+ )?layers ([\d. ]+?) ([\d.]+)min"
)
_HELD = re.compile(r"held-out loss ([\d.]+)")
_GAP = re.compile(r"\(gap ([+-][\d.]+)\)")


def train_status() -> dict | None:
    log = DATA / "train.log"
    if not log.exists():
        return None
    text = log.read_text()
    steps = _STEP.findall(text)
    held = _HELD.findall(text)
    if not steps:
        return {"state": "starting"}
    step, total, loss, layers, minutes = steps[-1]
    losses = [float(s[2]) for s in steps]
    return {
        "state": "done" if "saved to" in text else "training",
        "step": int(step),
        "total": int(total),
        "loss": float(loss),
        "losses": losses[-60:],
        "held": [float(h) for h in held],
        "gap": [float(g) for g in _GAP.findall(text)],
        "layers": [float(x) for x in layers.split()],
        "minutes": float(minutes),
        "samples": re.findall(r"truth:\s+(.*)\ncaption:\s+(.*)", text)[:4],
    }


_RL_STEP = re.compile(
    r"step (\d+)/(\d+) reward (-?[\d.]+) \(best-in-group (-?[\d.]+)\) kl ([\d.]+) len (\d+) ([\d.]+)min"
)
_RL_HELD = re.compile(
    r"held-out reward(?: before| after)?: policy (-?[\d.]+)(?:\s+reference (-?[\d.]+))?"
)


def rl_status() -> list[dict]:
    out = []
    for log in sorted(DATA.glob("rl*.log")):
        text = log.read_text()
        steps = _RL_STEP.findall(text)
        held = _RL_HELD.findall(text)
        samples = re.findall(r"^    (.+)$", text, flags=re.MULTILINE)
        name = log.stem.replace("rl_", "").replace("rl", "likelihood") or "rl"
        if name == "rt":
            name = "roundtrip"
        elif name == "hi":
            name = "likelihood, high lr"
        entry = {
            "name": name,
            "log": log.name,
            "state": (
                "done"
                if "saved to" in text
                else "failed"
                if "Traceback" in text
                else "running"
                if time.time() - log.stat().st_mtime < 600
                else "stopped"
            ),
            "step": int(steps[-1][0]) if steps else 0,
            "total": int(steps[-1][1]) if steps else 0,
            "rewards": [float(x[2]) for x in steps][-80:],
            "best": [float(x[3]) for x in steps][-80:],
            "kl": float(steps[-1][4]) if steps else 0.0,
            "len": int(steps[-1][5]) if steps else 0,
            "minutes": float(steps[-1][6]) if steps else 0.0,
            "held": [(float(a), float(b) if b else None) for a, b in held],
            "sample": samples[-1] if samples else "",
            "mtime": log.stat().st_mtime,
        }
        out.append(entry)
    out.sort(key=lambda e: -e["mtime"])
    return out


def status() -> dict:
    return {
        "time": time.time(),
        "jobs": [job_status(*j) for j in JOBS],
        "gpus": gpu_status(),
        "train": train_status(),
        "rl": rl_status(),
        "descriptions": sum(1 for _ in (DATA / "descriptions.jsonl").open())
        if (DATA / "descriptions.jsonl").exists()
        else 0,
    }


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>breeze, backwards</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0b0d12;--fg:#e8e6df;--dim:#8a8f9c;--card:#141823;--line:#222838;
  --a:#f7b267;--b:#7bdff2;--c:#b388ff;--d:#8ce99a;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;overflow-x:hidden}
canvas#sky{position:fixed;inset:0;z-index:0;opacity:.55}
main{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:28px 20px 60px}
h1{font-weight:500;font-size:26px;margin:0 0 4px;letter-spacing:.2px}
h1 small{color:var(--dim);font-size:14px;margin-left:10px}
.sub{color:var(--dim);margin:0 0 22px;font-family:var(--mono);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.card{background:color-mix(in oklab,var(--card) 88%,transparent);border:1px solid var(--line);border-radius:14px;padding:16px 18px;backdrop-filter:blur(6px)}
.card h2{margin:0 0 10px;font-size:13px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.big{font-size:34px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.1}
.big span{font-size:15px;color:var(--dim);font-weight:400;margin-left:6px}
.bar{height:10px;border-radius:99px;background:#1c2130;overflow:hidden;margin:10px 0 8px;position:relative}
.bar i{position:absolute;inset:0;width:0;border-radius:99px;transition:width .8s ease;
  background:linear-gradient(90deg,var(--a),var(--b));}
.bar i.on::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.35),transparent);
  animation:sheen 1.6s linear infinite}
@keyframes sheen{from{transform:translateX(-100%)}to{transform:translateX(100%)}}
.meta{display:flex;gap:14px;flex-wrap:wrap;color:var(--dim);font-family:var(--mono);font-size:12.5px}
.meta b{color:var(--fg);font-weight:500}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#3a4156;margin-right:6px;vertical-align:1px}
.dot.on{background:var(--d);box-shadow:0 0 10px var(--d);animation:pulse 1.2s ease-in-out infinite}
@keyframes pulse{50%{opacity:.35}}
.gpu{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.gauge{position:relative;height:64px}
.gauge svg{width:100%;height:100%}
.gauge .v{position:absolute;left:0;right:0;bottom:2px;text-align:center;font-family:var(--mono);font-size:13px}
.clip{padding:10px 0;border-top:1px dashed var(--line);animation:in .5s ease}
.clip:first-child{border-top:0}
@keyframes in{from{opacity:0;transform:translateY(6px)}}
.clip .p{margin:0 0 4px}
.clip .t{color:var(--dim);font-size:13px;font-style:italic;margin:0 0 6px}
.chip{display:inline-block;font-family:var(--mono);font-size:11px;padding:2px 8px;border-radius:99px;margin:2px 4px 0 0;
  background:#1c2130;color:var(--b);border:1px solid #263044}
.chip.g{color:var(--a)}.chip.m{color:var(--c)}
.wide{grid-column:1/-1}
.spark{width:100%;height:70px}
.foot{margin-top:26px;color:var(--dim);font-size:12.5px;font-family:var(--mono)}
.swatch{display:inline-block;width:14px;height:14px;border-radius:4px;margin-right:2px;vertical-align:-2px}
.layers{display:flex;gap:6px;align-items:flex-end;height:60px;margin-top:8px}
.layers div{flex:1;background:linear-gradient(180deg,var(--c),#3a2f66);border-radius:4px 4px 0 0;position:relative;transition:height .6s}
.layers div::after{content:attr(data-l);position:absolute;top:100%;left:0;right:0;text-align:center;font-size:11px;color:var(--dim);font-family:var(--mono)}
</style></head><body>
<canvas id="sky"></canvas>
<main>
<h1>breeze, backwards <small>voice design run in reverse</small></h1>
<p class="sub" id="sub">connecting…</p>
<div class="grid" id="jobs"></div>
<div class="grid" style="margin-top:14px">
  <div class="card"><h2>gpus</h2><div class="gpu" id="gpus"></div></div>
  <div class="card"><h2>rendered so far</h2><div class="big" id="audio">0<span>of synthetic speech</span></div>
    <div class="meta" id="audiometa"></div>
    <div style="margin-top:12px;color:var(--dim);font-size:13px">every clip is a voice that never existed, described in prose that an llm made up, so a small model can learn to describe voices that do.</div></div>
  <div class="card" id="traincard"><h2>captioner training</h2><div id="train" style="color:var(--dim)">not started</div></div>
</div>
<div class="grid" id="rl" style="margin-top:14px"></div>
<div class="grid" style="margin-top:14px">
  <div class="card wide"><h2>fresh off the gpu</h2><div id="recent"></div></div>
</div>
<p class="foot" id="foot"></p>
</main>
<script>
const $=s=>document.querySelector(s);
const fmtEta=s=>{if(s==null)return "—";if(s<60)return Math.round(s)+"s";if(s<3600)return Math.round(s/60)+" min";return (s/3600).toFixed(1)+" h"};
const fmtAudio=s=>{const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;return (h?h+"h ":"")+(m?m+"m ":"")+x+"s"};
function hue(v,i){return `hsl(${((v*40)+i*31)%360|0} 70% 62%)`}
function gauge(label,frac,text,color){const r=26,c=Math.PI*r; return `<div class="gauge"><svg viewBox="0 0 64 40"><path d="M6 36 A26 26 0 0 1 58 36" fill="none" stroke="#1c2130" stroke-width="7" stroke-linecap="round"/><path d="M6 36 A26 26 0 0 1 58 36" fill="none" stroke="${color}" stroke-width="7" stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${c*(1-frac)}" style="transition:stroke-dashoffset .8s"/></svg><div class="v">${text}<div style="color:var(--dim);font-size:11px">${label}</div></div></div>`}
let sky, particles=[], fingers=[];
function initSky(){sky=$("#sky");const ctx=sky.getContext("2d");const fit=()=>{sky.width=innerWidth;sky.height=innerHeight};fit();addEventListener("resize",fit);
  for(let i=0;i<90;i++)particles.push({x:Math.random()*innerWidth,y:Math.random()*innerHeight,r:1+Math.random()*2.5,s:.15+Math.random()*.5,h:200+Math.random()*100});
  (function tick(){ctx.clearRect(0,0,sky.width,sky.height);const busy=window._busy||0;
    for(const p of particles){p.y-=p.s*(1+busy*2);p.x+=Math.sin(p.y/60+p.r)*.3;if(p.y<-5){p.y=sky.height+5;p.x=Math.random()*sky.width;if(fingers.length){const f=fingers[Math.floor(Math.random()*fingers.length)];p.h=((f*40)+180+360)%360}}
      ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,7);ctx.fillStyle=`hsl(${p.h} 70% 65% / .7)`;ctx.fill()}
    requestAnimationFrame(tick)})()}
function render(d){
  const busy=d.jobs.filter(j=>j.active).length;window._busy=busy;
  $("#sub").textContent=`${d.descriptions} descriptions written · ${busy} generator${busy==1?"":"s"} running · ${new Date(d.time*1000).toLocaleTimeString()}`;
  $("#jobs").innerHTML=d.jobs.map(j=>{const f=Math.min(1,j.done/j.total);return `<div class="card"><h2><span class="dot ${j.active?"on":""}"></span>${j.name}</h2>
    <div class="big">${j.done}<span>/ ${j.total} clips</span></div>
    <div class="bar"><i class="${j.active?"on":""}" style="width:${(f*100).toFixed(1)}%"></i></div>
    <div class="meta"><span>${(f*100).toFixed(1)}%</span><span>rate <b>${j.rate_per_hour||"—"}</b>/h</span><span>eta <b>${j.active?fmtEta(j.eta_seconds):(j.done>=j.total?"done":"idle")}</b></span><span>audio <b>${fmtAudio(j.audio_seconds)}</b></span></div></div>`}).join("");
  $("#gpus").innerHTML=d.gpus.map(g=>`<div><div style="font-size:13px;margin-bottom:2px">${g.name}</div>${gauge("util",g.util/100,g.util+"%","var(--a)")}${gauge("vram",g.mem_used/g.mem_total,(g.mem_used/1024).toFixed(1)+" GB","var(--b)")}<div class="meta" style="margin-top:4px"><span>${g.temp}°C</span><span>${g.power.toFixed(0)} W</span></div></div>`).join("")||"<span style='color:var(--dim)'>no nvidia-smi</span>";
  const total=d.jobs.reduce((a,j)=>a+j.audio_seconds,0);$("#audio").innerHTML=`${fmtAudio(total)}<span>of synthetic speech</span>`;
  const clips=d.jobs.reduce((a,j)=>a+j.done,0);$("#audiometa").innerHTML=`<span><b>${clips}</b> clips</span><span><b>${(total*12.5|0).toLocaleString()}</b> frames</span><span><b>${(total*12.5*16|0).toLocaleString()}</b> codec tokens</span>`;
  const recent=d.jobs.flatMap(j=>j.recent.map(r=>({...r,job:j.key}))).sort((a,b)=>b.mtime-a.mtime).slice(0,8);
  fingers=recent.flatMap(r=>r.finger||[]);
  $("#recent").innerHTML=recent.map(r=>{const a=r.attrs||{};const chips=Object.entries(a).map(([k,v])=>`<span class="chip ${k=="gender"||k=="age"?"g":k=="mood"?"m":""}">${v}</span>`).join("");
    const sw=(r.finger||[]).slice(0,12).map((v,i)=>`<span class="swatch" style="background:${hue(v,i)}"></span>`).join("");
    return `<div class="clip"><p class="p">${r.prose}</p><p class="t">“${r.text}” — ${r.seconds}s · ${r.job} ${r.id}</p>${chips}<div style="margin-top:6px">${sw}<span style="color:var(--dim);font-size:11px;margin-left:6px;font-family:var(--mono)">layer-14 fingerprint</span></div></div>`}).join("")||"<span style='color:var(--dim)'>nothing yet</span>";
  const t=d.train;
  if(!t){$("#train").textContent="not started"}else if(t.state=="starting"){$("#train").textContent="loading…"}else{
    const f=t.step/t.total;const pts=t.losses;const w=300,h=60;const mx=Math.max(...pts),mn=Math.min(...pts);
    const path=pts.map((v,i)=>`${i?"L":"M"}${(i/(pts.length-1)*w).toFixed(1)},${(h-(v-mn)/(mx-mn+1e-9)*h).toFixed(1)}`).join(" ");
    $("#train").innerHTML=`<div class="big">${t.loss.toFixed(3)}<span>train loss · step ${t.step}/${t.total}</span></div>
      <div class="bar"><i class="${t.state=="training"?"on":""}" style="width:${(f*100).toFixed(1)}%;background:linear-gradient(90deg,var(--c),var(--d))"></i></div>
      <svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><path d="${path}" fill="none" stroke="var(--c)" stroke-width="1.5"/></svg>
      <div class="meta"><span>held-out <b>${t.held.length?t.held[t.held.length-1].toFixed(3):"—"}</b></span><span>prefix gap <b>${t.gap&&t.gap.length?t.gap[t.gap.length-1].toFixed(3):"—"}</b></span><span>${t.minutes.toFixed(1)} min</span><span>${t.state}</span></div>
      <div style="color:var(--dim);font-size:12px;margin-top:8px">which backbone layers it listens to</div>
      <div class="layers">${t.layers.map((v,i)=>`<div style="height:${Math.max(4,v*100)}%" data-l="L${[7,14,21,28][i]}"></div>`).join("")}</div>
      ${t.samples.length?`<div style="margin-top:22px;font-size:12.5px">${t.samples.map(([a,b])=>`<div style="margin:6px 0"><span style="color:var(--dim)">truth</span> ${a}<br><span style="color:var(--d)">model</span> ${b}</div>`).join("")}</div>`:""}`}
  $("#rl").innerHTML=(d.rl||[]).map(r=>{const f=r.total?r.step/r.total:0;const pts=r.rewards;const w=300,h=50;const mx=Math.max(...pts,...r.best),mn=Math.min(...pts,...r.best);
    const path=a=>a.map((v,i)=>`${i?"L":"M"}${(i/(a.length-1||1)*w).toFixed(1)},${(h-(v-mn)/(mx-mn+1e-9)*h).toFixed(1)}`).join(" ");
    const held=r.held.map(([p,q],i)=>`<span>${i==0?"before":"@"+(i*100)} <b>${p.toFixed(3)}</b>${q!=null?" vs "+q.toFixed(3):""}</span>`).join("");
    const last=r.held.length?r.held[r.held.length-1]:null;const delta=last&&last[1]!=null?(last[0]-last[1]):(r.held.length>1?r.held[r.held.length-1][0]-r.held[0][0]:null);
    return `<div class="card wide"><h2><span class="dot ${r.state=="running"?"on":""}"></span>rl · ${r.name} reward</h2>
      <div class="big">${delta==null?"—":(delta>=0?"+":"")+delta.toFixed(3)}<span>held-out policy − reference · step ${r.step}/${r.total} · ${r.state}</span></div>
      <div class="bar"><i class="${r.state=="running"?"on":""}" style="width:${(f*100).toFixed(1)}%;background:linear-gradient(90deg,var(--d),var(--b))"></i></div>
      <svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><path d="${path(r.best)}" fill="none" stroke="var(--dim)" stroke-width="1"/><path d="${path(pts)}" fill="none" stroke="var(--d)" stroke-width="1.5"/></svg>
      <div class="meta"><span>batch reward <b>${pts.length?pts[pts.length-1].toFixed(3):"—"}</b> (grey: best-in-group)</span><span>kl <b>${r.kl.toFixed(3)}</b></span><span>len <b>${r.len}</b></span><span>${r.minutes.toFixed(0)} min</span></div>
      <div class="meta" style="margin-top:6px">${held}</div>
      ${r.sample?`<div style="margin-top:10px;font-size:13px"><span style="color:var(--dim)">sampled</span> ${r.sample}</div>`:""}</div>`}).join("");
  $("#foot").textContent="read-only. refreshes every 2s. the particles take their colours from the newest clip's latent fingerprint, which is meaningless but nice.";
}
async function poll(){try{const r=await fetch("/api/status");render(await r.json())}catch(e){$("#sub").textContent="lost the server: "+e}finally{setTimeout(poll,2000)}}
initSky();poll();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps(status()).encode()
            ctype = "application/json"
        else:
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard on http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
