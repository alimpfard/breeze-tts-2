"""Track X-Conditioning header growth across a conversation.

nginx's default proxy_buffer_size is 4k (8k on some builds). If the upstream
response headers exceed it, nginx answers 502 without ever reaching the client
-- which looks like a server crash but is purely a header-size limit.
"""

from __future__ import annotations

import argparse

import requests

TURNS = [
    "Nobody was there at all, not one soul.",
    "He spread his hands.",
    "“Your kingdom is beautiful.",
    "Rich in potential, But potential does not reduce carriage strain.”",
    "The king saw us then, and said nothing for a long moment.",
    "It was the kind of silence that costs money.",
]

NGINX_DEFAULT_BUFFER = 4096


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9880")
    args = parser.parse_args()

    conditioning = ""
    print(f"{'turn':>4} {'audio':>6} {'X-Conditioning':>15} {'all headers':>12}  status")
    for index, text in enumerate(TURNS):
        payload = {"text": text, "speaker_wav": "default"}
        if conditioning:
            payload["conditioning"] = conditioning

        response = requests.post(f"{args.url}/tts_to_audio", json=payload, timeout=600)
        blob = response.headers.get("X-Conditioning", "")
        header_bytes = sum(len(k) + len(v) + 4 for k, v in response.headers.items())
        conditioning = blob

        verdict = "OK" if header_bytes < NGINX_DEFAULT_BUFFER else "EXCEEDS 4k -> 502"
        print(
            f"{index:>4} {len(response.content) / 48000:>5.1f}s "
            f"{len(blob):>15,} {header_bytes:>12,}  {verdict}"
        )


if __name__ == "__main__":
    main()
