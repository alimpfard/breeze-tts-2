"""Minimal text normalization for things the model reads badly.

Breeze reads text directly through its text encoder with no frontend, so a few
common written forms come out wrong. Observed failures:

    "$1,247.50"  -> currency symbol silently dropped
    "£1,478"     -> same
    "18.5%"      -> "18 bind 5 percent"  (decimal point mis-tokenised)

Bare digits and grouped thousands read correctly, so this rewrites only the
symbols and decimal points rather than spelling out whole numbers. Keeping the
digits intact avoids inventing a number-to-words implementation and its own
class of mistakes.

Deliberately conservative: times (3:47), dates (03/04/2027) and version-like
strings already read correctly and are left alone.
"""

from __future__ import annotations

import re

# Symbol -> (singular, plural). Plural is used unless the amount is exactly one.
CURRENCY = {
    "$": ("dollar", "dollars"),
    "£": ("pound", "pounds"),
    "€": ("euro", "euros"),
    "¥": ("yen", "yen"),
}

_CURRENCY_RE = re.compile(
    r"(?P<symbol>[$£€¥])\s?(?P<amount>\d[\d,]*(?:\.\d+)?)"
)
_PERCENT_RE = re.compile(r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s?%")
# A decimal point between digits, not part of a date or version (which have
# more than one separator, or non-digit neighbours).
_DECIMAL_RE = re.compile(r"(?<![\d.])(\d[\d,]*)\.(\d+)(?![\d.])")


def _is_one(amount: str) -> bool:
    return amount.replace(",", "") in {"1", "1.0", "1.00"}


# Minor units, for amounts written with exactly two decimal places.
MINOR_UNIT = {"$": "cent", "£": "penny", "€": "cent", "¥": None}
MINOR_PLURAL = {"cent": "cents", "penny": "pence"}


def _currency(match: re.Match) -> str:
    symbol = match["symbol"]
    singular, plural = CURRENCY[symbol]
    amount = match["amount"]

    whole, _, frac = amount.partition(".")
    minor = MINOR_UNIT.get(symbol)

    # Two decimals is money written with minor units: say them as such rather
    # than as "point fifty", which reads wrong.
    if minor and len(frac) == 2:
        unit = singular if _is_one(whole) else plural
        if int(frac) == 0:
            return f"{whole} {unit}"
        minor_unit = minor if int(frac) == 1 else MINOR_PLURAL[minor]
        # Strip a leading zero so "05" reads as five, not oh-five.
        return f"{whole} {unit} and {int(frac)} {minor_unit}"

    unit = singular if _is_one(amount) else plural
    return f"{amount} {unit}"


def _percent(match: re.Match) -> str:
    return f"{match['amount']} percent"


def _decimal(match: re.Match) -> str:
    return f"{match[1]} point {match[2]}"


def normalize_text(text: str) -> str:
    """Rewrite symbols and decimals into forms the model pronounces correctly."""
    text = _CURRENCY_RE.sub(_currency, text)
    text = _PERCENT_RE.sub(_percent, text)
    # After the above, so "18.5%" becomes "18.5 percent" then "18 point 5 percent".
    text = _DECIMAL_RE.sub(_decimal, text)
    return text
