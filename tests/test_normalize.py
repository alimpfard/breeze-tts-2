"""Text normalization for forms the model reads badly.

Each case here corresponds to an observed failure: currency symbols were
dropped silently, and "18.5%" came out as "18 bind 5 percent".
"""

import pytest

from breeze_infer.normalize import normalize_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Currency symbols were dropped entirely by the model.
        ("$1,247.50", "1,247 dollars and 50 cents"),
        ("£1,478", "1,478 pounds"),
        ("$1", "1 dollar"),
        ("$1.00", "1 dollar"),
        ("$3.00", "3 dollars"),
        ("£5.05", "5 pounds and 5 pence"),
        ("€12.34", "12 euros and 34 cents"),
        # Decimal points were mis-tokenised.
        ("18.5%", "18 point 5 percent"),
        ("50%", "50 percent"),
        ("0.5%", "0 point 5 percent"),
        ("26.2 miles", "26 point 2 miles"),
    ],
)
def test_rewrites_symbols_and_decimals(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        # These already read correctly and must not be touched: more than one
        # separator, or non-decimal punctuation.
        "Version 2.11.0",
        "192.168.3.133",
        "3:47 PM",
        "03/04/2027",
        "Dr. Meade arrives, i.e. eventually.",
        "No numbers here at all.",
    ],
)
def test_leaves_working_forms_alone(raw):
    assert normalize_text(raw) == raw


def test_currency_and_percent_in_one_sentence():
    raw = "It came to $1,247.50 plus 18.5% VAT, or £1,478 all in."
    assert normalize_text(raw) == (
        "It came to 1,247 dollars and 50 cents plus 18 point 5 percent VAT, "
        "or 1,478 pounds all in."
    )
