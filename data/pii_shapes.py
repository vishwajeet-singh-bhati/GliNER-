"""
Realistic-shape value generators used to mask the dataset's RECON_<Type> tokens
(see recon_synthesizer.py).  Stdlib only; imports nothing from this repo.

Why this exists (measured on the real AntShield data, see AUDIT notes):
  * 84.5% of the 10,376 genuine, UNLABELED UUID-shaped strings in the data are
    RFC-4122 version-4 UUIDs (version nibble '4', variant nibble 8/9/a/b), while the
    old masker produced random hex, so only ~1.4% of masked advertiser ids were v4-shaped.
    A model could separate "labelled" from "unlabelled" UUIDs by that one character:
    a shortcut that inflates scores here and fails on real traffic.  -> uuid4().
  * IMEI and IMSI were both "15 random digits", i.e. indistinguishable by value.
    Real IMEIs are Luhn-valid and start with a Type Allocation Code; real IMSIs start
    with MCC+MNC.  -> imei(), imsi(), iccid().
  * Location tokens sit under keys such as "lat" / "lon" / "lng" (a single number)
    as well as "ll" / "latlng" (a "lat,lon" pair).  -> location().

The TAC / MCC-MNC / ICCID lists here are deliberately DIFFERENT from the ones in
synth_values.py (used for synthetic TRAIN data), so train and test never share
concrete values even though both now have realistic shapes.
"""
import random
import re
import string

_HEX = "0123456789abcdef"
_DIG = string.digits

# value-level disjoint from synth_values._TACS / _MCC_MNC / _ICCID_CC
_TACS = ["35290611", "35393108", "35517909", "35695208", "35846306", "86441503", "86892704", "99000112"]
_MCC_MNC = ["310120", "310240", "311870", "234200", "404110", "262022", "208020", "460011", "440050", "505002"]
_ICCID_CC = ["03", "93", "46", "48", "32", "85", "82", "62", "35", "53"]


def luhn_check_digit(payload: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(payload)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choices(_DIG, k=n))


def uuid4(rng: random.Random) -> str:
    """RFC-4122 version-4 shaped UUID (what Google advertising ids look like)."""
    h = "".join(rng.choices(_HEX, k=32))
    return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-{rng.choice('89ab')}{h[17:20]}-{h[20:]}"


def imei(rng: random.Random) -> str:
    body = rng.choice(_TACS) + digits(rng, 6)
    return body + luhn_check_digit(body)


def imsi(rng: random.Random) -> str:
    return rng.choice(_MCC_MNC) + digits(rng, 9)


def iccid(rng: random.Random) -> str:
    body = "89" + rng.choice(_ICCID_CC) + digits(rng, 14)       # 18 digits + Luhn = 19
    return body + luhn_check_digit(body)


_LAT_KEY = re.compile(r"(?<![a-z])(?:latitude|lat)\W{0,6}$")
_LON_KEY = re.compile(r"(?<![a-z])(?:longitude|lng|lon|long)\W{0,6}$")


def location(rng: random.Random, context: str = "") -> str:
    """Single latitude / longitude when the text just before the token is a lat/lon key,
    otherwise a 'lat,lon' pair.  `context` = the (lowercased) text preceding the token."""
    ctx = context.lower()[-24:]
    dec = rng.randint(4, 6)
    if _LAT_KEY.search(ctx):
        return f"{rng.uniform(-60, 70):.{dec}f}"
    if _LON_KEY.search(ctx):
        return f"{rng.uniform(-170, 170):.{dec}f}"
    return f"{rng.uniform(-60, 70):.{dec}f},{rng.uniform(-170, 170):.{dec}f}"
