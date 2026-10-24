"""
Synthetic PII value generators + non-PII decoy fields.

Design notes
 * Values come from richer distributions than recon_synthesizer.synthesize (which
   is what the masked REAL flows use), so train and test values are not drawn
   from the same generator.
 * recon_synthesizer does not Luhn-check IMEI/ICCID/IMSI, so ~25% of the numeric
   IDs here are also non-Luhn / random: the model must not key on checksum validity.
 * Decoys look like injected fields but are not PII (timestamps, hashes, versions).
   Their shapes avoid every PII shape (15-digit, 19/20-digit, 16-hex, email- or
   phone-like values), with ONE deliberate exception: UUID-shaped values under
   explicit non-advertiser keys (session_id, request_id, ...).  Real traffic is full
   of such UUIDs and they are NOT advertising ids; without these hard negatives the
   model learns "any UUID next to another UUID is an advertiser id".  The keys never
   overlap with KEYS['advertiser id'] or NEUTRAL_KEYS (which PII values also use).
"""
import random
import string

from pii_shapes import uuid4
from synth_vocab import CITIES, EMAIL_DOMAINS, FIRST_NAMES, LAST_NAMES, PASSWORD_WORDS

_UPPER_ALNUM = string.ascii_uppercase + string.digits


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


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choices(string.digits, k=n))


_TACS = ["35328709", "35391806", "35456910", "86012303", "35824005", "35209008", "86738402", "35276806"]
_MCC_MNC = ["310260", "310410", "311480", "234150", "404450", "262011", "208010", "460001", "440010", "505001"]
_ICCID_CC = ["01", "91", "44", "49", "33", "86", "81", "61", "34", "52"]


def imei(rng):
    if rng.random() < 0.25:
        return _digits(rng, 15)
    body = rng.choice(_TACS) + _digits(rng, 6)
    return body + luhn_check_digit(body)


def imsi(rng):
    if rng.random() < 0.25:
        return _digits(rng, 15)
    return rng.choice(_MCC_MNC) + _digits(rng, 9)


def iccid(rng):
    length = rng.choice([19, 20])
    body = "89" + rng.choice(_ICCID_CC) + _digits(rng, length - 5)
    return body + (luhn_check_digit(body) if rng.random() < 0.8 else _digits(rng, 1))


def phone(rng):
    a, b, c = rng.randint(200, 999), rng.randint(200, 999), rng.randint(1000, 9999)
    style = rng.choice([0, 0, 1, 2, 3, 4, 5])
    if style == 0:
        return f"+1{a}{b}{c}"
    if style == 1:
        return f"1{a}{b}{c}"
    if style == 2:
        return f"{a}-{b}-{c}"
    if style == 3:
        return f"+91{rng.randint(6, 9)}{rng.randint(0, 10**9 - 1):09d}"
    if style == 4:
        return f"{a}{b}{c}"
    return f"+447{rng.randint(0, 10**9 - 1):09d}"


def email(rng):
    f, l = rng.choice(FIRST_NAMES).lower(), rng.choice(LAST_NAMES).lower()
    local = [
        f"{f}.{l}", f"{f}_{l}", f"{f}{l}{rng.randint(1, 99)}",
        f"{f[0]}{l}{rng.randint(70, 99)}", f"{f}{rng.randint(100, 9999)}",
    ][rng.randrange(5)]
    return f"{local}@{rng.choice(EMAIL_DOMAINS)}"


def password(rng):
    style = rng.randrange(4)
    if style == 0:
        return rng.choice(PASSWORD_WORDS) + str(rng.randint(1, 9999)) + rng.choice("!@#$%")
    if style == 1:
        return rng.choice(PASSWORD_WORDS).capitalize() + str(rng.randint(1970, 2024))
    if style == 2:
        return "".join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(8, 14)))
    return "".join(rng.choices(string.ascii_letters + string.digits + "!@#$%^&*", k=rng.randint(8, 16)))


def serial(rng):
    style = rng.randrange(4)
    if style == 0:
        return "".join(rng.choices(_UPPER_ALNUM, k=rng.randint(8, 12)))
    if style == 1:
        return rng.choice(["R5", "R3", "RF", "RZ", "ZY"]) + "".join(rng.choices(_UPPER_ALNUM, k=rng.randint(8, 10)))
    if style == 2:
        return "".join(rng.choices("0123456789ABCDEF", k=rng.choice([12, 16])))
    return "".join(rng.choices(string.ascii_uppercase, k=rng.randint(2, 4))) + _digits(rng, rng.randint(6, 9))


def first_name(rng):
    n = rng.choice(FIRST_NAMES)
    return n.lower() if rng.random() < 0.15 else n


def last_name(rng):
    n = rng.choice(LAST_NAMES)
    return n.lower() if rng.random() < 0.15 else n


def gender(rng):
    return rng.choice(["male", "female", "Male", "Female", "M", "F", "m", "f", "male", "female"])


def android_id(rng):
    return "".join(rng.choices("0123456789abcdef", k=16))


def advertiser_id(rng):
    h = "".join(rng.choices("0123456789abcdef", k=32))
    return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-{rng.choice('89ab')}{h[17:20]}-{h[20:]}"


def username(rng):
    style = rng.randrange(3)
    if style == 0:
        return rng.choice(FIRST_NAMES).lower() + rng.choice(["_", "", "."]) + rng.choice(LAST_NAMES).lower()
    if style == 1:
        return rng.choice(PASSWORD_WORDS) + "_" + _digits(rng, rng.randint(2, 4))
    return rng.choice(FIRST_NAMES).lower() + _digits(rng, rng.randint(2, 4))


def zip_code(rng):
    return _digits(rng, 5)


def city(rng):
    return rng.choice(CITIES)


def location(rng):
    return f"{rng.uniform(-60, 70):.{rng.randint(4, 6)}f},{rng.uniform(-170, 170):.{rng.randint(4, 6)}f}"


GENERATORS = {
    "imei number": imei, "imsi number": imsi, "sim card iccid": iccid, "phone number": phone,
    "email address": email, "password": password, "device serial number": serial,
    "first name": first_name, "last name": last_name, "gender": gender,
    "android device id": android_id, "advertiser id": advertiser_id, "username": username,
    "zip code": zip_code, "city": city, "location coordinates": location,
}


def generate_value(label: str, rng: random.Random) -> str:
    return GENERATORS[label](rng)


UUID_DECOY_KEYS = ["session_id", "request_id", "trace_id", "event_id", "transaction_id",
                   "correlation_id", "sessionid", "requestid", "traceid", "batch_id", "visit_id"]


def decoy_field(rng: random.Random):
    """(key, value) of a plausible non-PII field. Shapes chosen to avoid every PII shape."""
    kind = rng.randrange(13)
    if kind >= 9:   # 4 of 13 draws: UUID-shaped non-PII ids
        key = rng.choice(UUID_DECOY_KEYS)
        val = uuid4(rng)
        if rng.random() < 0.25:
            val = val.upper()
        return key, val
    if kind == 0:
        return "ts", str(rng.randint(1_500_000_000_000, 1_700_000_000_000))
    if kind == 1:
        return "timestamp", str(rng.randint(1_500_000_000, 1_700_000_000))
    if kind == 2:
        return "session", "".join(rng.choices("0123456789abcdef", k=24))
    if kind == 3:
        return "sig", "".join(rng.choices("0123456789abcdef", k=40))
    if kind == 4:
        return "order_id", _digits(rng, rng.choice([9, 10, 11, 12]))
    if kind == 5:
        return "ver", f"{rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 99)}"
    if kind == 6:
        return "locale", rng.choice(["en_US", "en-GB", "es_ES", "de_DE", "fr_FR", "pt_BR"])
    if kind == 7:
        return "screen", rng.choice(["1080x1920", "1080x2220", "720x1280", "1440x2960"])
    return "nonce", "".join(rng.choices(string.ascii_letters + string.digits, k=22))
