"""
Generates a realistic-looking synthetic value for a given PII label, used to
replace the dataset's literal "RECON_<Type>" redaction tokens.

Why this exists: the public AntShield release redacts every real PII value
with a sentinel string that spells out its own type, e.g. "RECON_AndroidId",
"RECON_Email". If you feed that text to a model as-is, the model doesn't need
to learn any real pattern -- it can just pattern-match the literal substring
"AndroidId" inside the token. That makes both zero-shot and fine-tuned
results meaningless as a measure of actual PII-detection capability (it also
retroactively means your binary BERT classifier could have been trivially
keying off the literal string "RECON_" as its positive-class signal, since
that prefix is present in 100% of positive packets and 0% of negative ones --
worth rechecking that model's learned weights/attention if you want to trust
those numbers).

Only stdlib is used (no Faker dependency) -- good enough for this purpose,
not meant to be demographically realistic.
"""
import random
import string

_FIRST_NAMES = ["James", "Maria", "Wei", "Fatima", "Liam", "Aiko", "Noah", "Priya", "Carlos", "Emma"]
_LAST_NAMES = ["Smith", "Garcia", "Chen", "Khan", "Muller", "Silva", "Kim", "Patel", "Rossi", "Nguyen"]
_CITIES = ["Springfield", "Riverside", "Fairview", "Georgetown", "Clinton", "Salem", "Franklin", "Greenville"]
_GENDERS = ["male", "female"]


def _rng_for(seed_key: str) -> random.Random:
    # deterministic per (record, occurrence) so repeated runs are reproducible,
    # but different occurrences of the same type in one record still differ
    return random.Random(seed_key)


def synthesize(gliner_label: str, seed_key: str) -> str:
    rng = _rng_for(seed_key)

    if gliner_label == "android device id":
        return "".join(rng.choices(string.hexdigits.lower()[:16], k=16))
    if gliner_label == "imei number":
        return "".join(rng.choices(string.digits, k=15))
    if gliner_label == "imsi number":
        return "".join(rng.choices(string.digits, k=15))
    if gliner_label == "device serial number":
        return "".join(rng.choices(string.ascii_uppercase + string.digits, k=10))
    if gliner_label == "sim card iccid":
        return "89" + "".join(rng.choices(string.digits, k=17))
    if gliner_label == "advertiser id":
        # uuid4-shaped, not a real uuid module call so it stays seeded
        hexd = "".join(rng.choices(string.hexdigits.lower()[:16], k=32))
        return f"{hexd[0:8]}-{hexd[8:12]}-{hexd[12:16]}-{hexd[16:20]}-{hexd[20:32]}"
    if gliner_label == "phone number":
        return f"+1{rng.randint(200,999)}{rng.randint(200,999)}{rng.randint(1000,9999)}"
    if gliner_label == "email address":
        user = "".join(rng.choices(string.ascii_lowercase, k=8))
        domain = rng.choice(["gmail.com", "yahoo.com", "outlook.com"])
        return f"{user}@{domain}"
    if gliner_label == "username":
        return "user_" + "".join(rng.choices(string.digits, k=5))
    if gliner_label == "password":
        return "".join(rng.choices(string.ascii_letters + string.digits + "!@#$", k=10))
    if gliner_label == "first name":
        return rng.choice(_FIRST_NAMES)
    if gliner_label == "last name":
        return rng.choice(_LAST_NAMES)
    if gliner_label == "gender":
        return rng.choice(_GENDERS)
    if gliner_label == "zip code":
        return "".join(rng.choices(string.digits, k=5))
    if gliner_label == "city":
        return rng.choice(_CITIES)
    if gliner_label == "location coordinates":
        lat = round(rng.uniform(-90, 90), 4)
        lon = round(rng.uniform(-180, 180), 4)
        return f"{lat},{lon}"

    # "other personal information" or anything unmapped
    return "".join(rng.choices(string.ascii_letters + string.digits, k=8))
