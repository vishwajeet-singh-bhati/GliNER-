"""
Real pii_types vocabulary, confirmed by scanning the actual
antshield_public_dataset.zip (47,076 records / 8,579 positive across
raw_data/**/*.json):

  AdvertiserId(6217) Location(3380) AndroidId(3314) IMEI(424) City(348)
  Zipcode(313) Username(227) SerialNumber(116) FirstName(101) IMSI(36)
  Email(31) ICCID(5) Gender(4) LastName(4) Password(2) PhoneNumber(2) Other(1)

This replaces the earlier guessed vocabulary (which had MACAddress,
Latitude/Longitude as separate types, and "Phone" instead of "PhoneNumber" --
none of those appear in the real data).
"""

ANTSHIELD_TO_GLINER_LABEL = {
    "AndroidId": "android device id",
    "IMEI": "imei number",
    "IMSI": "imsi number",
    "SerialNumber": "device serial number",
    "ICCID": "sim card iccid",
    "AdvertiserId": "advertiser id",
    "PhoneNumber": "phone number",
    "Email": "email address",
    "Username": "username",
    "Password": "password",
    "FirstName": "first name",
    "LastName": "last name",
    "Gender": "gender",
    "Zipcode": "zip code",
    "City": "city",
    "Location": "location coordinates",
    "Other": "other personal information",
}

# sorted longest-first: matters for regex alternation so a longer type name
# can't be short-circuited by a shorter one that happens to be a prefix
RAW_TYPES_BY_LENGTH = sorted(ANTSHIELD_TO_GLINER_LABEL.keys(), key=len, reverse=True)

import re as _re


def build_recon_pattern():
    """
    Matches a full RECON_<Type> redaction token INCLUDING any trailing
    junk AntShield's own tooling appended to it -- confirmed by direct scan:
    ~5,300 tokens have suffixes like "RECON_Location841329" (digit suffix,
    5105 of these) or "RECON_AdvertiserId___Y29tLnFpaG9vLnNlY3VyaXR5..."
    (base64-ish device string). Absorbing the whole thing as one entity span
    is the correct call, not a bug to route around: that suffix occupies the
    space the real PII value used to sit in, in AntShield's own scheme.

    The (?!RECON_) guard matters for the ~57 records where two redactions
    are concatenated with zero separator, e.g.
    "RECON_SerialNumberRECON_AndroidId" -- without the guard, greedy suffix
    consumption would eat straight through the second token's "RECON_"
    prefix and merge both into one garbage match (this is exactly the bug
    in the reference BERT preprocessing script's plain
    `RECON_[A-Za-z0-9]*`, which leaves a bare "_AndroidId" behind, minus its
    RECON_ prefix, in 57 real records).
    """
    types = "|".join(RAW_TYPES_BY_LENGTH)
    return _re.compile(rf"RECON_(?:{types})(?:(?!RECON_)[A-Za-z0-9_])*")


def extract_recon_label(token: str) -> str:
    """Given a full matched token (e.g. 'RECON_Location841329'), returns the
    GLiNER label for its base type."""
    for raw_type in RAW_TYPES_BY_LENGTH:
        if token.startswith(f"RECON_{raw_type}"):
            return ANTSHIELD_TO_GLINER_LABEL[raw_type]
    raise KeyError(f"Could not extract a known type from token '{token}'")


def normalize_type(raw_type: str) -> str:
    if raw_type in ANTSHIELD_TO_GLINER_LABEL:
        return ANTSHIELD_TO_GLINER_LABEL[raw_type]
    for k, v in ANTSHIELD_TO_GLINER_LABEL.items():
        if k.lower() == raw_type.strip().lower():
            return v
    raise KeyError(
        f"Unrecognized AntShield pii_type '{raw_type}'. "
        f"Add it to ANTSHIELD_TO_GLINER_LABEL in pii_type_mapping.py."
    )


def all_gliner_labels():
    return sorted(set(ANTSHIELD_TO_GLINER_LABEL.values()))
