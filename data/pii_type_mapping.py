"""
Maps AntShield's internal pii_types strings to natural-language labels that
GLiNER expects (GLiNER labels are free-text descriptions, not enum codes).

The left-hand keys are AntShield's documented type strings. If your local
copy of the dataset uses different capitalization/spelling, add aliases here
rather than touching the converter.
"""

ANTSHIELD_TO_GLINER_LABEL = {
    "AndroidId": "android device id",
    "IMEI": "imei number",
    "IMSI": "imsi number",
    "SerialNumber": "device serial number",
    "ICCID": "sim card iccid",
    "MACAddress": "mac address",
    "AdvertiserId": "advertiser id",
    "Phone": "phone number",
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
    "Latitude": "latitude",
    "Longitude": "longitude",
}


def normalize_type(raw_type: str) -> str:
    """Best-effort normalization for minor casing/spacing differences."""
    key = raw_type.strip()
    if key in ANTSHIELD_TO_GLINER_LABEL:
        return ANTSHIELD_TO_GLINER_LABEL[key]
    # fallback: try case-insensitive match
    for k, v in ANTSHIELD_TO_GLINER_LABEL.items():
        if k.lower() == key.lower():
            return v
    raise KeyError(
        f"Unrecognized AntShield pii_type '{raw_type}'. "
        f"Add it to ANTSHIELD_TO_GLINER_LABEL in pii_type_mapping.py."
    )


def all_gliner_labels():
    """Full label set to pass to GLiNER's predict_entities(text, labels)."""
    return sorted(set(ANTSHIELD_TO_GLINER_LABEL.values()))
