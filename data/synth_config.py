"""
Editable policy for the rebuilt AntShield dataset (labels are the GLiNER label
strings from pii_type_mapping.py).

Real-data counts that motivated these choices (audit, raw span counts):
  Gender 5, ICCID 5, LastName 4, Password 2, PhoneNumber 2, Email 32, IMSI 36
  FirstName 102, SerialNumber 116.  Everything else has 200+ spans.

Policies
  SYNTHETIC_ONLY : every real flow containing the class goes to TEST; the class
                   appears in TRAIN only as synthetic data.  (With 2-36 real
                   examples, a 20% real share would be 0-7 examples: pointless.)
  SPLIT_20_80    : ~20% of the real flows (grouped by app package) go to TRAIN,
                   ~80% to TEST; synthetic data tops TRAIN up to the target.
  (anything else): "campaign" -- the original rule, auto_anteater -> train,
                   manual_anteater -> test, no synthetic data.
Run `build_dataset.py --stats-only` first; move a label between sets if the
class x campaign table shows a class is starved in train.
"""
SYNTHETIC_ONLY = {
    "gender",
    "sim card iccid",
    "last name",
    "password",
    "phone number",
    "email address",
    "imsi number",
}
SPLIT_20_80 = {
    "first name",
    "device serial number",
}
REAL_TRAIN_FRACTION = 0.20

# synthetic top-up: raise every class in the two sets above to this many TRAIN
# windows containing the class (real + synthetic)
TARGET_TRAIN_WINDOWS = 300

# classes that may appear together in one synthetic record (login forms,
# identity forms, SIM/device info) -- only classes in the sets above are used
COOCCUR_GROUPS = [
    ("email address", "password", "first name", "last name"),
    ("first name", "last name", "gender"),
    ("imsi number", "sim card iccid", "phone number"),
]
