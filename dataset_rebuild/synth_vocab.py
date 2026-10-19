"""
Vocabulary for synthetic PII values and the field names they are injected under.

The names below are deliberately DISJOINT from the 10 first / 10 last names that
recon_synthesizer.py puts into the (masked) real flows, so a model trained on
synthetic names is tested on names it has never seen -- no value leakage.
"""
from recon_synthesizer import _FIRST_NAMES as _TEST_FIRST, _LAST_NAMES as _TEST_LAST

FIRST_NAMES = [
    "Olivia", "Ethan", "Sofia", "Mohammed", "Hannah", "Lucas", "Ananya", "Daniel", "Chloe", "Ravi",
    "Isabella", "Mateo", "Yuki", "Grace", "Omar", "Zoe", "Arjun", "Natalie", "Diego", "Leila",
    "Samuel", "Mei", "Jack", "Amara", "Ivan", "Sophie", "Kenji", "Layla", "Ben", "Rosa",
    "Tariq", "Nora", "Felix", "Aisha", "Owen", "Camila", "Hiroshi", "Elena", "Dev", "Lily",
    "Andre", "Mina", "Tyler", "Ingrid", "Rahul", "Julia", "Marcus", "Selin", "Victor", "Ayesha",
    "Leo", "Katya", "Jonas", "Nadia", "Eli", "Tara", "Pablo", "Hana", "Dylan", "Zara",
]
LAST_NAMES = [
    "Johnson", "Williams", "Brown", "Jones", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson",
    "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Lopez", "Lee", "Gonzalez", "Clark",
    "Lewis", "Walker", "Hall", "Allen", "Young", "King", "Wright", "Scott", "Torres", "Hill",
    "Adams", "Baker", "Nelson", "Carter", "Mitchell", "Roberts", "Turner", "Phillips", "Campbell", "Parker",
    "Evans", "Edwards", "Collins", "Stewart", "Morris", "Murphy", "Cook", "Rogers", "Reed", "Bailey",
    "Sharma", "Singh", "Ali", "Hassan", "Ivanov", "Tanaka", "Sato", "Park", "Costa", "Ferreira", "Novak",
]
assert not set(FIRST_NAMES) & set(_TEST_FIRST), "synthetic first names overlap the masked-real names"
assert not set(LAST_NAMES) & set(_TEST_LAST), "synthetic last names overlap the masked-real names"

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "aol.com", "proton.me", "live.com", "mail.com", "msn.com",
]
PASSWORD_WORDS = [
    "sunshine", "dragon", "monkey", "football", "princess", "letmein", "welcome", "shadow", "master", "hello",
    "soccer", "tiger", "summer", "flower", "cookie", "ninja", "pepper", "rocket", "orange", "guitar",
]
CITIES = [
    "Austin", "Denver", "Boston", "Seattle", "Portland", "Phoenix", "Atlanta", "Chicago", "Houston", "Dallas",
    "Miami", "Detroit", "Nashville", "Columbus", "Charlotte", "Tampa", "Orlando", "Madison", "Omaha", "Tulsa",
    "Reno", "Boise", "Raleigh", "Memphis", "Richmond", "Albany", "Toledo", "Fresno", "Tucson", "Mesa",
]

# field names used when a value is injected into a URL / JSON / form / header
KEYS = {
    "email address": ["email", "user_email", "e_mail", "emailAddress", "mail", "login_email", "account_email", "contact_email"],
    "password": ["password", "passwd", "pwd", "pass", "user_password", "login_password", "secret"],
    "phone number": ["phone", "phone_number", "phoneNumber", "mobile", "msisdn", "tel", "contact_number", "user_phone"],
    "imsi number": ["imsi", "subscriber_id", "imsi_number", "sim_imsi", "IMSI"],
    "sim card iccid": ["iccid", "sim_serial", "sim_serial_number", "simSerial", "iccid_number", "ICCID"],
    "gender": ["gender", "sex", "user_gender", "gender_code"],
    "last name": ["last_name", "lastname", "lname", "surname", "family_name", "lastName"],
    "first name": ["first_name", "firstname", "fname", "given_name", "firstName", "name"],
    "device serial number": ["serial", "serial_number", "serialno", "device_serial", "ro.serialno", "sn", "serialNumber"],
    "imei number": ["imei", "device_imei", "imei_number", "IMEI", "imei1"],
    "android device id": ["android_id", "androidId", "aid"],
    "advertiser id": ["advertising_id", "adid", "gaid", "idfa", "ad_id"],
    "username": ["username", "user_name", "login", "uname", "user"],
    "zip code": ["zip", "zipcode", "postal_code", "postcode"],
    "city": ["city", "town", "user_city"],
    "location coordinates": ["latlng", "location", "coords", "geo"],
}
# neutral names: force the model to use the value's shape, not just the key
NEUTRAL_KEYS = ["id", "uid", "value", "data", "token", "key", "ref", "param", "info", "u", "v", "x", "payload"]
NESTING_KEYS = ["user", "device", "profile", "data", "account", "params", "info"]
