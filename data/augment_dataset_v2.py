import json
import random
import string
from recon_synthesizer import synthesize

random.seed(42)

NETWORK_TEMPLATES = [
    lambda idx: _http_cookie(idx),
    lambda idx: _url_query(idx),
    lambda idx: _log_line(idx),
    lambda idx: _telemetry_blob(idx),
    lambda idx: _form_post(idx),
    lambda idx: _tracker_sdk(idx),
]

def _rand_path():
    segs = random.randint(1, 3)
    return "/".join(
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        for _ in range(segs)
    )

def _rand_key():
    prefixes = ["x_", "p", "q", "k_", "v_", "d", "h_", "m", "uid_", "sid_", "tid_"]
    return random.choice(prefixes) + "".join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(2, 5)))

def _http_cookie(idx):
    adv_id = synthesize("advertiser id", f"adv_{idx}")
    dev_id = synthesize("android device id", f"dev_{idx}")
    email = synthesize("email address", f"email_{idx}")
    session = "".join(random.choices(string.hexdigits.lower()[:16], k=24))

    k1, k2, k3 = _rand_key(), _rand_key(), _rand_key()
    tokens = [
        "Cookie:", f"{k1}=", adv_id, ";", f"{k2}=", dev_id, ";",
        f"{k3}=", email, ";", f"sess={session};", "Path=/;", "Secure"
    ]
    ner = [
        [2, 2, "advertiser id"],
        [5, 5, "android device id"],
        [8, 8, "email address"],
    ]
    return {"tokenized_text": tokens, "ner": ner}

def _url_query(idx):
    phone = synthesize("phone number", f"phone_{idx}")
    coords = synthesize("location coordinates", f"loc_{idx}")
    imei = synthesize("imei number", f"imei_{idx}")

    path = _rand_path()
    k1, k2, k3 = _rand_key(), _rand_key(), _rand_key()
    tokens = [
        "GET", f"/{path}?",
        f"{k1}=", phone, "&",
        f"{k2}=", coords, "&",
        f"{k3}=", imei,
        "HTTP/1.1", "Host:", "api." + "".join(random.choices(string.ascii_lowercase, k=6)) + ".com"
    ]
    ner = [
        [3, 3, "phone number"],
        [6, 6, "location coordinates"],
        [9, 9, "imei number"],
    ]
    return {"tokenized_text": tokens, "ner": ner}

def _log_line(idx):
    fname = synthesize("first name", f"fn_{idx}")
    lname = synthesize("last name", f"ln_{idx}")
    email = synthesize("email address", f"em_{idx}")
    ip = f"{random.randint(10,200)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    ts = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}Z"

    tokens = [
        f"[{ts}]", "INFO", "auth.handler", "-",
        "login_success", f"src={ip}", "user=", fname, lname,
        "contact=", email, "session_start"
    ]
    ner = [
        [7, 7, "first name"],
        [8, 8, "last name"],
        [10, 10, "email address"],
    ]
    return {"tokenized_text": tokens, "ner": ner}

def _telemetry_blob(idx):
    adv_id = synthesize("advertiser id", f"tel_adv_{idx}")
    serial = synthesize("device serial number", f"tel_ser_{idx}")
    imsi = synthesize("imsi number", f"tel_imsi_{idx}")
    gender = synthesize("gender", f"tel_g_{idx}")
    zipcode = synthesize("zip code", f"tel_z_{idx}")

    k1, k2, k3, k4, k5 = [_rand_key() for _ in range(5)]
    tokens = [
        "{", f'"{k1}":', f'"{adv_id}",',
        f'"{k2}":', f'"{serial}",',
        f'"{k3}":', f'"{imsi}",',
        f'"{k4}":', f'"{gender}",',
        f'"{k5}":', f'"{zipcode}"',
        "}"
    ]
    ner = [
        [2, 2, "advertiser id"],
        [4, 4, "device serial number"],
        [6, 6, "imsi number"],
        [8, 8, "gender"],
        [10, 10, "zip code"],
    ]
    return {"tokenized_text": tokens, "ner": ner}

def _form_post(idx):
    username = synthesize("username", f"form_u_{idx}")
    password = synthesize("password", f"form_p_{idx}")
    phone = synthesize("phone number", f"form_ph_{idx}")
    city = synthesize("city", f"form_c_{idx}")

    k1, k2, k3, k4 = [_rand_key() for _ in range(4)]
    tokens = [
        "POST", "/submit", "HTTP/1.1",
        "Content-Type:", "application/x-www-form-urlencoded", "\r\n",
        f"{k1}=", username, "&",
        f"{k2}=", password, "&",
        f"{k3}=", phone, "&",
        f"{k4}=", city
    ]
    ner = [
        [7, 7, "username"],
        [10, 10, "password"],
        [13, 13, "phone number"],
        [16, 16, "city"],
    ]
    return {"tokenized_text": tokens, "ner": ner}

def _tracker_sdk(idx):
    adv_id = synthesize("advertiser id", f"sdk_a_{idx}")
    dev_id = synthesize("android device id", f"sdk_d_{idx}")
    coords = synthesize("location coordinates", f"sdk_l_{idx}")
    iccid = synthesize("sim card iccid", f"sdk_s_{idx}")

    noise1 = "".join(random.choices(string.hexdigits.lower()[:16], k=16))
    noise2 = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    tokens = [
        f"v=2&sdk=4.1.{random.randint(0,9)}&ts={random.randint(1700000000,1710000000)}",
        f"&h={noise1}&", adv_id,
        f"&z={noise2}&", dev_id,
        "&", coords,
        "&c=", iccid
    ]
    ner = [
        [2, 2, "advertiser id"],
        [4, 4, "android device id"],
        [6, 6, "location coordinates"],
        [8, 8, "sim card iccid"],
    ]
    return {"tokenized_text": tokens, "ner": ner}

def generate_hard_negative(idx):
    rng = random.Random(f"neg_{idx}")
    templates = [
        {
            "tokenized_text": [
                "txn_id=", f"{rng.randbytes(4).hex()}-{rng.randbytes(2).hex()}-{rng.randbytes(2).hex()}-{rng.randbytes(2).hex()}-{rng.randbytes(6).hex()}",
                "status=completed", "amount=49.99"
            ],
            "ner": []
        },
        {
            "tokenized_text": [
                "Content-MD5:", "".join(rng.choices(string.hexdigits.lower()[:16], k=32)),
                "ETag:", f'"{"".join(rng.choices(string.hexdigits.lower()[:16], k=16))}"'
            ],
            "ner": []
        },
        {
            "tokenized_text": [
                "order", f"#{rng.randint(1000000000, 9999999999)}",
                "shipped", "via", "FedEx",
                "tracking", f"{rng.randint(100000000000, 999999999999)}"
            ],
            "ner": []
        },
        {
            "tokenized_text": [
                "Springfield", "Armory", "XD-M", "Elite", "9mm",
                "SKU:", f"SA{rng.randint(10000,99999)}"
            ],
            "ner": []
        },
    ]
    return rng.choice(templates)


def main():
    count_per_template = 150
    hard_neg_count = 300

    windows = []
    for template_fn in NETWORK_TEMPLATES:
        for i in range(count_per_template):
            try:
                w = template_fn(len(windows))
                windows.append(w)
            except Exception as e:
                pass

    for i in range(hard_neg_count):
        windows.append(generate_hard_negative(i))

    random.shuffle(windows)

    with open('data/train_finetune.jsonl', 'a', encoding='utf-8') as f:
        for w in windows:
            f.write(json.dumps(w, ensure_ascii=False) + '\n')

    print(f"✅ Appended {len(windows)} windows to data/train_finetune.jsonl")

if __name__ == "__main__":
    main()
