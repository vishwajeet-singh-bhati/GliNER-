import json
from recon_synthesizer import synthesize

def generate_json_payload(idx):
    email = synthesize("email address", f"email_{idx}")
    pwd = synthesize("password", f"pwd_{idx}")
    fname = synthesize("first name", f"fname_{idx}")
    lname = synthesize("last name", f"lname_{idx}")
    phone = synthesize("phone number", f"phone_{idx}")
    
    tokens = [
        "{", 
        "\"first_name\"", ":", fname, ",",
        "\"last_name\"", ":", lname, ",",
        "\"email\"", ":", email, ",",
        "\"phone\"", ":", phone, ",",
        "\"password\"", ":", pwd, 
        "}"
    ]
    ner = [
        [3, 3, "first name"],
        [7, 7, "last name"],
        [11, 11, "email address"],
        [15, 15, "phone number"],
        [19, 19, "password"]
    ]
    return {"tokenized_text": tokens, "ner": ner}

def generate_form_payload(idx):
    email = synthesize("email address", f"email2_{idx}")
    pwd = synthesize("password", f"pwd2_{idx}")
    
    tokens = [
        "POST", "/api/v1/login", "HTTP/1.1", "Host:", "app.server.local",
        "\r\n\r\n",
        "email", "=", email, "&",
        "password", "=", pwd
    ]
    ner = [
        [8, 8, "email address"],
        [12, 12, "password"]
    ]
    return {"tokenized_text": tokens, "ner": ner}

def generate_telemetry_payload(idx):
    imsi = synthesize("imsi number", f"imsi_{idx}")
    serial = synthesize("device serial number", f"serial_{idx}")
    imei = synthesize("imei number", f"imei_{idx}")
    gender = synthesize("gender", f"gender_{idx}")
    iccid = synthesize("sim card iccid", f"iccid_{idx}")
    
    tokens = [
        "GET", "/track", "?", 
        "imsi", "=", imsi, "&", 
        "serial", "=", serial, "&", 
        "imei", "=", imei, "&", 
        "user_gender", "=", gender, "&",
        "iccid", "=", iccid, "HTTP/1.1"
    ]
    ner = [
        [5, 5, "imsi number"],
        [9, 9, "device serial number"],
        [13, 13, "imei number"],
        [17, 17, "gender"],
        [21, 21, "sim card iccid"]
    ]
    return {"tokenized_text": tokens, "ner": ner}

if __name__ == "__main__":
    count = 0
    with open('data/train_finetune.jsonl', 'a', encoding='utf-8') as f:
        for i in range(500):
            f.write(json.dumps(generate_json_payload(i)) + "\n")
            f.write(json.dumps(generate_form_payload(i)) + "\n")
            f.write(json.dumps(generate_telemetry_payload(i)) + "\n")
            count += 3
            
    print(f"Successfully appended {count} synthetic active-login/telemetry payloads to train_finetune.jsonl!")
