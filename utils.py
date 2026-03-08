import unicodedata
from datasets import load_dataset

def normalize_text(text):
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = ' '.join(text.split())
    return text

def load_afrimedqa():
    return load_dataset("intronhealth/afrimedqa_v2")["train"]

def normalie_afrimedqa():
    dataset = load_afrimedqa()