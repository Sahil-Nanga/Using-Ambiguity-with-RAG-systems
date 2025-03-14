import re

class Cleaner():
    @staticmethod
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
