import re

def normalized_choice(text):
    text = re.sub(r'^(?:</think>\s*)+', '', text.strip(), flags=re.I)
    if text.startswith('**') and text.endswith('**') and len(text) > 4:
        text = text[2:-2].strip()
    m = re.fullmatch(r'(?:(?:option|answer)\s*:?\s*)?(?:\(([a-d])\)|([a-d]))[.]?', text, flags=re.I)
    return '(' + (m.group(1) or m.group(2)).lower() + ')' if m else None
