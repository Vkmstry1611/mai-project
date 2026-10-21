"""Small explicit question router; no model or reference answers are used."""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    kind: str
    cue: str = ''
    supported: bool = True


def classify(question):
    text = ' '.join(question.lower().split())
    # Polite extraction requests are not yes/no verification. Bare "can you
    # read this?" stays unsupported because it may ask about capability.
    polite = re.match(r'^(?:can|could|would) you (?:please )?(?:tell (?:me|us)|identify|help me identify)\s+', text)
    if polite:
        text = text[polite.end():]
    elif re.match(r'^(?:can|could|would) you (?:please )?read\b', text) and re.search(r'\b(to me|for me|out loud|please)\b', text):
        text = re.sub(r'^(?:can|could|would) you (?:please )?', '', text)
    if re.search(r'\b(colou?r|shape|material|language|translate|meaning|how many|where|who)\b', text):
        return Intent('general', supported=False)
    # Do not turn verification/negative/future questions into extraction requests.
    if re.search(r'\b(is|are|does|do|can|will|was|were)\b', text.split('?')[0].split(' ')[0]) or re.search(r'\b(not|except|tomorrow|next week|will|back)\b', text):
        return Intent('general', supported=False)
    if re.search(r'\b(price|cost)\b', text) or re.search(r'\bhow much (?:is|does .* cost)\b', text):
        return Intent('price')
    if re.search(r'\b(expir\w*|best before|use by)\b', text):
        return Intent('date', 'expiry')
    if re.search(r'\b(manufactur\w*|packed|packing)\b', text) and re.search(r'\b(date|when)\b', text):
        return Intent('date', 'manufacture')
    if re.search(r'\b(date|when)\b', text):
        return Intent('date')
    if re.search(r'\b(brand|manufacturer|company)\b', text):
        if re.search(r'\b(flavo[u]?r|name and|and.*name)\b', text):
            return Intent('general', supported=False)
        return Intent('brand')
    if re.search(r'\b(read|written|printed|text|say|says|word|words|letter|letters)\b', text):
        return Intent('reading')
    return Intent('general', supported=False)
