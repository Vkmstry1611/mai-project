"""Deterministic UI routing only; never a confidence score or VQA policy."""
from dataclasses import dataclass
import re

VERSION='workspace-intent-v1'


@dataclass(frozen=True)
class Route:
    task: str
    reason: str
    message: str = ''


def classify(question):
    q=re.sub(r'\s+',' ',str(question or '').lower().replace('’',"'")).strip()
    bare=q.strip(' .?!')
    if not bare or bare in {'help','help me','please help','please help me','can you help','can you help me','check this','what about this','tell me about this','is this okay','is this ok'}:
        return Route('clarify','The question does not specify a visual or reading task.',
                     'Would you like me to describe something in the image or read its visible text? Please ask one specific question.')
    if re.search(r"\b(?:do not|don't|never)\s+(?:read|transcribe)\b",q):
        return Route('clarify','Negated reading requests are not interpreted automatically.',
                     'Please ask what you would like to know about the image, without a reading instruction.')
    if re.search(r'\btranslate\b|\bwhat does .+ mean\b|\bmeaning\b',q):
        return Route('clarify','Literal reading does not establish meaning or translation.',
                     'I can describe visible content or read visible words. Do you want me to read the text or describe something in the image?')
    visual=bool(re.search(r'\b(colou?r|shape|how many|count|doing|activity|activities|scene)\b',q))
    explicit_read=bool(re.search(r'\b(read|transcribe|written|printed|text|words?|letters?)\b',q))
    if visual and explicit_read and re.search(r'\band\b',q):
        return Route('clarify','The request mixes reading and visual tasks.',
                     'Please ask one question at a time: what should I read, or what would you like to know about the image?')
    fields=bool(re.search(r'\b(expir(?:y|ation|e|es|ed)|use[ -]*by|best[ -]*before|manufactur(?:e|ed|ing)|mfg|mfd|date|price|costs?|mrp|product name|item name)\b',q)
                or re.search(r'\bwhen .+ made\b|\bhow much is (?:this|that|it)\b',q))
    field_appearance=bool(re.search(r'\b(colou?r|shape)\b',q)) and not explicit_read
    if fields and not field_appearance:
        return Route('unsupported_field','Reliable question-focused field extraction is not implemented.',
                     'I cannot reliably identify or validate a specific printed field yet, such as an expiry date, manufacturing date, product name or price. '
                     'You can ask me to read all visible text, but I cannot confirm which value belongs to that field.')
    # Reuse the reader's supported question forms; do not rewrite the question.
    from visionbridge.experimental_reading import is_reading_request
    says=bool(re.search(r'\b(?:this|that|it|sign|label|notice|poster|package|bottle|image|picture|photo)\b.*\bsays?\b',q))
    if not visual and (explicit_read or says) and is_reading_request(q):
        return Route('reading','An explicit request asks for literal visible text; use the existing experimental reader.')
    if not visual and (explicit_read or says):
        return Route('clarify','This wording is not accepted by the existing literal reader; do not silently switch to VQA.',
                     'Would you like me to read the visible words? Please ask “Read the text in this image.” For a visual description, ask a specific visual question.')
    return Route('vqa','General visual question or unmatched wording; use frozen Visual Q&A with its existing answer-or-abstain policy.')
