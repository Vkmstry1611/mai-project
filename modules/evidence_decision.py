"""Restricted OCR answer extraction. Missing/ambiguous evidence retains BLIP.

This module makes no calibrated confidence or learned answerability claims.
"""
import re
from datetime import date
from modules.question_intent import classify

DATE = re.compile(r'\b(?:\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:\d{1,2}[/-])?\d{1,2}[/-]\d{4})\b')
PRICE = re.compile(r'(?:[$€£₹]\s*\d+(?:[.,]\d{1,2})?|\b(?:USD|INR|Rs\.?|EUR|GBP)\s*\d+(?:[.,]\d{1,2})?)(?![\d.,])', re.I)
EXPIRY = re.compile(r'\b(exp(?:iry|ires|iration)?|best before|use by)\b', re.I)
MANUFACTURE = re.compile(r'\b(mfg|mfd|manufactur\w*|packed|packing)\b', re.I)
BRAND = re.compile(r'^\s*(?:brand|manufacturer|company)\s*[:\-]\s*(.+?)\s*$', re.I)


def valid_date(span):
    parts = re.split(r'[/-]', span)
    numbers = [int(value) for value in parts]
    possibilities = []
    if len(parts) == 2:
        month, year = numbers
        possibilities = [(year, month, 1)]
    elif len(parts[0]) == 4:
        possibilities = [tuple(numbers)]
    else:
        first, second, year = numbers
        possibilities = [(year, second, first), (year, first, second)]
    for values in possibilities:
        try:
            date(*values)
            return True
        except ValueError:
            pass
    return False


def candidate_spans(question, item):
    """Return only literal spans from this OCR item, retaining provenance."""
    intent = classify(question)
    if not intent.supported or item.get('kind') != 'ocr' or item.get('status') != 'complete':
        return []
    text = item.get('text', '').strip()
    if not text or not item.get('id') or not item.get('box'):
        return []
    if intent.kind == 'reading':
        return [text]
    if intent.kind == 'price':
        # A price labelled discount/saving is not the selling price.
        if re.search(r'\b(save|saving|off|discount|was|old)\b', text, re.I):
            return []
        spans = [m.group() for m in PRICE.finditer(text)]
        if not spans and re.search(r'\b(price|mrp|cost)\s*[:\-]', text, re.I):
            spans = re.findall(r'\d+(?:[.,]\d{1,2})?', text)
        return spans
    if intent.kind == 'date':
        if intent.cue == 'expiry' and not EXPIRY.search(text):
            return []
        if intent.cue == 'manufacture' and not MANUFACTURE.search(text):
            return []
        if intent.cue and EXPIRY.search(text) and MANUFACTURE.search(text):
            return []
        return [m.group() for m in DATE.finditer(text) if valid_date(m.group())]
    if intent.kind == 'brand':
        match = BRAND.match(text)
        if match:
            return [match.group(1)]
        return [text] if item.get('brand_role') == 'prominent_literal_in_label' else []
    return []


def decide(inputs):
    from modules.fusion import FusionOutput
    intent = classify(inputs.question)
    trace = {'threshold': inputs.ocr_min_score, 'candidates': [],
             'filtered_evidence_ids': [i.get('id') for i in inputs.evidence
                                       if i.get('confidence', 0) < inputs.ocr_min_score]}
    def retain(reason):
        return FusionOutput(inputs.baseline_answer, reason=reason,
                            baseline_answer=inputs.baseline_answer, candidate_trace=trace)
    if not intent.supported:
        return retain('Question type is not supported by the restricted OCR selector.')
    if inputs.selection_truncated:
        return retain('Relevant evidence was incomplete or selection truncated; unseen candidates could conflict.')
    from modules.ocr_phrases import assemble, valid_box, overlap
    statuses = inputs.extraction_status
    if statuses.get('ocr', {}).get('status') == 'disabled':
        return retain('OCR was not extracted for this question in the saved capture; no inference during replay.')
    if any(s.get('status') == 'unavailable' for s in statuses.values()):
        return retain('Evidence extraction unavailable; retaining BLIP.')
    for kind, status in statuses.items():
        if status.get('status') != 'partial':
            continue
        # Partial cache without rejected-row provenance is never assumed safe.
        scopes = [i.get('target_scope_box') for i in inputs.evidence]
        rejections = status.get('rejections', [])
        scoped = (kind == 'ocr' and scopes and all(valid_box(s) and s == scopes[0] for s in scopes)
                  and all(valid_box(i.get('box')) and scopes[0][0] <= i['box'][0] < i['box'][2] <= scopes[0][2]
                          and scopes[0][1] <= i['box'][1] < i['box'][3] <= scopes[0][3] for i in inputs.evidence)
                  and statuses.get('object', statuses.get('objects', {})).get('status') == 'complete'
                  and len(rejections) == status.get('rejected_items', -1) and bool(rejections))
        if not scoped or any(not valid_box(r.get('box')) or overlap(r['box'], scopes[0]) for r in rejections):
            return retain('Partial extraction cannot be validated outside the requested region; retaining BLIP.')
    groups = {}
    if intent.kind == 'reading':
        candidates, _ = assemble([i for i in inputs.evidence if candidate_spans(inputs.question, i)],
                                 inputs.ocr_min_score)
        trace['candidates'] = candidates
        for c in candidates:
            key = ' '.join(c['text'].casefold().split())
            groups.setdefault(key, []).extend((c['text'], i) for i in c['ids'])
    else:
        for item in inputs.evidence:
            if item.get('confidence', 0) < inputs.ocr_min_score:
                continue
            for span in candidate_spans(inputs.question, item):
                trace['candidates'].append({'text': span, 'ids': [item['id']],
                                            'confidence': item['confidence'], 'box': item['box']})
                key = ' '.join(span.casefold().split())
                groups.setdefault(key, []).append((span, item['id']))
    if not groups:
        return retain('No usable OCR candidate passes intent and candidate-level confidence filtering.')
    if len(groups) != 1:
        return retain('Multiple distinct OCR answers remain; retaining BLIP rather than guessing.')
    values = next(iter(groups.values()))
    answer = values[0][0]
    # Empty BLIP or its abstention should not be overridden in Stages 1–4.
    if not inputs.baseline_answer.strip() or inputs.baseline_answer.strip().casefold() == 'unanswerable':
        return retain('BLIP abstained; OCR override requires later reliability validation.')
    return FusionOutput(answer, grounded=True, baseline_answer=inputs.baseline_answer,
                        reason='One distinct literal OCR answer satisfies the supported intent; this is extraction support, not calibrated correctness.',
                        supporting_evidence_ids=sorted({evidence_id for _, evidence_id in values}),
                        decision='ocr_supported', source='ocr', candidate_trace=trace)
