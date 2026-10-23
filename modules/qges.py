"""Training-free lexical relevance; scores are similarities, never answer confidence."""
import re

def select(question, evidence, top_k=5):
    from modules.question_intent import classify
    from modules.evidence_decision import candidate_spans
    intent = classify(question)
    from modules.reliability import mentioned_objects
    targets = mentioned_objects(question)
    target_boxes = [item['box'] for item in evidence if item.get('kind') == 'object'
                    and item.get('text') in targets and item.get('box')
                    and item.get('confidence', 1) >= .35]
    # Question asks for a brand, but a bare word alone is insufficient. Allow
    # a unique prominent literal only in a populated label with smaller text.
    ocr_items = [i for i in evidence if i.get('kind') == 'ocr' and i.get('status') == 'complete']
    prominent = set()
    if intent.kind == 'brand' and len(ocr_items) > 1:
        from modules.ocr_phrases import valid_box
        words_only = [i for i in ocr_items if valid_box(i.get('box'))
                      and re.fullmatch(r"[A-Za-z][A-Za-z '&-]{2,40}", i['text'].strip())]
        for i in words_only:
            area = lambda j: (j['box'][2]-j['box'][0]) * (j['box'][3]-j['box'][1])
            others = [j for j in words_only if j['id'] != i['id']]
            if others and all(area(i) >= 2 * area(j) for j in others):
                prominent.add(i['id'])
    words = set(re.findall(r'\w+', question.lower())) - {'what', 'is', 'the', 'a', 'on', 'in', 'this'}
    ranked = []
    for item in evidence:
        item = dict(item)
        if item.get('id') in prominent:
            item['brand_role'] = 'prominent_literal_in_label'
        terms = set(re.findall(r'\w+', item['text'].lower()))
        score = len(words & terms) / max(1, len(words | terms))
        if intent.supported:
            spans = candidate_spans(question, item)
            if not spans:
                continue
            if targets:
                box = item.get('box', [])
                if len(box) != 4:
                    continue
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if not any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in target_boxes):
                    continue
                matching = [b for b in target_boxes if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
                if len(matching) == 1:
                    item['target_scope_box'] = matching[0]
            score += 1.0
        elif score == 0:
            continue
        ranked.append(dict(item, relevance=score, selection_reason=f'{intent.kind} intent and lexical relevance; not confidence'))
    ranked.sort(key=lambda item: (-item['relevance'], item.get('id', item['text'])))
    return ranked if top_k is None else ranked[:max(0, top_k)]
