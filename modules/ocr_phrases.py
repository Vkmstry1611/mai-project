"""Conservative geometry-only OCR assembly; no labels, language model or spell correction.

Only nearby aligned boxes may join. Overlapping alternative recognitions,
separate columns, and unreadable intervening fragments remain separate.
"""
import math


def valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
            and box[0] < box[2] and box[1] < box[3])


def overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def assemble(items, threshold):
    """Return literal coherent candidates and low-confidence item IDs.

    Confidence of a phrase is its weakest fragment; it cannot rise by merging.
    Geometric constants are fixed heuristics, not probabilities.
    """
    usable = [i for i in items if valid_box(i.get('box')) and i.get('text', '').strip()]
    weak = [i for i in usable if i.get('confidence', 0) < threshold]
    strong = [i for i in usable if i.get('confidence', 0) >= threshold]
    nodes = []
    for i in sorted(strong, key=lambda i: (i['box'][1], i['box'][0], i['id'])):
        key = (' '.join(i['text'].casefold().split()), tuple(i['box']))
        duplicate = next((n for n in nodes if n['key'] == key), None)
        if duplicate:
            duplicate['ids'].append(i['id'])
            duplicate['confidence'] = min(duplicate['confidence'], i['confidence'])
        else:
            nodes.append(dict(text=i['text'].strip(), box=i['box'], ids=[i['id']],
                              confidence=i['confidence'], key=key))

    def blocked(a, b):
        # Never bridge across a discarded word, e.g. KEEP [NOT] ENTER.
        corridor = union([a, b])
        return any(overlap(w['box'], corridor) and not overlap(w['box'], a)
                   and not overlap(w['box'], b) for w in weak)

    lines = []
    for n in sorted(nodes, key=lambda n: (n['box'][0], n['box'][1], n['text'])):
        b = n['box']; h = b[3] - b[1]; cy = (b[1] + b[3]) / 2
        matches = []
        for line in lines:
            last = line[-1]; a = last['box']; ah = a[3] - a[1]
            gap = b[0] - a[2]
            if (0 <= gap <= 1.5 * min(h, ah)
                    and abs(cy - (a[1] + a[3]) / 2) <= .45 * min(h, ah)
                    and min(h, ah) >= .6 * max(h, ah) and not blocked(a, b)):
                matches.append(line)
        if len(matches) == 1:
            matches[0].append(n)
        else:
            lines.append([n])
    # Build each line left-to-right, then join only compact, aligned lines.
    candidates = []
    for line in lines:
        line.sort(key=lambda n: n['box'][0])
        candidates.append(dict(text=' '.join(n['text'] for n in line),
                               box=union([n['box'] for n in line]),
                               ids=[i for n in line for i in n['ids']],
                               confidence=min(n['confidence'] for n in line), lines=1))
    candidates.sort(key=lambda n: (n['box'][1], n['box'][0], n['text']))
    blocks = []
    for n in candidates:
        b = n['box']; h = b[3] - b[1]; matches = []
        for block in blocks:
            a = block['last_box']; ah = a[3] - a[1]
            gap = b[1] - a[3]
            width = min(a[2] - a[0], b[2] - b[0])
            horizontal = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            if (block['lines'] < 3 and 0 <= gap <= .6 * min(h, ah)
                    and horizontal >= .6 * width and min(h, ah) >= .6 * max(h, ah)
                    and not blocked(a, b)):
                matches.append(block)
        if len(matches) == 1:
            block = matches[0]
            block.update(text=block['text'] + ' ' + n['text'],
                         box=union([block['box'], b]), ids=block['ids'] + n['ids'],
                         confidence=min(block['confidence'], n['confidence']),
                         lines=block['lines'] + 1, last_box=b)
        else:
            blocks.append(dict(n, last_box=b))
    return blocks, [i['id'] for i in weak]
