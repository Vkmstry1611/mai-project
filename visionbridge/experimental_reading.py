"""Literal OCR demonstration. Not the frozen VQA policy or a calibrated reader."""
import math
import re


def is_reading_request(question):
    q=str(question or '').lower()
    if re.search(r'\b(translate|meaning|mean|colour|color|brand|price|cost|date|language)\b', q):
        return False
    return bool(re.search(r'\b(read|written|printed|words?|letters?|text|says?)\b',q))


def route_request(enabled, question, frozen, reading):
    """Do not even extract experimental evidence unless explicitly opted in."""
    return reading() if enabled and is_reading_request(question) else frozen()


def _valid(item):
    try:
        b=item['box']; c=float(item['confidence'])
        return (item.get('kind')=='ocr' and item.get('status')=='complete'
                and item.get('id') and isinstance(item.get('text'),str) and item['text'].strip()
                and len(b)==4 and all(math.isfinite(float(v)) for v in b)
                and b[2]>b[0] and b[3]>b[1] and math.isfinite(c) and 0<=c<=1)
    except (KeyError,TypeError,ValueError):
        return False


def _overlap(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=x*y
    return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def reconstruct(evidence, threshold=.90):
    """Read separate regions separately; never add missing words or spell-correct.

    Boxes refer to the displayed, manually oriented image. A line requires similar
    height/vertical centre and a small horizontal gap. Weak text between strong
    words becomes an explicit gap. Conflicting overlapping readings are omitted.
    """
    valid=[dict(i) for i in evidence if _valid(i)]
    strong=[];seen=set()
    for i in valid:
        key=(i['text'].strip().casefold(),tuple(i['box']))
        if (i['confidence']>=threshold and any(c.isalnum() for c in i['text']) and key not in seen):
            strong.append(i);seen.add(key)
    weak=[i for i in valid if i['confidence']<threshold]
    conflicts=set()
    for n,a in enumerate(strong):
        for b in strong[n+1:]:
            if _overlap(a['box'],b['box'])>.8 and a['text'].casefold().strip()!=b['text'].casefold().strip():
                conflicts.update([a['id'],b['id']])
    strong=[i for i in strong if i['id'] not in conflicts]
    # Process left to right: OCR boxes on one line commonly have small y jitter.
    # Sorting by y first can encounter the right-hand word before its neighbour.
    strong.sort(key=lambda i:(i['box'][0],i['box'][1],i['id']))
    groups=[]
    for i in strong:
        b=i['box']; h=b[3]-b[1]; candidates=[]
        for group in groups:
            last=group[-1]; a=last['box']; ah=a[3]-a[1]
            gap=b[0]-a[2]
            if (0<=gap<=2*min(h,ah) and min(h,ah)>=.6*max(h,ah)
                    and abs((a[1]+a[3]-b[1]-b[3])/2)<=.45*min(h,ah)
                    and all(abs((j['box'][1]+j['box'][3]-b[1]-b[3])/2)
                            <=.45*min(h,j['box'][3]-j['box'][1]) for j in group)):
                candidates.append((abs((a[1]+a[3]-b[1]-b[3])/2),gap,len(candidates),group))
        if candidates:min(candidates,key=lambda c:c[:3])[3].append(i)
        else:groups.append([i])
    # Same-row fragments may arrive with slightly different y centres.
    # Sort within each group and retain row-major order across separate regions.
    groups.sort(key=lambda g:(min(i['box'][1] for i in g),min(i['box'][0] for i in g)))
    lines=[]; ids=[]
    for group in groups:
        group.sort(key=lambda i:i['box'][0]); words=[]; previous=None
        for i in group:
            if previous:
                a=previous['box'];b=i['box']
                if any(a[2]<= (w['box'][0]+w['box'][2])/2 <=b[0]
                       and min(a[1],b[1])<=(w['box'][1]+w['box'][3])/2<=max(a[3],b[3]) for w in weak):
                    words.append('[unclear]')
            words.append(i['text'].strip());ids.append(i['id']);previous=i
        lines.append(' '.join(words))
    partial=bool(weak or conflicts or len(valid)!=len(evidence))
    answer=('Recognized text fragments: '+ '; '.join(lines)+'.'
            +(' Other text is unclear or omitted.' if partial else '')) if lines else 'I cannot read reliable text in this image. Please rotate it or take a clearer photo.'
    return dict(answer=answer, lines=lines, decision='experimental_reading_supported' if lines else 'experimental_reading_abstained',
                supporting_evidence_ids=ids, conflicting_ids=sorted(conflicts),partial=partial,
                threshold=threshold, answer_source='ocr' if lines else 'abstention',
                reason='Literal OCR fragments only; separate regions are not alternative VQA answers. OCR scores are not calibrated probabilities.')
