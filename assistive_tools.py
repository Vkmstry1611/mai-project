"""Experimental capture hints and label-grounded field extraction, no models."""
import re
import math


def quality(image):
    import numpy as np
    from PIL import ImageOps
    gray=ImageOps.grayscale(image);gray.thumbnail((512,512));a=np.asarray(gray,dtype=float)
    brightness=float(a.mean());contrast=float(a.std())
    sharpness=float(np.var(np.diff(a,axis=0))+np.var(np.diff(a,axis=1))) if min(a.shape)>1 else 0.
    warnings=[]
    if min(image.size)<320:warnings.append('Some text may be too small. Try moving closer or using a higher-resolution photo.')
    if brightness<45:warnings.append('The image is dark. Try improving the lighting.')
    if brightness>235:warnings.append('The image is very bright. Reduce glare or strong direct lighting.')
    if contrast<25:warnings.append('The image has low contrast. Try even lighting and a plain background.')
    if sharpness<15:warnings.append('The image may be blurry or have little texture. Hold the camera steady and focus on the text.')
    return dict(version='capture-heuristics-v1',brightness=brightness,contrast=contrast,edge_variance=sharpness,
        resolution=list(image.size),warnings=warnings,heuristic=True,blocks_inference=False)


LABELS={
    'serving size':r'\bserving\s+size\b',
    'net quantity':r'\b(?:net\s*(?:weight|wt\.?|quantity|contents)|container\s+quantity)\b',
    'calories':r'\bcalories(?:\s+per\s+(?:serving|container))?\b',
    'expiry':r'\b(?:exp(?:iry|iration)?|expires?|best\s*before|use\s*by)\b',
    'manufacturing':r'\b(?:manufactur(?:ed|ing|e)(?:\s*date)?|mfg|mfd)\b',
    'price':r'\b(?:price|mrp|cost)\b',
    'brand':r'\b(?:brand(?:\s*name)?|product\s*name)\b',
}


def field_intent(question):
    q=str(question).lower()
    if re.search(r'\b(?:net weight|weight of|quantity|package weight)\b',q):return 'net quantity'
    if re.search(r'\b(?:is .+ expired|safe to|eat|take this medicine)\b',q):return None
    for key,pattern in LABELS.items():
        if re.search(pattern,q):return key
    return None


def field_value(kind,text):
    text=text.strip(' :;=-\n')
    if kind in ('serving size','net quantity'):
        match=re.fullmatch(r'\d+(?:[.,]\d+)?\s*(?:fl\.?\s*oz|oz|ounces?|g|grams?|kg|ml|l|cups?)(?:\s*\(\d+(?:[.,]\d+)?\s*(?:g|ml|oz)\))?',text,re.I)
    elif kind=='calories':match=re.fullmatch(r'\d{1,4}(?:\s*kcal)?',text,re.I)
    elif kind in ('expiry','manufacturing'):
        # Return printed notation, never infer a missing date/format or safety.
        match=re.fullmatch(r'(?:\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}|\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|[A-Za-z]{3,9}\s+\d{2,4})',text)
    elif kind=='price':match=re.fullmatch(r'(?:(?:Rs\.?|INR|USD|EUR|[$£₹€])\s*)?\d+(?:[.,]\d{1,2})?(?:\s*(?:INR|USD|EUR))?',text,re.I)
    else:match=re.fullmatch(r'[A-Za-z][A-Za-z0-9 &\x27-]{1,60}',text)
    return text if match else None


def targeted(question,evidence):
    kind=field_intent(question)
    candidates=[]
    if kind:
        for label in evidence:
            if label.get('confidence',0)<.90:continue
            m=re.search(LABELS[kind],label['text'],re.I)
            if not m:continue
            inline=field_value(kind,label['text'][m.end():])
            if inline:candidates.append((inline,[label['id']]))
            else:
                a=label['box'];h=a[3]-a[1]
                for value in evidence:
                    if value['id']==label['id'] or value.get('confidence',0)<.90:continue
                    b=value['box']
                    right=0<=b[0]-a[2]<=3*h and abs((a[1]+a[3]-b[1]-b[3])/2)<h*.6
                    below=0<=b[1]-a[3]<=1.5*h and abs(b[0]-a[0])<2*h
                    text=field_value(kind,value['text']) if right or below else None
                    if text:candidates.append((text,[label['id'],value['id']]))
    unique={value.casefold() for value,ids in candidates}
    if len(unique)==1:
        value=candidates[0][0];ids=sorted({i for v,group in candidates for i in group})
        suffix=' The date format and whether the product is safe are not verified.' if kind in ('expiry','manufacturing') else ''
        return dict(answer=f'Printed {kind} field: {value}.'+suffix,answer_source='ocr',decision='experimental_labeled_field',supporting_evidence_ids=ids)
    return dict(answer='I cannot identify a unique, clearly labeled value for that field. Please show the relevant label and value together in a clearer photo.',
        answer_source='abstention',decision='experimental_field_conflicting' if len(unique)>1 else 'experimental_field_insufficient',supporting_evidence_ids=[])


def compare(previous,current):
    # Upload creates conversation state before any quality measurement exists.
    # That is not a completed earlier analysis (nor a numeric zero baseline).
    keys=('brightness','contrast','edge_variance')
    measurements=previous.get('quality') if isinstance(previous,dict) else None
    if not isinstance(measurements,dict) or not all(isinstance(measurements.get(k),(int,float)) for k in keys):
        return {'message':'No earlier completed analysis in this browser session.'}
    return dict(previous_image_id=previous.get('image_id'),current_image_id=current.get('image_id'),
        brightness_change=current['quality']['brightness']-previous['quality']['brightness'],
        contrast_change=current['quality']['contrast']-previous['quality']['contrast'],
        edge_variance_change=current['quality']['edge_variance']-previous['quality']['edge_variance'],
        previous_ocr_lines=previous.get('ocr_lines',0),current_ocr_lines=current.get('ocr_lines',0),
        message='Quality and detected text can change without improving answer correctness.')
