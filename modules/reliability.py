"""Conservative inference safeguards, not a trained/calibrated answerability gate.

Detector non-observation cannot prove an object is absent. Responses describe what
the app can verify and avoid generating attributes for an unsupported object.
"""
from dataclasses import dataclass, field
import re

COCO_NAMES = ('person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,'
              'stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,'
              'backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,'
              'baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,'
              'banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,'
              'bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,'
              'sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush').split(',')
ALIASES = {name: name for name in COCO_NAMES}
ALIASES.update({'people': 'person', 'man': 'person', 'men': 'person', 'woman': 'person', 'women': 'person',
                'child': 'person', 'children': 'person', 'bike': 'bicycle', 'motorbike': 'motorcycle',
                'sofa': 'couch', 'table': 'dining table', 'phone': 'cell phone', 'mobile phone': 'cell phone',
                'television': 'tv', 'remote control': 'remote', 'fridge': 'refrigerator', 'aeroplane': 'airplane'})

@dataclass
class Check:
    decision: str = 'not_checked'
    response: str | None = None
    reason: str = ''
    targets: list = field(default_factory=list)
    observed: list = field(default_factory=list)
    method: str = 'conservative rules and object detection; not calibrated'
    probability: None = None

def mentioned_objects(question):
    text = re.sub(r'[^a-z0-9 ]', ' ', question.lower())
    # Match longest phrases first so "hot dog" does not also require a dog.
    targets = []
    for alias in sorted(ALIASES, key=len, reverse=True):
        pattern = r'\b' + re.escape(alias) + r'(?:s|es)?\b'
        if re.search(pattern, text):
            target = ALIASES[alias]
            if target not in targets:
                targets.append(target)
            text = re.sub(pattern, ' ', text)
    return targets

def question_check(question):
    text = question.lower()
    if re.search(r'\b(weather|temperature|forecast|raining|rainy|snowing|sunny|cloudy|windy|humidity)\b', text):
        return Check('abstain', "I can't determine the weather from this image.",
                     'Weather questions are conservatively declined. A photograph is not a reliable weather measurement.')
    identities = {'person', 'cat', 'dog', 'horse', 'bird', 'cow', 'sheep'}
    if (re.search(r'\b(names?|named)\b', text) and identities.intersection(mentioned_objects(question))
            and not re.search(r'\b(brand|product|label|company)\b', text)):
        return Check('abstain', "I can't determine a person's or animal's name from this image.",
                     'Identity cannot be established by visual appearance alone.')
    if re.search(r'\b(tomorrow|next week|will .*arrive|printed on the back)\b', text):
        return Check('abstain', "I can't determine that from what is visible in this image.",
                     'Question requests future or unseen information.')
    return Check(targets=mentioned_objects(question), reason='Checking recognizable objects mentioned in the question.')

def check_objects(targets, evidence, detection_ok, threshold=0.35):
    if not targets:
        return Check(reason='No supported object category mentioned; answer remains unverified.')
    if not detection_ok:
        return Check('abstain', "I can't verify that item in this image. Please try a clearer photo or another question.",
                     'Object detection unavailable; do not invent object attributes.', targets)
    observed = sorted({item['text'] for item in evidence if item.get('kind') == 'object'
                       and item.get('confidence', 0) >= threshold})
    missing = [target for target in targets if target not in observed]
    if missing:
        label = ', '.join(missing)
        response = f"I can't see that item in this image ({label}). It may be obscured or too small to recognize."
        return Check('abstain', response, 'Mentioned object was not observed by YOLO above the detection threshold.', targets, observed)
    return Check('supported_objects', None, 'Mentioned objects were detected; this does not verify the generated answer.', targets, observed)
