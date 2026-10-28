"""Isolated CPU full-line OCR. No frozen thresholds, engines or outputs altered."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from visionbridge.experimental_orientation import transform,bounds,matrices
from modules.evidence_store import digest,image_identity

ROOT=Path(__file__).resolve().parents[1]
VERSION='full-text-ppocr-en-v2'
ENGLISH=ROOT/'.deps/fulltext/en_PP-OCRv3_rec_infer.onnx'
ENGLISH_SHA='ef7abd8bd3629ae57ea2c28b425c1bd258a871b93fd2fe7c433946ade9b5d9ea'


def assemble(items):
    """Order full lines by their projected baseline; never guess missing letters."""
    if not items:return '',[],True
    slopes=[math.atan2(i['polygon'][1][1]-i['polygon'][0][1],i['polygon'][1][0]-i['polygon'][0][0]) for i in items]
    slope=statistics.median(slopes);h=statistics.median(i['box'][3]-i['box'][1] for i in items)
    def position(i):
        x=sum(p[0] for p in i['polygon'])/4;y=sum(p[1] for p in i['polygon'])/4
        return y*math.cos(slope)-x*math.sin(slope)
    ordered=sorted(items,key=lambda i:(position(i),i['box'][0],i['id']))
    groups=[]
    for i in ordered:
        if groups and abs(position(i)-statistics.mean(position(x) for x in groups[-1]))<.45*h:
            groups[-1].append(i)
        else:groups.append([i])
    lines=[];ids=[];partial=False;last=None
    for group in groups:
        group.sort(key=lambda i:i['box'][0]);parts=[]
        y=statistics.mean(position(i) for i in group)
        if last is not None and y-last>2*h:lines.append('')
        for i in group:
            c=i['confidence']
            if c<.5:parts.append('[unreadable text]');partial=True
            elif c<.75:parts.append('[uncertain: '+i['text']+']');ids.append(i['id']);partial=True
            else:parts.append(i['text']);ids.append(i['id'])
        lines.append(' '.join(parts));last=y
    return '\n'.join(lines),ids,partial


class FullTextEngine:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        from importlib.metadata import version
        start=time.perf_counter()
        if not ENGLISH.is_file() or hashlib.sha256(ENGLISH.read_bytes()).hexdigest()!=ENGLISH_SHA:
            raise ValueError('Verified English OCR model missing or changed; run setup_full_text.ps1. No fallback.')
        self.engine=RapidOCR(rec_model_path=str(ENGLISH),intra_op_num_threads=4,inter_op_num_threads=1,text_score=0.)
        import rapidocr_onnxruntime
        folder=Path(rapidocr_onnxruntime.__file__).parent/'models'
        self.source={'engine':'RapidOCR','package':version('rapidocr-onnxruntime'),'runtime':version('onnxruntime'),
            'device':'CPUExecutionProvider','english_sha256':ENGLISH_SHA,
            'bundled_models':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.onnx')}}
        self.loading_seconds=time.perf_counter()-start

    def orientation(self,image):
        import numpy as np
        tick=time.perf_counter()
        boxes,_=self.engine(image,use_cls=False,use_rec=False)
        if not boxes:return None,dict(reason='no_text_regions',seconds=time.perf_counter()-tick,voters=[])
        array=self.engine.load_img(image)
        crops=self.engine.get_crop_img_list(array,np.asarray(boxes,dtype=np.float32))
        _,labels,_=self.engine.text_cls(crops)
        voters=[];votes={a:0. for a in (0,90,180,270)}
        for box,(label,confidence) in zip(boxes,labels):
            b=bounds(box);w=b[2]-b[0];h=b[3]-b[1]
            if min(w,h)<=0:continue
            base=90 if h>1.5*w else 0
            angle=(base+int(label))%360
            weight=min(w*h,100000.)
            voters.append(dict(polygon=box,angle=angle,confidence=float(confidence),weight=weight))
            if confidence>=.9:votes[angle]+=weight
        ranked=sorted(votes,key=lambda a:(-votes[a],a));total=sum(votes.values())
        fraction=votes[ranked[0]]/total if total else 0
        winner=ranked[0] if fraction>=.75 and total else None
        return winner,dict(reason='detector_geometry_and_learned_crop_polarity' if winner is not None else 'mixed_or_uncertain_orientation',
            votes={str(a):v for a,v in votes.items()},dominant_fraction=fraction,voters=voters,seconds=time.perf_counter()-tick,
            warning='Classifier agreement is not a calibrated page-reading probability.')

    def read(self,image,cache,manual=None):
        from PIL import ImageOps
        from visionbridge.assistive_tools import quality
        start=time.perf_counter();original=ImageOps.exif_transpose(image).convert('RGB')
        if original.width*original.height>40_000_000:raise ValueError('Resize below 40 megapixels.')
        work=original.copy();work.thumbnail((2000,2000))
        identity=dict(version=VERSION,image=image_identity(original),source=self.source,manual=manual,
            preprocessing='RGB bounded downscale only',size=list(work.size),policy='line-score-markers-0p50-0p75-v1')
        path=Path(cache)/(digest(identity)+'.json')
        if path.exists():
            payload=json.loads(path.read_text(encoding='utf-8'))
            if payload['identity']!=identity or digest(payload['result'])!=payload['sha256']:raise ValueError('Full-text cache integrity mismatch')
            result=payload['result'];result['cache_hit']=True;result['request_seconds']=time.perf_counter()-start
            return result
        angle,orientation=self.orientation(work) if manual is None else (int(manual),{'reason':'explicit_manual_override'})
        if angle not in (None,0,90,180,270):raise ValueError('Unsupported manual angle')
        records=[];raw=[];stages=None
        if angle is not None:
            corrected=work.rotate(angle,expand=True)
            values,stages=self.engine(corrected,use_cls=False,text_score=0.)
            def recognition_quality(values):
                scores=[float(v[2]) for v in values or [] if sum(c.isalnum() for c in v[1])>=2]
                median=statistics.median(scores) if scores else 0.
                fraction=sum(c>=.75 for c in scores)/len(scores) if scores else 0.
                return dict(median=median,reliable_fraction=fraction,score=(median+fraction)/2)
            first=recognition_quality(values)
            orientation['recognition_validation']={str(angle):first}
            if manual is None and (first['median']<.65 or first['reliable_fraction']<.6):
                opposite=(angle+180)%360
                alternative,alternative_stages=self.engine(work.rotate(opposite,expand=True),use_cls=False,text_score=0.)
                other=recognition_quality(alternative)
                orientation['recognition_validation'][str(opposite)]=other
                if other['score']-first['score']>=.2:
                    angle=opposite;corrected=work.rotate(angle,expand=True);values=alternative;stages=alternative_stages
                    orientation['recognition_override']='Opposite orientation verified by median score and reliable-line fraction, not output length.'
            for index,(polygon,text,confidence,*extra) in enumerate(values or []):
                polygon=[[float(x),float(y)] for x,y in polygon]
                if not text.strip() or not math.isfinite(float(confidence)):continue
                row=dict(kind='ocr',text=text,confidence=float(confidence),polygon=polygon,box=bounds(polygon),status='complete',source=self.source)
                row['id']='fullocr-'+digest(dict(identity=identity,angle=angle,index=index,row=row))[:24]
                raw.append(row)
            transcript,ids,partial=assemble(raw)
            for row in raw:
                orig=[]
                for point in row['polygon']:
                    x,y=transform(point,work.width,work.height,angle,True);orig.append([x*original.width/work.width,y*original.height/work.height])
                records.append(dict(row,corrected_box=row['box'],corrected_polygon=row['polygon'],original_box=bounds(orig),
                    original_polygon=orig,box=bounds(orig),polygon=orig,coordinate_space='original-EXIF-RGB'))
            final_quality=recognition_quality(values)
            usable=final_quality['median']>=.65 and final_quality['reliable_fraction']>=.6
        else:transcript='';ids=[];partial=True;usable=False
        hints=quality(original)
        if usable:
            answer='Experimental transcription — OCR may contain errors.\n'+transcript
            if partial:answer+='\nSome portions are uncertain or unreadable.'
        else:
            answer='I cannot read reliable text from this photo. Please try manual rotation or upload a clearer upright image.'
            if hints['warnings']:answer+=' '+hints['warnings'][0]
        result=dict(answer=answer,final_answer=answer,transcription=transcript if usable else '',answer_source='ocr' if usable else 'abstention',
            decision='experimental_full_text' if usable else 'experimental_full_text_abstained',partial=partial,
            evidence=records,selected_evidence=[i for i in records if i['id'] in ids],supporting_evidence_ids=ids if usable else [],
            source=self.source,quality=hints,orientation=dict(orientation,selected_ccw_degrees=angle,
                original_size=list(original.size),working_size=list(work.size),
                transform=matrices(original.size,work.size,angle) if angle is not None else {}),
            image_id=identity['image'],ocr_lines=len(raw),cache_hit=False,request_seconds=time.perf_counter()-start,
            stages=stages,extraction_status={'ocr':{'status':'complete' if usable else 'unavailable'}},
            warning='Not a guaranteed complete reading. Scores are not correctness probabilities. No frozen VQA evaluation.')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as file:json.dump({'identity':identity,'result':result,'sha256':digest(result)},file,ensure_ascii=False)
        return result
