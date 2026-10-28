"""Observational timing wrapper; frozen source files are never edited."""
import argparse
import json
from pathlib import Path
import time


def run(args):
    started=time.perf_counter(); timings={}; originals=[]
    def wrap(module,name,label):
        original=getattr(module,name)
        def observed(*a,**kw):
            start=time.perf_counter()
            try:return original(*a,**kw)
            finally:timings[label]=timings.get(label,0)+time.perf_counter()-start
        originals.append((module,name,original));setattr(module,name,observed)
    try:
        from modules import vqa,ocr,object_detection,qges,evidence_decision,evidence_store
        from modules import answerability_gate
        timings['instrumentation_imports_seconds']=time.perf_counter()-started
        for module,name,label in [(vqa,'answer','blip_seconds_including_loading'),
            (vqa,'load_blip','blip_loading_seconds'),(ocr,'extract','ocr_seconds_including_loading'),
            (ocr,'load_reader','ocr_loading_seconds'),(object_detection,'extract','yolo_seconds_including_loading'),
            (qges,'select','evidence_selection_seconds'),(evidence_decision,'decide','ocr_answer_selection_seconds'),
            (answerability_gate,'decide','learned_gate_decision_seconds'),
            (evidence_store,'extract','evidence_cache_and_extraction_seconds')]:
            wrap(module,name,label)
        if args.reading:
            from phase4_gpu_infer import verify_application
            verify_application(args.run_dir)
            from PIL import Image,ImageOps
            from dataclasses import replace
            from config import Config
            from visionbridge.experimental_reading import is_reading_request,reconstruct
            if not is_reading_request(args.question):raise ValueError('Experimental mode requires an explicit reading request')
            if Path(args.image).name.startswith('VizWiz_'):raise ValueError('This is a demonstration, not an evaluation runner')
            image=ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
            config=replace(Config(),device='cpu',ocr_device='cpu',evidence_cache_dir=Path(args.run_dir)/'experimental-reading-cache')
            evidence,status=evidence_store.extract(image,'ocr',ocr,config)
            start=time.perf_counter();decision=reconstruct(evidence);timings['reading_selection_seconds']=time.perf_counter()-start
            details=dict(decision, mode='EXPERIMENTAL literal OCR reading — not frozen Phase 4',
                baseline_answer=None,final_answer=decision['answer'],evidence=evidence,selected_evidence=[e for e in evidence if e['id'] in decision['supporting_evidence_ids']],
                extraction_status={'ocr':status},blip_status='not run; no BLIP fallback',yolo_status='not required for whole-image reading',
                coordinate_space='displayed rotated image pixels',warnings=['OCR may misread text even at high confidence. No accuracy improvement has been established.'])
            print(json.dumps(dict(display_answer=decision['answer'],status='Experimental OCR reading; incomplete fragments may be omitted.',details=details),ensure_ascii=False))
        else:
            from phase4_gpu_infer import run as frozen_run
            frozen_run(args)
    finally:
        for module,name,original in reversed(originals):setattr(module,name,original)
        timings['worker_body_seconds']=time.perf_counter()-started
        timings['scope']='Inclusive nested timers overlap: do not sum. Missing stage means not called (e.g. OCR cache hit).'
        with (Path(args.image).parent/'stage-timings.json').open('x',encoding='utf-8') as file:json.dump(timings,file,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--image',required=True)
    p.add_argument('--question',required=True);p.add_argument('--reading',action='store_true');run(p.parse_args())
