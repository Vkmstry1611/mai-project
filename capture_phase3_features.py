"""Training-only resumable CUDA BLIP/CPU evidence capture, batch size one."""
import argparse
from dataclasses import replace
import gc
import json
import importlib.metadata
import os
from pathlib import Path
import sys
import time

from config import Config
from evaluate_phase2_dev import capture, PROTOCOL
from evaluate_vqa import OfficialScorer
from modules import evidence_decision, fusion, vqa
from modules.answerability_features import features, image_quality, SCHEMA
from phase3_common import checked_splits, CHECKPOINT, MODEL_CACHE, ROOT, ensure_new_or_equal, digest, safe_run, sha, verify_freeze, write_new
from utils.image_utils import normalize_image


def candidate(captured):
    inputs=fusion.FusionInput(captured['question'],captured['baseline_answer'],captured['selected_evidence'],{},
        selection_truncated=captured.get('selection_truncated',captured['incomplete'] and not any(s.get('status') in ('partial','unavailable') for s in captured.get('extraction_status',{}).values())),
        ocr_min_score=.9,extraction_status=captured['extraction_status'])
    result=evidence_decision.decide(inputs)
    return {**captured,'final_answer':result.answer,'decision':result.decision,'decision_reason':result.reason,
            'answer_source':result.source,'supporting_evidence_ids':result.supporting_evidence_ids,'candidate_trace':result.candidate_trace}


def checked_entry(path, row, image_hash):
    payload=json.loads(path.read_text())
    value=payload['record']
    if payload['sha256']!=digest(value) or value['image']!=row['image'] or value['question']!=row['question'] or value['image_sha256']!=image_hash:
        raise ValueError(f'Capture identity/integrity mismatch: {path}')
    return value


def execute(run_dir, limit=5000):
    if Path(sys.prefix).name!='.venv-gpu': raise RuntimeError('Use .venv-gpu exclusively')
    root=safe_run(run_dir)
    verify_freeze()
    _,train,_,payload=checked_splits()
    if not 1<=limit<=5000: raise ValueError('Training capture limit must be 1..5000')
    if not (root/'data/prepared.json').is_file(): raise RuntimeError('Exact training images are not verified yet')
    prepared=json.loads((root/'data/prepared.json').read_text())
    if prepared['train_rows_sha256']!=digest(train): raise ValueError('Training image preparation does not match manifest')
    os.environ['YOLO_CONFIG_DIR']=str(root/'runtime/ultralytics')
    os.environ['MPLCONFIGDIR']=str(root/'runtime/matplotlib')
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    if not torch.cuda.is_available() or '1650' not in torch.cuda.get_device_name(0): raise RuntimeError('Verified GTX 1650 CUDA required')
    if os.environ.get('VISIONBRIDGE_LOW_MEMORY_VQA')=='1': raise RuntimeError('Do not mix half-weight captures with frozen FP32 baseline protocol')
    torch.set_num_threads(4)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    print('GPU',torch.cuda.get_device_name(0),'free/total',torch.cuda.mem_get_info(),flush=True)
    scorer=OfficialScorer(payload['official_rules'])
    cfg=replace(Config(),vqa_mode='v2',improved_blip_dir=CHECKPOINT,device='cuda',ocr_device='cpu',object_device='cpu',
                cache_dir=MODEL_CACHE,yolo_model=str(MODEL_CACHE/'yolov8n.pt'),evidence_cache_dir=root/'evidence-cache',ocr_min_score=.9)
    vqa.model_source(cfg)
    image_hashes={r['image']:json.loads((root/'data/image-integrity'/(r['image']+'.json')).read_text())['sha256'] for r in train}
    codes=['phase3_common.py','capture_phase3_features.py','modules/answerability_features.py']
    identity={'schema':1,'split':'train','rows_sha256':digest(train),'source_manifest_sha256':payload['source_manifest_sha256'],
              'feature_schema':SCHEMA,'protocol':PROTOCOL,'ocr_threshold':.9,'device':'cuda','ocr_device':'cpu','object_device':'cpu',
              'source_sha256':{p:sha(ROOT/p) for p in codes},'image_sha256':image_hashes,'scorer':scorer.identity,
              'torch':torch.__version__,'cpu_threads':4}
    identity['packages']={p:importlib.metadata.version(p) for p in ('torch','torchvision','transformers','Pillow','easyocr','ultralytics','numpy','safetensors')}
    ensure_new_or_equal(root/'captures/identity.json',identity)
    for name in ('blip','train'): (root/'captures'/name).mkdir(exist_ok=True)
    start=time.perf_counter(); reused=0; new_calls=0
    for index,row in enumerate(train[:limit]):
        image_path=root/'data/train-images'/row['image']
        if sha(image_path)!=image_hashes[row['image']]: raise ValueError(f'Training image changed: {row["image"]}')
        path=root/'captures/train'/f'{index:04d}.json'
        if path.exists():
            checked_entry(path,row,image_hashes[row['image']]); reused+=1
        else:
            image=normalize_image(image_path,thumbnail=False)
            blip_path=root/'captures/blip'/f'{index:04d}.json'
            if blip_path.exists():
                prediction=checked_entry(blip_path,row,image_hashes[row['image']])
            else:
                began=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
                try:
                    answer=vqa.answer(image,row['question'],cfg)
                except torch.cuda.OutOfMemoryError:
                    gc.collect(); torch.cuda.empty_cache()
                    with (root/'captures/oom-retries.jsonl').open('a') as log: log.write(json.dumps({'index':index,'action':'same-protocol garbage collection and cache clear; one retry'})+'\n')
                    answer=vqa.answer(image,row['question'],cfg)
                prediction={'image':row['image'],'question':row['question'],'image_sha256':image_hashes[row['image']],
                            'baseline_answer':answer,'blip_seconds':time.perf_counter()-began,'peak_vram_mb':torch.cuda.max_memory_allocated()/1024**2}
                write_new(blip_path,{'record':prediction,'sha256':digest(prediction)}); new_calls+=1
            # Reuse saved BLIP even if evidence extraction was interrupted.
            captured=capture({'image':row['image'],'question':row['question']},root/'data/train-images',cfg,
                             predict=lambda image,question,cfg:prediction['baseline_answer'])
            if any(s.get('status')=='unavailable' for s in captured['extraction_status'].values()):
                write_new(root/'captures'/f'extraction-error-{index:04d}-{time.time_ns()}.json',captured)
                raise RuntimeError('Evidence extraction unavailable; BLIP prediction preserved for safe resume')
            captured['timing_seconds']['blip']=prediction['blip_seconds']
            record=candidate(captured); record['peak_vram_mb']=prediction['peak_vram_mb']
            record['image_quality']=image_quality(image)
            record['features']=features(record,record['image_quality'],scorer.normalize)
            write_new(path,{'record':record,'sha256':digest(record)})
            del image,captured,record
        if (index+1)%10==0 or index+1==limit:
            print(f'TRAIN_CAPTURE {index+1}/{limit} saved; new BLIP={new_calls}; reused={reused}; elapsed={(time.perf_counter()-start)/60:.1f} min',flush=True)
        if (index+1)%25==0: gc.collect(); torch.cuda.empty_cache()
    verification={'split':'train','examples':limit,'capture_digest':digest([sha(root/'captures/train'/f'{i:04d}.json') for i in range(limit)]),
                  'holdout_accessed':False,'pilot_debug_only':limit<5000,'all_required_captures_complete':True}
    ensure_new_or_equal(root/'captures'/('pilot-verification.json' if limit<5000 else 'training-verification.json'),verification)
    print('TRAINING_PILOT_PASSED' if limit<5000 else 'TRAINING_CAPTURE_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir',required=True);p.add_argument('--limit',type=int,default=5000)
    args=p.parse_args(); execute(args.run_dir,args.limit)
