"""Single frozen-policy holdout execution; resumable images and predictions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import gc
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import uuid
import zipfile
import zlib

from phase4_policy import verify_policy, run_path, verify_previous, P3
from phase3_common import ROOT, MANIFEST, CHECKPOINT, MODEL_CACHE, digest, sha, ensure_new_or_equal, write_new, checked_splits
from prepare_phase3_data import RangeArchive, member_bytes
from prepare_phase2_local import check_image

URL='https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip'


def holdout_rows(root):
    verify_policy(root)
    # This is the only new path that decodes holdout annotations, AFTER freeze.
    _,train,dev,payload=checked_splits()
    rows=json.loads(MANIFEST.read_text(encoding='utf-8-sig'))['holdout']
    excluded={r['image'] for r in train+dev};names=set()
    if len(rows)!=3319:raise ValueError('Expected exactly original 3319 holdout rows')
    for r in rows:
        name=r['image']
        if not name.startswith('VizWiz_val_') or Path(name).name!=name or any(c in name for c in '/\\:') or name in excluded or name in names:
            raise ValueError('Unsafe, overlapping or duplicate holdout ID')
        if len(r['answers'])!=10 or r['answerable'] not in (0,1):raise ValueError('Original reference slots/labels invalid')
        names.add(name)
    return rows,payload


def prepare(root, rows):
    images=root/'holdout/images';ledger=root/'holdout/image-integrity'
    images.mkdir(parents=True,exist_ok=True);ledger.mkdir(exist_ok=True)
    prepared=root/'holdout/prepared.json'
    if prepared.exists():
        saved=json.loads(prepared.read_text())
        if saved['rows_sha256']!=digest(rows):raise ValueError('Prepared membership differs')
        for r in rows:
            p=images/r['image'];entry=json.loads((ledger/(r['image']+'.json')).read_text())
            if sha(p)!=entry['sha256']:raise ValueError('Prepared image changed')
        return
    remote=RangeArchive(URL)
    wanted={r['image'] for r in rows};matches={}
    with zipfile.ZipFile(remote) as archive:
        for info in archive.infolist():
            name=PurePosixPath(info.filename)
            if name.name in wanted:
                if name.is_absolute() or '..' in name.parts or '\\' in info.filename or name.name in matches:raise ValueError('Unsafe archive member')
                matches[name.name]=info
    if set(matches)!=wanted:raise ValueError('Official validation archive missing original images')
    size=sum(i.file_size for i in matches.values())
    existing=sum(p.stat().st_size for p in images.glob('*.jpg'))
    if shutil.disk_usage(root).free<size-existing+2*1024**3:raise RuntimeError('Insufficient image storage plus 2 GiB reserve')
    ensure_new_or_equal(root/'holdout/archive-identity.json',{'url':URL,'archive_bytes':remote.size,'etag':remote.etag,'images':3319,'rows_sha256':digest(rows)})
    def fetch(name):
        p=images/name;l=ledger/(name+'.json');info=matches[name]
        if l.exists():
            e=json.loads(l.read_text())
            if sha(p)!=e['sha256'] or e['crc32']!=info.CRC:raise ValueError('Existing verified image changed')
            return
        if p.exists():data=p.read_bytes()
        else:
            data=member_bytes(remote,info)
            temp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial')
            with temp.open('xb') as f:f.write(data)
            check_image(temp)
            if p.exists():raise FileExistsError('Concurrent image writer')
            temp.rename(p)
        if len(data)!=info.file_size or zlib.crc32(data)&0xffffffff!=info.CRC:raise ValueError('CRC/size mismatch')
        check_image(p)
        write_new(l,{'image':name,'sha256':sha(p),'crc32':info.CRC,'bytes':info.file_size,'source':URL})
    print('HOLDOUT_PREPARATION official validation archive, exact 3319 images',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fetch,n) for n in sorted(wanted,key=lambda n:matches[n].header_offset)]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%50==0 or i==3319:print(f'HOLDOUT_IMAGES {i}/3319 verified',flush=True)
    write_new(prepared,{'images':3319,'rows_sha256':digest(rows),'verified_crc_sha_readable':True})


def execute(path):
    if Path(sys.prefix).name!='.venv-gpu':raise RuntimeError('Use .venv-gpu exclusively')
    root=run_path(path);policy=verify_policy(root);verify_previous()
    if (root/'holdout/summary.json').exists():
        print('HOLDOUT_ALREADY_COMPLETE: existing single evaluation preserved',flush=True);return
    (root/'holdout').mkdir(exist_ok=True)
    rows,payload=holdout_rows(root)
    ensure_new_or_equal(root/'holdout/inputs.json',{'split':'holdout','rows':rows,'rows_sha256':digest(rows),'manifest_sha256':sha(MANIFEST)})
    prepare(root,rows)
    from evaluate_vqa import OfficialScorer
    from modules.answerability_gate import Gate,decide
    from modules.answerability_features import features,image_quality
    from evaluate_phase2_dev import capture,PROTOCOL
    from capture_phase3_features import candidate,checked_entry
    from utils.image_utils import normalize_image
    from modules import vqa
    from config import Config
    from evaluate_phase3_dev import summarize
    import torch
    import numpy as np
    if not torch.cuda.is_available() or '1650' not in torch.cuda.get_device_name(0):raise RuntimeError('Verified GTX 1650 CUDA required')
    if os.environ.get('VISIONBRIDGE_LOW_MEMORY_VQA')=='1':raise RuntimeError('Cannot alter frozen inference precision')
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['YOLO_CONFIG_DIR']=str(root/'runtime/ultralytics')
    os.environ['MPLCONFIGDIR']=str(root/'runtime/matplotlib')
    cfg=replace(Config(),vqa_mode='v2',improved_blip_dir=CHECKPOINT,device='cuda',ocr_device='cpu',object_device='cpu',
        cache_dir=MODEL_CACHE,yolo_model=str(MODEL_CACHE/'yolov8n.pt'),evidence_cache_dir=root/'holdout/evidence-cache',ocr_min_score=.9)
    gate=Gate(P3/'models');scorer=OfficialScorer(payload['official_rules'])
    import importlib.metadata
    identity={'policy_sha256':sha(root/'selected-policy.json'),'freeze_sha256':sha(root/'POLICY_FREEZE.json'),
        'rows_sha256':digest(rows),'protocol':PROTOCOL,'model_sha256':policy['models_sha256'],'torch':torch.__version__,
        'packages':{n:importlib.metadata.version(n) for n in ('transformers','easyocr','ultralytics','scikit-learn','Pillow','numpy')},'cpu_threads':4}
    ensure_new_or_equal(root/'holdout/capture-identity.json',identity)
    for n in ('blip','records'):(root/'holdout'/n).mkdir(exist_ok=True)
    all_records=[];began=time.perf_counter();new=0
    for i,row in enumerate(rows):
        images=root/'holdout/images';p=images/row['image'];image_hash=json.loads((root/'holdout/image-integrity'/(row['image']+'.json')).read_text())['sha256']
        if sha(p)!=image_hash:raise ValueError('Holdout image changed')
        path=root/'holdout/records'/f'{i:04d}.json'
        if path.exists():record=checked_entry(path,row,image_hash)
        else:
            image=normalize_image(p,thumbnail=False);blip_path=root/'holdout/blip'/f'{i:04d}.json'
            if blip_path.exists():prediction=checked_entry(blip_path,row,image_hash)
            else:
                t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                try:answer=vqa.answer(image,row['question'],cfg)
                except torch.cuda.OutOfMemoryError:
                    gc.collect();torch.cuda.empty_cache()
                    with (root/'holdout/oom-retries.jsonl').open('a') as f:f.write(json.dumps({'index':i,'retry':'unchanged protocol once'})+'\n')
                    answer=vqa.answer(image,row['question'],cfg)
                prediction={'image':row['image'],'question':row['question'],'image_sha256':image_hash,'baseline_answer':answer,
                    'blip_seconds':time.perf_counter()-t,'peak_vram_mb':torch.cuda.max_memory_allocated()/1024**2}
                write_new(blip_path,{'record':prediction,'sha256':digest(prediction)});new+=1
            captured=capture({'image':row['image'],'question':row['question']},images,cfg,predict=lambda image,question,cfg:prediction['baseline_answer'])
            if any(s['status']=='unavailable' for s in captured['extraction_status'].values()):
                write_new(root/'holdout'/f'extraction-error-{i}-{time.time_ns()}.json',captured)
                raise RuntimeError('Unavailable evidence extraction; cached BLIP preserved for resume')
            record=candidate(captured);record['timing_seconds']['blip']=prediction['blip_seconds'];record['peak_vram_mb']=prediction['peak_vram_mb']
            record['features']=features(record,image_quality(image),scorer.normalize)
            pa,pr=gate.predict(record['features'],record['final_answer'])
            t=time.perf_counter();decision=decide(record['final_answer'],pa,pr,policy['threshold_answerable'],policy['threshold_reliable'])
            record.update(phase2_final_answer=record['final_answer'],phase2_answer_source=record['answer_source'],
                phase2_supporting_evidence_ids=list(record['supporting_evidence_ids']),final_answer=decision['answer'],
                gate_reason=decision['reason'],answerability_probability=pa,candidate_full_agreement_probability=pr,
                answer_source='abstention' if decision['abstained'] else record['answer_source'],
                supporting_evidence_ids=[] if decision['abstained'] else record['supporting_evidence_ids'])
            record['timing_seconds']['gate_decision']=time.perf_counter()-t
            # This capture contains no reference answers, labels or VQA scores.
            write_new(path,{'record':record,'sha256':digest(record)});del image
        all_records.append(record)
        if (i+1)%10==0 or i+1==3319:print(f'HOLDOUT_CAPTURE {i+1}/3319 cached; new BLIP={new}; elapsed={(time.perf_counter()-began)/60:.1f} min',flush=True)
        if (i+1)%25==0:gc.collect();torch.cuda.empty_cache()
    # Scoring occurs only after all predictions are fixed; no alternative gate policies.
    final=np.array([scorer.score(r['final_answer'],row['answers']) for r,row in zip(all_records,rows)])
    phase2=np.array([scorer.score(r['phase2_final_answer'],row['answers']) for r,row in zip(all_records,rows)])
    blip=np.array([scorer.score(r['baseline_answer'],row['answers']) for r,row in zip(all_records,rows)])
    answered=np.array([scorer.normalize(r['final_answer'])!='unanswerable' for r in all_records])
    original=np.array([scorer.normalize(r['phase2_final_answer'])!='unanswerable' for r in all_records])
    labeled=[{**row,'ocr_supported_question':r['ocr_supported_question']} for row,r in zip(rows,all_records)]
    metrics=summarize(labeled,final,answered,original,phase2);metrics['answer_coverage_percent']=float(100*answered.mean())
    summary={'split':'holdout','examples':3319,'selected':metrics,'baseline_blip_v2_vqa_percent':float(100*blip.mean()),
        'frozen_phase2_vqa_percent':float(100*phase2.mean()),'historical_phase1_holdout_percent':59.48,
        'rows_sha256':digest(rows),'policy_sha256':sha(root/'selected-policy.json'),'no_threshold_selection':True,
        'capture_digest':digest([sha(root/'holdout/records'/f'{i:04d}.json') for i in range(3319)]),
        'timing_recorded_component_seconds':{k:sum(r['timing_seconds'].get(k,0) for r in all_records) for k in ('image_preparation','blip','evidence_extraction','selection','gate_decision')},
        'capture_wall_seconds_this_execution':time.perf_counter()-began,'peak_vram_mb':max(r['peak_vram_mb'] for r in all_records)}
    scored=[{**r,'answers':row['answers'],'answerable':row['answerable'],'final_vqa_score':float(fs),'phase2_vqa_score':float(bs),
        'new_abstention':bool(o and not a),'score_delta':float(fs-bs)} for r,row,fs,bs,o,a in zip(all_records,rows,final,phase2,original,answered)]
    ensure_new_or_equal(root/'holdout/scored_records.json',scored)
    verify_policy(root);verify_previous()
    write_new(root/'holdout/summary.json',summary)
    print('HOLDOUT_EVALUATION_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args();execute(args.run_dir)
