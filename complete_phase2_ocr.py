"""Extract ONLY missing supported dev OCR; replay the unchanged targeted policy.

No BLIP loading, new object detection, training, holdout access or threshold sweep.
New extraction overlays and caches never modify original inference captures.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import time

from config import Config
from evaluate_phase2_dev import replay, summarize, write_new
from evaluate_vqa import OfficialScorer
from export_phase2_dev import validate_dev
from modules import evidence_store
from modules.question_intent import classify
from modules.reliability import mentioned_objects
from replay_phase2_dev import reselect_capture, group
from utils.image_utils import normalize_image
from visionbridge_training import sha256


def missing_indices(captures):
    return [i for i, c in enumerate(captures)
            if classify(c['question']).supported and c['extraction_status']['ocr']['status'] == 'disabled']


def apply_ocr(captured, items, status, elapsed):
    """Only OCR fields/timing change. No references or BLIP prediction supplied."""
    if captured['extraction_status']['ocr']['status'] != 'disabled' or captured['ocr_evidence']:
        raise ValueError('Refusing to replace existing OCR evidence.')
    if any(k in captured for k in ('answers', 'answerable', 'final_vqa_score')):
        raise ValueError('Expected an inference-only capture.')
    if status['status'] not in ('complete', 'empty', 'partial'):
        raise RuntimeError('OCR extraction failed; preserve outputs and diagnose before resuming.')
    if any(i.get('kind') != 'ocr' or i.get('image_id') != captured['image_identity'] for i in items):
        raise ValueError('OCR evidence image identity differs.')
    updated = copy.deepcopy(captured)
    updated['evidence'] = [i for i in captured['evidence'] if i['kind'] != 'ocr'] + items
    updated['ocr_evidence'] = items
    updated['extraction_status']['ocr'] = {k:v for k,v in status.items() if k != 'items'}
    updated['incomplete'] = any(s['status'] in ('partial', 'unavailable') for s in updated['extraction_status'].values())
    updated['timing_seconds']['evidence_extraction'] += elapsed
    return updated


def reader_rows(reader, image):
    # Same readtext defaults and polygon-to-box conversion as modules/ocr.py.
    import numpy as np
    rows = reader.readtext(np.array(image))
    return [{'kind': 'ocr', 'text': text, 'polygon': [[float(x), float(y)] for x,y in polygon],
             'box': [float(min(p[0] for p in polygon)), float(min(p[1] for p in polygon)),
                     float(max(p[0] for p in polygon)), float(max(p[1] for p in polygon))],
             'confidence': float(score)} for polygon,text,score in rows]


def run(source_run, previous_replay, output, checkpoint, resume=False):
    source_run, previous_replay, output, checkpoint = map(lambda p:Path(p).resolve(),
                                                        (source_run, previous_replay, output, checkpoint))
    for protected in (source_run, previous_replay, checkpoint):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Output must be separate from original artifacts/checkpoint.')
    if output.exists() and not resume:
        raise FileExistsError('Choose a new versioned output, or explicitly --resume this coverage run.')
    manifest = source_run/'inputs/run02-dev-only.json'
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    rows = validate_dev(payload)
    scorer = OfficialScorer(payload['official_rules'])
    prior_summary = json.loads((previous_replay/'summary.json').read_text(encoding='utf-8'))
    if prior_summary['rows_sha256'] != payload['rows_sha256'] or prior_summary['threshold'] != .90:
        raise ValueError('Previous targeted replay split/threshold differs.')
    # Freeze every existing evaluation source file used by the previous policy.
    # No policy or preprocessing changes are made to complete coverage.
    if any(sha256(p) != h for p,h in prior_summary['code_sha256'].items()):
        raise ValueError('Previous targeted policy code changed; do not combine a policy change with coverage completion.')
    full = source_run/'evaluation/fp32/full'
    identity = json.loads((full/'identity.json').read_text(encoding='utf-8'))
    if identity['rows_sha256'] != payload['rows_sha256'] or identity['scorer'] != scorer.identity:
        raise ValueError('Original capture identity differs.')
    for name,h in identity['checkpoint_hashes'].items():
        if sha256(checkpoint/name) != h:
            raise ValueError('Checkpoint checksum differs from original evaluation.')
    captures = []
    for index,row in enumerate(rows):
        entry=json.loads((full/'captures'/f'{index:04d}.json').read_text(encoding='utf-8'))
        c=entry['capture']
        if evidence_store.digest(c) != entry['sha256'] or (c['image'],c['question']) != (row['image'],row['question']):
            raise ValueError('Saved inference capture differs.')
        captures.append(c)
    prior_records = json.loads((previous_replay/'targeted_records.json').read_text(encoding='utf-8'))
    if len(prior_records) != 1000:
        raise ValueError('Previous targeted evaluation must contain exactly 1,000 examples.')
    missing=missing_indices(captures)
    prior_missing=[i for i,r in enumerate(prior_records) if r['ocr_supported_question'] and not r['cached_ocr_available']]
    if missing != prior_missing or len(missing) != 17:
        raise ValueError('Expected exactly the same 17 missing supported development examples.')
    for row,c,prior in zip(rows,captures,prior_records):
        if (row['image'],row['question'],row['answers'],row['answerable']) != (prior['image'],prior['question'],prior['answers'],prior['answerable']):
            raise ValueError('Prior reference slots/answerability differ.')
        if c['baseline_answer'] != prior['baseline_answer'] or scorer.score(prior['final_answer'],row['answers']) != prior['final_vqa_score']:
            raise ValueError('Prior prediction or official score differs.')
    planned=[]
    for index in missing:
        c=captures[index];path=source_run/'data/images'/c['image']
        if sha256(path) != c['image_sha256'] or identity['image_hashes'][c['image']] != c['image_sha256']:
            raise ValueError('Required original development image differs: '+c['image'])
        from PIL import Image
        with Image.open(path) as raw: raw.verify()
        planned.append({'index':index,'image':c['image'],'question':c['question'],
                        'image_sha256':c['image_sha256'],'object_targets':mentioned_objects(c['question'])})
    # Protect all earlier input/cache/checkpoint/report files already fingerprinted
    # by the targeted replay, plus every file in that report directory.
    protected_hashes = dict(prior_summary['input_sha256'])
    protected_hashes.update({str(p):sha256(p) for p in previous_replay.rglob('*') if p.is_file()})
    protected_hashes[str(full/'identity.json')] = sha256(full/'identity.json')
    for p in planned:
        path=source_run/'data/images'/p['image'];protected_hashes[str(path)]=p['image_sha256']
    if any(sha256(p) != h for p,h in protected_hashes.items()):
        raise ValueError('Previous artifacts changed before extraction.')
    cfg=Config(device='cpu',ocr_device='cpu',object_device='cpu',vqa_mode='v2',
               cache_dir=source_run/'model-cache',evidence_cache_dir=output/'evidence-cache')
    weight_paths=sorted((cfg.cache_dir/'easyocr').glob('*.pth'))
    from prepare_easyocr_models import file_md5
    expected={'craft_mlt_25k.pth':'2f8227d2def4037cdb3b34389dcf9ec1',
              'english_g2.pth':'5864788e1821be9e454ec108d61b887d'}
    if any(not (cfg.cache_dir/'easyocr'/name).is_file() or file_md5(cfg.cache_dir/'easyocr'/name) != h for name,h in expected.items()):
        raise ValueError('Existing EasyOCR weights missing/invalid; no download or replacement permitted.')
    protected_hashes.update({str(p):sha256(p) for p in weight_paths})
    old_ocr_source=next(c['extraction_status']['ocr']['source'] for c in captures if c['extraction_status']['ocr']['status']=='complete')
    if evidence_store.descriptor('ocr',cfg) != old_ocr_source:
        raise ValueError('EasyOCR weights/package/settings differ from original extraction.')
    guard={'rows_sha256':payload['rows_sha256'],'missing_examples':planned,
           'prior_policy_code_sha256':prior_summary['code_sha256'],'protected_sha256':protected_hashes,
           'runner_sha256':sha256(__file__),'threshold':.90,'ocr_device':'cpu',
           'blip_inference_calls':0,'object_inference_calls':0}
    if not output.exists():
        output.mkdir(parents=True,exist_ok=False)
        write_new(output/'identity.json',guard)
        write_new(output/'missing_ocr_examples.json',planned)
        (output/'ocr-overlays').mkdir()
    elif not (output/'identity.json').is_file() or json.loads((output/'identity.json').read_text()) != guard:
        raise ValueError('Resume identity differs; preserve output and choose a new run.')
    if (output/'results').exists():
        raise FileExistsError('Coverage results already complete; preserve them.')
    import torch
    torch.manual_seed(42)
    reader=None;new_calls=0;started=time.perf_counter()
    for n,index in enumerate(missing,1):
        original=captures[index];dest=output/'ocr-overlays'/f'{index:04d}.json'
        if dest.exists():
            entry=json.loads(dest.read_text(encoding='utf-8'))
            if entry['original_capture_sha256'] != evidence_store.digest(original) or entry['sha256'] != evidence_store.digest(entry['capture']):
                raise ValueError('Resume overlay checksum differs.')
            c=entry['capture'];status=c['extraction_status']['ocr']
            # Reconstruct OCR-only changes to prohibit altered BLIP/object evidence.
            reconstructed=apply_ocr(original,c['ocr_evidence'],status,entry['extraction_seconds'])
            if reconstructed != c:raise ValueError('Resume overlay changed non-OCR capture data.')
            captures[index]=c
            print(f'OCR {n}/17 reused overlay: {original["image"]}',flush=True)
            continue
        if reader is None:
            import easyocr
            reader=easyocr.Reader(['en'],gpu=False,model_storage_directory=str(cfg.cache_dir/'easyocr'),
                                  user_network_directory=str(output/'easyocr-user-network'),
                                  download_enabled=False,verbose=False)
        image=normalize_image(source_run/'data/images'/original['image'],thumbnail=False)
        if evidence_store.image_identity(image) != original['image_identity']:
            raise ValueError('Decoded image/preprocessing identity differs.')
        print(f'OCR {n}/17 extracting on CPU: {original["image"]}',flush=True)
        before=time.perf_counter()
        component=SimpleNamespace(extract=lambda im,config:reader_rows(reader,im))
        items,status=evidence_store.extract(image,'ocr',component,cfg)
        elapsed=time.perf_counter()-before;image.close();new_calls+=not status.get('cache_hit',False)
        updated=apply_ocr(original,items,status,elapsed)
        write_new(dest,{'original_capture_sha256':evidence_store.digest(original),'sha256':evidence_store.digest(updated),
                        'capture':updated,'extraction_seconds':elapsed})
        captures[index]=updated
        print(f'  {status["status"]}; {len(items)} OCR fragments; {elapsed:.1f}s',flush=True)
    results=[]
    for row,c in zip(rows,captures):
        result=replay(row,reselect_capture(c),scorer,.90)
        result.update(policy='targeted-ocr-v1-expanded-coverage',
                      original_ocr_supported_question=c['ocr_supported_question'],
                      original_ocr_question_intent=c['ocr_question_intent'],
                      cached_ocr_available=c['extraction_status']['ocr']['status']!='disabled')
        results.append(result)
    untouched=[i for i in range(1000) if i not in missing]
    if any((results[i]['final_answer'],results[i]['final_vqa_score'],results[i]['evidence']) != (
            prior_records[i]['final_answer'],prior_records[i]['final_vqa_score'],prior_records[i]['evidence']) for i in untouched):
        raise ValueError('Unchanged 983 examples differ; policy was not held fixed.')
    if any(sha256(p) != h for p,h in protected_hashes.items()) or any(sha256(p) != h for p,h in prior_summary['code_sha256'].items()):
        raise RuntimeError('Protected artifacts/policy changed; refusing results.')
    target=summarize(results)
    summary={'purpose':'development-only OCR coverage completion; not held-out accuracy','threshold':.90,
             'policy':'targeted-ocr-v1-expanded-coverage','baseline':prior_summary['baseline'],
             'original_ocr':prior_summary['original_ocr'],'previous_targeted':prior_summary['targeted'],
             'targeted':target,'rows_sha256':payload['rows_sha256'],'new_ocr_images':17,
             'new_extraction_calls_this_invocation':int(new_calls),'unchanged_examples_verified':983,
             'total_wall_seconds':time.perf_counter()-started,'replay_seconds':None,
             'protected_sha256':protected_hashes,'code_sha256':prior_summary['code_sha256'],
             'verified_cache_files':prior_summary['verified_cache_files'],'blip_inference_calls':0,
             'object_inference_calls':0,'gpu_inference_calls':0,
             'delta_vqa_percentage_points':target['vqa_agreement_percent']-prior_summary['baseline']['vqa_agreement_percent'],
             'delta_vs_previous_targeted_percentage_points':target['vqa_agreement_percent']-prior_summary['targeted']['vqa_agreement_percent']}
    from collections import Counter
    summary['remaining_reasons']=dict(Counter(r['decision_reason'] for r in results if r['answer_source']=='blip'))
    summary['new_ocr_status_counts']=dict(Counter(captures[i]['extraction_status']['ocr']['status'] for i in missing))
    summary['missing17_subgroup']={
        'baseline':group([dict(prior_records[i],final_vqa_score=prior_records[i]['baseline_vqa_score']) for i in missing]),
        'prior_targeted':group([prior_records[i] for i in missing]),'expanded':group([results[i] for i in missing])}
    old_ids={c['image'] for c in captures if c['ocr_supported_question']}
    revised_ids={r['image'] for r in results if r['ocr_supported_question']}
    baseline_rows=[dict(r,final_vqa_score=r['baseline_vqa_score']) for r in results]
    summary['frozen_original_ocr_subgroup']={'baseline':group([r for r in baseline_rows if r['image'] in old_ids]),
                                           'targeted':group([r for r in results if r['image'] in old_ids])}
    summary['revised_router_ocr_subgroup']={'baseline':group([r for r in baseline_rows if r['image'] in revised_ids]),
                                         'targeted':group([r for r in results if r['image'] in revised_ids]),
                                         'missing_cached_ocr':sum(not r['cached_ocr_available'] for r in results if r['image'] in revised_ids)}
    corrected=[r for r in results if r['ocr_corrected_blip']];worsened=[r for r in results if r['ocr_worsened_blip']]
    dest=output/'results';dest.mkdir()
    for name,data in [('summary.json',summary),('expanded_records.json',results),('corrected_examples.json',corrected),
                     ('worsened_examples.json',worsened),('new17_records.json',[results[i] for i in missing])]:
        write_new(dest/name,data)
    with (dest/'coverage_comparison.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['configuration','examples','official_vqa_percent','overrides','beneficial','harmful','neutral'])
        for name,s in [('BLIP baseline',summary['baseline']),('Prior targeted OCR',summary['previous_targeted']),('Expanded OCR',target)]:
            writer.writerow([name,s['examples'],s['vqa_agreement_percent'],s['ocr_overrides'],s['beneficial_overrides'],s['harmful_overrides'],s['neutral_overrides']])
    write_coverage_report(dest,summary,[results[i] for i in missing])
    return dest


def write_coverage_report(dest,s,new_rows):
    lines=['# Phase 2 development: completed OCR coverage','',
           'Exact run02 development split, n=1,000. Same targeted policy and threshold 0.90. EasyOCR CPU extraction only for the 17 originally disabled supported questions; all BLIP predictions and the other 983 examples are unchanged. No training, new object detection, or holdout access.','',
           '| Configuration | VQA agreement | Overrides | Beneficial | Harmful | Neutral |','|---|---:|---:|---:|---:|---:|']
    for name,k in [('Original BLIP baseline','baseline'),('Previous targeted replay','previous_targeted'),('Expanded OCR coverage','targeted')]:
        v=s[k];lines.append(f'| {name} | {v["vqa_agreement_percent"]:.2f}% | {v["ocr_overrides"]} | {v["beneficial_overrides"]} | {v["harmful_overrides"]} | {v["neutral_overrides"]} |')
    lines+=['',f'Additional change versus 59.15%: **{s["delta_vs_previous_targeted_percentage_points"]:+.2f} percentage points**. '+('Additional development improvement is confirmed by official scoring.' if s['delta_vs_previous_targeted_percentage_points']>0 else 'No additional development improvement is confirmed.'),'',
            '| Subgroup | n | Baseline | Expanded |','|---|---:|---:|---:|']
    for name in ['answerable','unanswerable']:
        a=s['baseline']['subgroups'][name];b=s['targeted']['subgroups'][name];lines.append(f'| {name} | {b["examples"]} | {a["vqa_agreement_percent"]:.2f}% | {b["vqa_agreement_percent"]:.2f}% |')
    for name,k in [('Original OCR-supported subgroup','frozen_original_ocr_subgroup'),('Revised OCR-supported subgroup','revised_router_ocr_subgroup')]:
        a=s[k]['baseline'];b=s[k]['targeted'];lines.append(f'| {name} | {b["examples"]} | {a["vqa_agreement_percent"]:.2f}% | {b["vqa_agreement_percent"]:.2f}% |')
    a=s['missing17_subgroup']['baseline'];b=s['missing17_subgroup']['expanded'];lines.append(f'| Newly covered questions | 17 | {a["vqa_agreement_percent"]:.2f}% | {b["vqa_agreement_percent"]:.2f}% |')
    lines+=['',f'New OCR statuses: {s["new_ocr_status_counts"]}. Missing supported OCR captures after completion: {s["revised_router_ocr_subgroup"]["missing_cached_ocr"]}. Empty or partial extraction is recorded, never replaced with fabricated evidence.','',
            '## Exact 17 examples and decisions','',
            '| Image | Question | OCR fragments | BLIP → final | Score before → after | Decision |','|---|---|---:|---|---|---|']
    for r in new_rows:
        vals=[r['image'],r['question'].strip(),len(r['ocr_evidence']),r['baseline_answer']+' → '+r['final_answer'],
              f'{r["baseline_vqa_score"]:.2f} → {r["final_vqa_score"]:.2f}',r['decision_reason']]
        lines.append('| '+' | '.join(str(v).replace('|','\\|').replace('\n',' ') for v in vals)+' |')
    lines+=['','## Remaining first rejection reasons','','| Reason | Count |','|---|---:|']
    for reason,count in sorted(s['remaining_reasons'].items(),key=lambda a:-a[1]):lines.append(f'| {reason} | {count} |')
    lines+=['','## Preservation and limits','',
            '- Original input/checkpoint/cache/report hashes matched before and after extraction/replay. All 1,000 original questions, reference-answer slots and answerability labels are preserved.',
            '- Extraction used the same verified English EasyOCR weights/default readtext settings, EXIF-corrected RGB images at full resolution and CPU. Downloads were disabled. New evidence caches/overlays are saved in this coverage run only.',
            '- The existing BLIP-abstention, uncertainty and ambiguity safeguards remain active. Missing object detection is not invented or bypassed; the bottle question may remain unselected because no target region was cached.',
            '- Previously disabled captures remain intact. The 17 new overlays are checksum-protected and resumable. No new OCR runs for the other 983 examples.',
            '- Only new OCR extraction time was measured; historical BLIP/OCR timing on other examples is reused, so this is not a new end-to-end latency benchmark.',
            '- No held-out claim follows from development scores. Historical holdout baseline remains 59.48% and was not accessed. Stop after this development evaluation.']
    with (dest/'COVERAGE_REPORT.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['source-run','previous-replay','output','checkpoint']:parser.add_argument('--'+arg,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    print(run(args.source_run,args.previous_replay,args.output,args.checkpoint,args.resume))
