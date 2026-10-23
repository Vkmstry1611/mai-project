"""Private stdio worker prototype. Reuses models; never edits frozen sources."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import time


class FileGuard:
    """Fast per-request change detection after startup SHA verification.

    This is a local-session guard, not protection against an adversary restoring
    file timestamps. Restart/reverify after any artifact or source change.
    """
    def __init__(self,paths):
        self.snapshot={Path(p):(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in paths}

    def check(self):
        for path,expected in self.snapshot.items():
            try:actual=(path.stat().st_size,path.stat().st_mtime_ns)
            except OSError as error:raise RuntimeError('Session artifact disappeared; restart: '+str(path)) from error
            if actual!=expected:raise RuntimeError('Session artifact changed; restart and reverify: '+str(path))


class Engine:
    def __init__(self,run_dir):
        start=time.perf_counter();self.timings={};self.originals=[];self.pipeline=None
        from phase4_gpu_infer import verify_application,prime_cuda
        self.root=verify_application(run_dir)
        self.verification_seconds=time.perf_counter()-start
        os.environ['HF_HUB_OFFLINE']='1'
        # Standalone experimental reading previously used PyTorch's default CPU
        # thread count; frozen CUDA VQA explicitly uses four. Preserve both.
        import torch
        self.reading_threads=torch.get_num_threads()
        cuda_start=time.perf_counter();self.torch=prime_cuda();self.cuda_seconds=time.perf_counter()-cuda_start
        from modules import vqa,ocr,object_detection,qges,evidence_store,evidence_decision,answerability_gate
        self.ocr=ocr;self.evidence_store=evidence_store
        for obj,name,label in [(vqa,'answer','blip_seconds_including_loading'),(vqa,'load_blip','blip_loader_seconds'),
            (ocr,'extract','ocr_seconds_including_loading'),(ocr,'load_reader','ocr_loader_seconds'),
            (object_detection,'extract','yolo_seconds_including_loading'),(qges,'select','evidence_selection_seconds'),
            (evidence_decision,'decide','ocr_answer_selection_seconds'),(answerability_gate,'decide','learned_gate_decision_seconds'),
            (evidence_store,'extract','evidence_cache_and_extraction_seconds')]:
            self.wrap(obj,name,label)
        from phase3_common import ROOT,CHECKPOINT,MODEL_CACHE,MANIFEST,DEV_INPUT,FREEZE
        from phase4_policy import P3
        paths=list(ROOT.glob('*.py'))+list((ROOT/'modules').glob('*.py'))+list((ROOT/'utils').glob('*.py'))
        paths+=list(CHECKPOINT.glob('*'))+list((MODEL_CACHE/'easyocr').glob('*.pth'))+list((ROOT/'models/cache/easyocr').glob('*.pth'))
        paths+=list((P3/'models').glob('*'))+[MANIFEST,DEV_INPUT,FREEZE,P3/'selected-policy.json',MODEL_CACHE/'yolov8n.pt']
        paths+=[self.root/name for name in ('APPLICATION_FREEZE.json','POLICY_FREEZE.json','selected-policy.json')]
        self.guard=FileGuard([p for p in paths if p.is_file()])
        self.startup_seconds=time.perf_counter()-start

    def wrap(self,obj,name,label):
        original=getattr(obj,name)
        def observed(*a,**kw):
            start=time.perf_counter()
            try:return original(*a,**kw)
            finally:self.timings[label]=self.timings.get(label,0)+time.perf_counter()-start
        self.originals.append((obj,name,original));setattr(obj,name,observed)

    def infer(self,path,question,reading):
        from PIL import Image,ImageOps
        if Path(path).name.startswith('VizWiz_'):raise ValueError('Demonstration inputs only; no evaluation dataset requests')
        if not isinstance(question,str) or not question.strip() or len(question)>1000:raise ValueError('Provide a question of 1–1000 characters')
        self.guard.check();self.timings={};start=time.perf_counter()
        self.torch.set_num_threads(self.reading_threads if reading else 4)
        self.torch.cuda.reset_peak_memory_stats()
        if reading:
            from dataclasses import replace
            from config import Config
            from experimental_reading import is_reading_request,reconstruct
            if not is_reading_request(question):raise ValueError('Explicit text-reading question required')
            image=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
            cfg=replace(Config(),device='cpu',ocr_device='cpu',evidence_cache_dir=self.root/'experimental-reading-cache')
            evidence,status=self.evidence_store.extract(image,'ocr',self.ocr,cfg)
            tick=time.perf_counter();decision=reconstruct(evidence);self.timings['reading_selection_seconds']=time.perf_counter()-tick
            details=dict(decision,mode='EXPERIMENTAL literal OCR reading — not frozen Phase 4',baseline_answer=None,
                final_answer=decision['answer'],evidence=evidence,selected_evidence=[e for e in evidence if e['id'] in decision['supporting_evidence_ids']],
                extraction_status={'ocr':status},blip_status='not run; no BLIP fallback',yolo_status='not required for whole-image reading',
                coordinate_space='displayed rotated image pixels',warnings=['OCR may misread text even at high confidence. No accuracy improvement has been established.'])
            result=dict(display_answer=decision['answer'],status='Experimental OCR reading; incomplete fragments may be omitted.',details=details)
        else:
            self.torch.manual_seed(42);self.torch.cuda.manual_seed_all(42)
            if self.pipeline is None:
                tick=time.perf_counter()
                from phase4_pipeline import Phase4Pipeline
                self.pipeline=Phase4Pipeline(self.root)
                self.timings['pipeline_initialization_seconds']=time.perf_counter()-tick
            shown,_,status,debug,_=self.pipeline.run(Path(path),question)
            debug['warnings']=[w for w in debug.get('warnings',[]) if 'Learned answerability is not implemented.' not in w]
            debug['gpu_runtime']={'device':self.torch.cuda.get_device_name(0),'cuda_initialized_before_cpu_evidence':True,
                'precision':'FP32 model weights, CUDA FP16 autocast','CPU_evidence':True}
            result=dict(display_answer=shown,status=status,details=debug)
        self.timings.update(inference_seconds=time.perf_counter()-start,
            peak_allocated_mib=self.torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=self.torch.cuda.max_memory_reserved()/2**20)
        return result,dict(self.timings)


def serve(run_dir):
    if Path(sys.prefix).name!='.venv-gpu':raise RuntimeError('Use .venv-gpu for the worker')
    protocol=sys.stdout
    def send(value):protocol.write(json.dumps(value,ensure_ascii=False)+'\n');protocol.flush()
    try:
        with redirect_stdout(sys.stderr):engine=Engine(run_dir)
        send(dict(ready=True,startup_seconds=engine.startup_seconds,verification_seconds=engine.verification_seconds,cuda_seconds=engine.cuda_seconds))
    except Exception as error:
        send(dict(ready=False,error=str(error)));return
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('action')=='shutdown':send({'stopped':True});break
            if request.get('action')!='infer':raise ValueError('Unknown worker action')
            with redirect_stdout(sys.stderr):result,timings=engine.infer(request['image'],request['question'],bool(request.get('reading')))
            send(dict(ok=True,result=result,timings=timings))
        except Exception as error:
            import traceback
            traceback.print_exc(file=sys.stderr)
            send(dict(ok=False,error=str(error)))
            if 'out of memory' in str(error).lower():break  # Restart explicitly; no precision/CPU fallback.


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);serve(p.parse_args().run_dir)
