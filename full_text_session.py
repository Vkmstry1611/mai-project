"""Unified experimental facade; general VQA and manual OCR delegate unchanged."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid
from launch_persistent_demo import Session
from persistent_demo_client import WorkerClient,ROOT

FULL='Full Text Reading (experimental)'


class FullTextClient(WorkerClient):
    environment='.venv-ocr'
    worker='full_text_worker.py'
    def __init__(self,run_dir):
        super().__init__(run_dir)
        self.temporary=tempfile.TemporaryDirectory(prefix='visionbridge-fulltext-')
        self.base=Path(self.temporary.name)
        self.cache_namespace='cache'

    def start(self):
        with self.lock:
            if self.process is not None:
                if self.process.poll() is not None:raise RuntimeError('Full-text worker stopped; restart workspace')
                return
            executable=ROOT/self.environment/'Scripts/python.exe'
            if not executable.is_file():raise FileNotFoundError('Run setup_full_text.ps1 to install the isolated OCR environment')
            self.log=(self.base/'worker.log').open('x',encoding='utf-8')
            startup_tick=time.perf_counter()
            self.process=subprocess.Popen([str(executable),'-u',str(ROOT/self.worker),'--session-dir',str(self.base)],
                cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',
                env=dict(os.environ,PYTHONIOENCODING='utf-8'),creationflags=0x08000000)
            threading.Thread(target=self._read,daemon=True).start()
            ready=self.receive();self.startup=ready
            self.startup['observed_process_startup_seconds']=time.perf_counter()-startup_tick
            if not ready.get('ready'):self.close(force=True);raise RuntimeError(ready.get('error','OCR startup failed'))

    def read(self,image,manual=None):
        with self.lock:
            self.start()
            path=self.base/(uuid.uuid4().hex+'.png');image.save(path)
            self.process.stdin.write(json.dumps(dict(action='read',image=str(path),manual=manual,cache_namespace=self.cache_namespace))+'\n');self.process.stdin.flush()
            response=self.receive()
            if not response.get('ok'):raise RuntimeError('Full-text OCR failed: '+response.get('error','unknown'))
            return response

    def close(self,force=False):
        super().close(force)
        self.temporary.cleanup()


class FullTextSession:
    def __init__(self,run_dir):
        self.run_dir=Path(run_dir).resolve();self.legacy=Session(run_dir);self.reader=FullTextClient(run_dir)
        self.easy=EasyTextClient(run_dir);self.engine_choice='Automatic'
        self.scene=SceneClient(run_dir)

    def set_engine(self,value):
        if value not in ('Automatic','EasyOCR','RapidOCR'):raise ValueError('Unsupported OCR engine')
        self.engine_choice=value
        return 'Experimental OCR engine: '+value+'. General VQA is unchanged.'

    def for_conversation(self,state,image):
        from image_conversation import ConversationSession
        return ConversationSession(self,state,image)

    def classify(self,question):
        import re
        from workspace_intent import classify,Route
        from ingredient_reading import intent as ingredient_intent
        if ingredient_intent(question):return Route('reading','Experimental OCR ingredient mentions; no visual ingredient inference.')
        from text_hierarchy import request
        if request(question):return Route('reading','Explicit experimental OCR hierarchy selection.')
        from assistive_tools import field_intent
        route=classify(question)
        q=str(question)
        if field_intent(question) in ('serving size','net quantity','calories') and not re.search(r'\b(colou?r|shape|doing|translate)\b',q,re.I):
            return Route('targeted','Experimental printed nutrition/quantity field; explicit labels and unique values required.')
        from reading_relevance import intent
        if not re.search(r"\b(colou?r|shape|doing|how many|count|translate|meaning|never|don't|not)\b",q,re.I) and intent(q) in ('all','interface','summary','text_question','instructions','heating'):
            return Route('reading','Experimental question-aware OCR; literal/extractive evidence only.')
        if re.search(r'\b(?:exact|specific)\s+(?:product|brand|item)\b',q,re.I):
            return Route('targeted','Exact identity requires an explicit printed product/brand field; no product guessing.')
        full_read=bool(re.search(r'\bread\s+(?:everything|all|the\s+(?:whole|entire|complete))|\b(?:complete|entire|whole)\s+(?:text|page|paragraph)\b',q,re.I))
        blocked=bool(re.search(r"\b(?:not|don't|never|translate|colou?r|shape|meaning|doing|count)\b",q,re.I))
        if not blocked and (full_read or re.search(r'\btranscribe\b',q,re.I)):
            return Route('reading','Explicit whole-text transcription; no field selection or semantic completion.')
        if (route.task!='clarify' and field_intent(question) and
                not re.search(r'\b(colou?r|shape|doing|how many)\b',str(question),re.I)):
            return Route('targeted','Experimental label-to-value OCR; insufficient/conflicting evidence abstains.')
        return route

    def request(self,*args,**kwargs):return self.legacy.request(*args,**kwargs)

    def reading_cache_key(self,image,rotation=None):
        from modules.evidence_store import digest,image_identity
        from full_text_ocr import VERSION,ENGLISH_SHA
        return digest([image_identity(image),self.engine_choice,rotation,VERSION,ENGLISH_SHA,'easy-reading-original-v1-threads4'])

    def cached_reading_evidence(self,image,cache):
        """Current image/config only. Never starts an OCR worker on a cache miss."""
        import copy
        from PIL import ImageOps
        from modules.evidence_store import image_identity
        image=ImageOps.exif_transpose(image).convert('RGB')
        payload=cache.get(self.reading_cache_key(image))
        if not payload:return None
        result=payload.get('result',{})
        if result.get('image_id')!=image_identity(image):return None
        return copy.deepcopy(result)

    def request_auto(self,image,question,speak,experimental,rotation='0'):
        return self.full_request(image,question,speak,None,target_field=self.classify(question).task=='targeted')

    def full_request(self,image,question,speak=False,rotation=None,target_field=False,evidence_cache=None,omit_ids=None,selected_region_ids=None):
        from PIL import ImageOps
        from utils.image_utils import annotate
        from submission_ui_helpers import evidence_text
        from assistive_tools import field_intent,targeted
        start=time.perf_counter();image=ImageOps.exif_transpose(image).convert('RGB')
        preparation_seconds=time.perf_counter()-start
        from reading_relevance import compose,local_quality
        from modules.evidence_store import digest,image_identity
        from full_text_ocr import VERSION,ENGLISH_SHA
        import copy
        engine_choice=self.engine_choice;fallback=None
        reader=self.easy if engine_choice=='EasyOCR' else self.reader
        key_start=time.perf_counter()
        key=self.reading_cache_key(image,rotation)
        identity_seconds=time.perf_counter()-key_start
        primary_seconds=0.;fallback_seconds=0.;startup_seconds=0.
        reused=evidence_cache is not None and key in evidence_cache
        if reused:
            response=copy.deepcopy(evidence_cache[key]);response['result']['cache_hit']=True
            response['result']['original_extraction_seconds']=response['result'].get('request_seconds')
            response['result']['request_seconds']=0.
        else:
            was_running=getattr(reader,'process',None) is not None
            tick=time.perf_counter();response=reader.read(image,None if rotation is None else int(rotation));primary_seconds=time.perf_counter()-tick
            if not was_running:startup_seconds+=reader.startup.get('observed_process_startup_seconds',0.)
        details=response['result']
        # Fallback only after insufficient evidence, never as unconditional dual OCR.
        accepted=[r for r in details['evidence'] if r['confidence']>=(.9 if engine_choice=='EasyOCR' else .75)]
        if not reused and engine_choice=='Automatic' and (not accepted or details.get('extraction_status',{}).get('ocr',{}).get('status')=='unavailable'):
            fallback='EasyOCR: primary returned no accepted evidence'
            original=details
            try:
                was_running=getattr(self.easy,'process',None) is not None
                tick=time.perf_counter()
                response=self.easy.read(image,details['orientation']['selected_ccw_degrees']);details=response['result']
                fallback_seconds=time.perf_counter()-tick
                if not was_running:startup_seconds+=self.easy.startup.get('observed_process_startup_seconds',0.)
                details=dict(details,alternate_engine_evidence=original,engine_fallback=fallback)
            except Exception as error:
                fallback_seconds=time.perf_counter()-tick
                details=dict(original,engine_fallback=fallback,fallback_error=str(error))
        if evidence_cache is not None and not reused and not details.get('fallback_error'):
            evidence_cache[key]=copy.deepcopy(response)
            evidence_cache[key]['result']=copy.deepcopy(details)
            while len(evidence_cache)>2:evidence_cache.pop(next(iter(evidence_cache)))
        details=compose(question,details,omit_ids=omit_ids,selected_region_ids=selected_region_ids)
        details['local_quality']=local_quality(details['evidence'],image.size)
        details['engine_selection']={'requested':engine_choice,'selected':details['source']['engine'],'fallback':fallback,
            'rule':'RapidOCR first based on permitted fixture probes; EasyOCR only if no accepted primary evidence. No cross-engine score comparison.'}
        details['worker_startup']=reader.startup
        if fallback:details['fallback_worker_startup']=self.easy.startup
        field_question=question if field_intent(question) else ('What is the product name?' if target_field else question)
        if target_field and field_intent(field_question):
            # Relation tests use the corrected reading plane, not the sideways
            # original coordinates. Preserve the source IDs and both box spaces.
            decision=targeted(field_question,[dict(i,box=i['corrected_box']) for i in details['evidence']])
            details=dict(details,**decision,final_answer=decision['answer'],
                selected_evidence=[i for i in details['evidence'] if i['id'] in decision['supporting_evidence_ids']])
            used=set(decision['supporting_evidence_ids'])
            details['excluded_evidence']=[i for i in details['evidence'] if i['id'] not in used]
            details['regions']=[dict(g,selected=bool(set(g['evidence_ids'])&used),
                selection_reason='Explicit label-to-value field evidence' if set(g['evidence_ids'])&used else 'Not supporting the requested field') for g in details['regions']]
        annotation_start=time.perf_counter()
        angle=details['orientation']['selected_ccw_degrees'];shown=None
        if angle is not None:
            corrected=image.copy()
            if details['source']['engine']!='EasyOCR':corrected.thumbnail((2000,2000))
            corrected=corrected.rotate(angle,expand=True)
            # Overlay only supporting rows; all excluded boxes remain in JSON.
            # The label explicitly distinguishes this from the complete raw trace.
            shown=annotate(corrected,[dict(i,box=i['corrected_box']) for i in details['selected_evidence']])
        annotation_seconds=time.perf_counter()-annotation_start
        primary_orientation=details.get('alternate_engine_evidence',details).get('orientation',{})
        stage_profile=dict(image_preparation_seconds=preparation_seconds,image_identity_seconds=identity_seconds,
            primary_worker_wall_seconds=primary_seconds,fallback_worker_wall_seconds=fallback_seconds,
            worker_startup_seconds=startup_seconds,annotation_seconds=annotation_seconds,
            orientation_seconds=0. if reused else primary_orientation.get('seconds',0.),
            rapid_orientation_recognition_checks=0 if reused else len(primary_orientation.get('recognition_validation',{})),
            fallback_worker_requests=int(not reused and bool(fallback)),
            raw_model_stage_seconds=details.get('stages'),cache_reused=bool(reused),
            note='Nested measurements overlap. Primary/fallback wall times include startup, transport and preprocessing; cached original model stages are not current work.')
        details['orchestration_timings']=stage_profile
        status='Response ready. Recognized text may be incomplete; raw evidence is available in Technical Details.'
        # Workspace speech adapter speaks once after publishing text. Never
        # synthesize here, even when callers pass speak=True.
        return (details['final_answer'],None,status,details,shown,evidence_text(details),
            dict(callback_seconds=time.perf_counter()-start,persistent_worker=True,experimental_reading_active=True,
                 full_text=True,worker_memory=response['memory'],cache_hit=details['cache_hit'],conversation_evidence_cache_hit=reused,
                 ocr_processing_seconds=primary_seconds+fallback_seconds,answer_composition_seconds=details.get('composition_seconds',0.),
                 stage_profile=stage_profile))

    def close(self):
        self.legacy.close();self.reader.close();self.easy.close()
        if hasattr(self,'scene'):self.scene.close()


class EasyTextClient(FullTextClient):
    environment='.venv-gpu'
    worker='easy_reading_worker.py'


class SceneClient(FullTextClient):
    environment='.venv-gpu'
    worker='experimental_scene_worker.py'
