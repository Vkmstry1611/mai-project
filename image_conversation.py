"""Experimental browser-state conversations; assistant prose is never evidence."""
import copy
import json
from html import escape
from pathlib import Path
import re
import time
import uuid

MAX_TURNS=6
TTL=1800
ARTIFACT_ROOT=Path(__file__).resolve().parent

def visual_fingerprint(base):
    """Invalidate on policy, protected-source or weight-file stat changes.

    A miss still goes through the original worker's full integrity checks.
    No stat-only check is presented as a substitute for that verification.
    """
    root=ARTIFACT_ROOT;paths={base.run_dir/'selected-policy.json'}
    locks=[base.run_dir/'POLICY_FREEZE.json',base.run_dir/'APPLICATION_FREEZE.json',
        root/'reports/phase3_answerability/run-local-20261012-v1/models/metadata.json',
        root/'reports/phase3_answerability/plan-20261012-023211-3dbd1268/PHASE2_FREEZE.json']
    for lock in locks:
        paths.add(lock)
        if lock.is_file():
            payload=json.loads(lock.read_text(encoding='utf-8'))
            for key in ('source_sha256','frozen_source_sha256'):paths.update(root/n for n in payload.get(key,{}))
    paths.update((root/'models/blip_vizwiz_v2').glob('*'))
    paths.add(root/'reports/phase3_answerability/run-local-20261012-v1/models/heads.joblib')
    return [(str(p),p.stat().st_size,p.stat().st_mtime_ns) if p.is_file() else (str(p),None,None) for p in sorted(paths)]

def prepare(previous,image):
    from modules.evidence_store import image_identity
    from PIL import ImageOps
    identity=image_identity(ImageOps.exif_transpose(image).convert('RGB')) if image is not None else None
    prior=copy.deepcopy(previous or {})
    context=prior.get('conversation',{})
    if context.get('image_id')!=identity or time.time()-context.get('updated',0)>TTL:
        context={'session_id':uuid.uuid4().hex,'image_id':identity,'turns':[],'ocr':{},'visual':{},'read_ids':[]}
    context['updated']=time.time();prior['conversation']=context
    return prior

def hierarchy_omit(context,kind):
    nodes=[n for n in context.get('text_hierarchy',{}).get('nodes',[]) if n['kind']==kind]
    active=set(context.get('active_hierarchy_ids',[]))
    if kind=='step' and context.get('step_cursor',{}).get('id'):active={context['step_cursor']['id']}
    positions=[i for i,n in enumerate(nodes) if n['id'] in active]
    return sorted({v for n in nodes[:max(positions)+1] for v in n['accepted_ids']}) if positions else context.get('read_ids',[])


def resolve(question,context):
    q=' '.join(str(question).lower().strip(' .?!').split());turns=context.get('turns',[])
    last=turns[-1] if turns else {}
    from text_hierarchy import request
    from text_hierarchy import navigation
    nav=navigation(question,context)
    if nav:return nav.get('question',question),nav.get('message'),None
    if request(question):return question,None,(hierarchy_omit(context,request(question)['kind']) if request(question).get('rest') else None)
    if q in ('read the rest','read rest','continue','continue reading') and context.get('active_hierarchy_ids'):
        kind=context.get('active_hierarchy_kind','step')
        if kind not in ('step','paragraph','label'):return question,'Do you mean the remaining instructions or the remaining paragraphs?',None
        canonical={'step':'Read the remaining instructions','paragraph':'Read remaining paragraphs','label':'Read remaining labels'}[kind]
        return canonical,None,hierarchy_omit(context,kind)
    from reading_sequences import passage_reference
    selection=passage_reference(question,context)
    if selection:
        return selection.get('question',question),selection.get('message'),selection.get('omit_ids')
    if q in ('can you read it','read it'):
        return 'Read the visible text in this image.',None,None
    if q in ('what about the label','what about its label'):
        return 'Read the visible text on the label in this image.',None,None
    if q in ('what website is that','which website is that'):
        return 'What website is shown in this image?',None,None
    if q in ('explain that','explain it'):
        supported=set(last.get('support_ids',[]))
        source=[r for bundle in context.get('ocr',{}).values() for r in bundle['result'].get('evidence',[]) if r['id'] in supported]
        if last.get('scope') in ('main','summary','text_question','purpose') and len({r['id'] for r in source if len(r['text'].split())>=3})>=2:
            return 'Summarize this passage.',None,None
        return question,'Please specify what you want explained. I can read visible text or give an extractive overview of a passage.',None
    if q in ('read the rest','read rest'):
        scope=last.get('scope')
        if scope not in ('main','summary','text_question','all') or not last.get('support_ids'):
            return question,'Do you mean the rest of the passage or every visible word in the image?',None
        return ('Read every visible word.' if scope=='all' else 'Read this passage.'),None,context.get('read_ids',[])
    return question,None,None

def task_quality(image,task,evidence=None):
    from assistive_tools import quality
    from reading_relevance import local_quality
    result=quality(image);reading=task in ('reading','targeted')
    suggestions=[]
    for item in result['warnings']:
        if not reading and ('text' in item.lower() or 'focus' in item.lower()):
            item='For object recognition, hold the camera steady and show the object clearly. Unreadable print alone does not rule out visual recognition.'
        suggestions.append(item)
    result=dict(result,task=task,warnings=list(dict.fromkeys(suggestions)),blocks_inference=False,
        note='Task-specific heuristics are advisory and cannot establish model correctness.')
    if reading:
        result['local_text']=local_quality(evidence or [],image.size)
        if evidence is not None and not evidence:
            result['warnings'].append('Text recognition returned no usable regions. For reading, move closer and focus on the print; this does not establish that the whole scene is unrecognizable.')
        elif evidence and not any(r.get('confidence',0)>=(.9 if r.get('source',{}).get('engine')=='EasyOCR' else .75) for r in evidence):
            result['warnings'].append('Detected text fragments are not reliable enough to read. Try a steady close-up of the print. Recognition failure alone does not establish blur.')
    return result

def history_html(state):
    turns=(state or {}).get('conversation',{}).get('turns',[])
    if not turns:return '<p>No earlier questions for this image. Context stays in this browser session.</p>'
    def body(value):return escape(str(value)).replace('\n','<br>')
    return '<ol aria-label="Recent conversation about this image" style="padding-left:1.4em">'+''.join(
        '<li tabindex="0" style="display:list-item;margin-bottom:1.5em;padding:12px 0;border-bottom:1px solid #38536e">'
        '<div style="display:block;margin-bottom:14px"><strong style="display:block">User:</strong><br>'+body(t['question'])+
        '</div><div style="display:block"><strong style="display:block">VisionBridge:</strong><br>'+body(t['answer'])+'</div></li>' for t in turns)+'</ol>'

class ConversationSession:
    """Per-request facade with state supplied by Gradio, never process-global history."""
    def __init__(self,base,state,image):
        self.base=base;self.state=state;self.context=state['conversation'];self.image=image
        self.question='';self.original='';self.omit=None;self.route=None;self.routing_seconds=0.

    def classify(self,question):
        from workspace_intent import Route
        start=time.perf_counter();self.original=question
        from conversation_recall import recall
        self.recalled=recall(question,self.context)
        if self.recalled is not None:
            self.question=question;self.route=Route('history','Session-only recall; no OCR or BLIP.')
            self.routing_seconds=time.perf_counter()-start
            return self.route
        from reading_sequences import passage_reference
        self.selection=passage_reference(question,self.context)
        self.question,message,self.omit=resolve(question,self.context)
        if re.search(r"\b(?:make (?:it|this|the image|the label) clearer|retake|is (?:this|the) image blurry|why can(?:not|'t) you read|how (?:can|should) i.*(?:photo|picture|clear)|should i move closer|what should i do.*clear)\b",str(question),re.I):
            self.route=Route('guidance','Task-specific capture suggestions without inference.')
        else:self.route=Route('clarify','Ambiguous reference; do not derive facts from earlier prose.',message) if message else self.base.classify(self.question)
        self.routing_seconds=time.perf_counter()-start
        return self.route

    def guidance(self):
        last=[t for t in self.context.get('turns',[]) if t.get('task') not in ('guidance','history')];task='reading' if last and last[-1].get('task') in ('reading','targeted') else 'vqa'
        metrics=task_quality(self.image,task)
        answer=('To read small text, move closer, face the print toward the camera and hold still in even light.' if task=='reading' else
                'For a clearer object photo, hold the camera steady, use even light and keep the object in view. Small text may need a separate close-up.')
        answer+=' A better-looking photo does not guarantee a correct answer.'
        findings=metrics.get('warnings',[])
        answer=('Capture checks suggest: '+' '.join(findings)+' ' if findings else 'These checks do not establish a specific image defect. ')+answer
        return answer,None,'Capture suggestions; no model inference.',{'final_answer':answer,'answer_source':'assistant_notice','decision':'experimental_capture_guidance','quality':metrics,'selected_evidence':[],'supporting_evidence_ids':[]},None,'',{'callback_seconds':0.,'model_inference_performed':False}

    def full_request(self,*args,**kwargs):return self.base.full_request(*args,**kwargs)

    def request_auto(self,image,question,speak,experimental,rotation='0'):
        from text_hierarchy import request
        spec=request(self.question)
        if self.question.lower().strip(' .?!') in ('read the rest','read rest','continue','continue reading') and self.context.get('active_hierarchy_ids'):
            self.question='Read the remaining instructions';spec=request(self.question)
        result=self.base.full_request(image,self.question,False,None,target_field=self.route.task=='targeted',evidence_cache=self.context['ocr'],omit_ids=self.omit,
            selected_region_ids=(self.selection or {}).get('region_ids') or (self.context.get('active_passage_ids') if spec and spec['kind']=='paragraph' else None))
        return self.record(result)

    def request(self,image,question,speak,experimental,rotation):
        from modules.evidence_store import digest
        if self.route and self.route.task=='history':
            answer=self.recalled
            return self.record((answer,None,'Session recall; no model inference.',{'final_answer':answer,'answer_source':'session_history',
                'decision':'experimental_session_recall','selected_evidence':[],'supporting_evidence_ids':[]},None,'Session recall.',{'callback_seconds':0.,'model_inference_performed':False}))
        if self.route and self.route.task=='guidance':return self.record(self.guidance())
        policy=self.base.run_dir/'selected-policy.json'
        fingerprint=[policy.read_text(encoding='utf-8') if policy.is_file() else str(self.base.run_dir),visual_fingerprint(self.base)]
        key=digest([self.context['image_id'],self.question,fingerprint])
        cache=self.context['visual'];hit=key in cache;start=time.perf_counter()
        if hit:
            value=list(copy.deepcopy(cache[key]));value[6]={'callback_seconds':time.perf_counter()-start,'conversation_visual_cache_hit':True,'cached_original_timings':value[6]};result=tuple(value)
        else:
            result=self.base.request(image,self.question,False,False,'0');cache[key]=copy.deepcopy(result)
            while len(cache)>2:cache.pop(next(iter(cache)))
        result=self.purpose_fallback(result,image)
        from experimental_scene import requested,describe
        if requested(self.question) and hasattr(self.base,'scene') and result[3].get('final_answer','').lower()!='unanswerable' and not result[3].get('phase3_error'):
            tick=time.perf_counter()
            try:
                if 'scene' not in self.context:self.context['scene']=self.base.scene.read(image,0)['result']
                supplement=describe(self.question,result[3],self.context['scene']['evidence'])
                if supplement:
                    values=list(result);original=copy.deepcopy(values[3]);values[0]=supplement['answer']
                    values[3]=dict(original,frozen_vqa_result=original,final_answer=values[0],answer_source='experimental_scene_candidates',
                        selected_evidence=supplement['evidence'],supporting_evidence_ids=supplement['supporting_evidence_ids'],scene_evidence=self.context['scene'])
                    values[6]=dict(values[6],scene_seconds=time.perf_counter()-tick);result=tuple(values)
            except Exception as error:
                from workspace_diagnostics import report_failure
                values=list(result);values[3]=dict(values[3],scene_error_reference=report_failure('experimental-scene',error));result=tuple(values)
        return self.record(result,visual=True,visual_hit=hit)

    def purpose_fallback(self,result,image):
        from ocr_purpose import purpose_question,describe
        if not purpose_question(self.question):return result
        tick=time.perf_counter();values=list(result);frozen=copy.deepcopy(values[3])
        if frozen.get('final_answer','').strip().lower()!='unanswerable' or frozen.get('phase3_error'):
            return result
        try:
            evidence=self.base.cached_reading_evidence(image,self.context['ocr'])
            decision=describe(evidence) if evidence else {'used':False,'reason':'no_current_image_configuration_cache'}
        except Exception as error:
            from workspace_diagnostics import report_failure
            reference=report_failure('ocr-purpose-fallback',error)
            decision={'used':False,'reason':'fallback_processing_error','diagnostic_reference':reference}
        decision=dict(decision,ocr_extraction_requested=False,cache_only=True)
        if decision['used']:
            d=dict(frozen,evidence=evidence['evidence'],regions=decision['regions'],source=evidence.get('source',{}),
                selected_evidence=decision['selected_evidence'],supporting_evidence_ids=decision['supporting_evidence_ids'],
                selected_region_ids=decision['selected_region_ids'],passage_candidates=decision['passage_candidates'],
                excluded_evidence=[r for r in evidence['evidence'] if r['id'] not in decision['supporting_evidence_ids']],
                frozen_vqa_result=frozen,frozen_display_answer=values[0],frozen_final_answer=frozen['final_answer'],
                final_answer=decision['answer'],answer_source='ocr_content_description',reading_intent='purpose',partial=True,
                abstention_reason=None,content_description_not_scored_by_frozen_gates=True,
                decision='experimental_cached_ocr_purpose',decision_reason=decision['reason'],description_kind=decision['description_kind'])
            values[0]=decision['answer'];values[3]=d
            values[5]='OCR-grounded printed-content description. Supporting IDs: '+', '.join(decision['supporting_evidence_ids'])
        else:values[3]=dict(frozen)
        values[3]['purpose_fallback']=decision
        values[6]=dict(values[6],purpose_fallback_seconds=time.perf_counter()-tick,
            purpose_fallback_ocr_seconds=0.,purpose_fallback_used=decision['used'])
        return tuple(values)

    def record(self,result,visual=False,visual_hit=False):
        values=list(result);d=copy.deepcopy(values[3]);timings=dict(values[6])
        quality_task='reading' if d.get('answer_source')=='ocr_content_description' else (d.get('quality',{}).get('task',self.route.task) if self.route.task=='guidance' else self.route.task)
        quality=task_quality(self.image,quality_task,d.get('evidence'))
        frozen_abstained=visual and d.get('final_answer','').lower()=='unanswerable'
        useful_answer=d.get('answer_source') not in ('abstention','assistant_notice') and not frozen_abstained
        needs_capture_help=not useful_answer or (not visual and d.get('partial') and d.get('reading_intent')!='interface' and d.get('decision')!='experimental_labeled_field')
        quality['capture_suggestion_needed']=bool(needs_capture_help)
        if not needs_capture_help:
            quality['advisory_findings']=quality['warnings'];quality['warnings']=[]
            if 'local_text' in quality:quality['local_text']=dict(quality['local_text'],advisory_findings=quality['local_text']['warnings'],warnings=[])
            if 'local_quality' in d:d['local_quality']=dict(d['local_quality'],advisory_findings=d['local_quality']['warnings'],warnings=[])
        if frozen_abstained:
            d['frozen_display_answer']=values[0];d['frozen_final_answer']=d['final_answer']
            values[0]='I cannot reliably identify the object or scene from this photo. Try a steady, well-lit photo with the subject clearly in view. To read printed words, ask me to read the label.'
            d['experimental_display_explanation']=values[0]
        trace={'original_question':self.original,'resolved_question':self.question,'route':self.route.task,'routing_seconds':self.routing_seconds,
            'ocr_engine':d.get('source',{}).get('engine'),'ocr_cache_hit':timings.get('conversation_evidence_cache_hit',d.get('cache_hit',False)),
            'raw_evidence_count':len(d.get('evidence',[])),'region_count':len(d.get('regions',[])),
            'selected_count':len(d.get('selected_evidence',[])),'excluded_count':len(d.get('excluded_evidence',[])),
            'supporting_evidence_ids':d.get('supporting_evidence_ids',[]),'abstention_reason':d.get('abstention_reason') or (d.get('phase3_gate_reason') if frozen_abstained else None),
            'blip_status':('cached_frozen_result' if visual_hit else 'frozen_result_returned') if visual else 'not_requested',
            'blip_inference_observed':False if not visual or visual_hit else 'blip_seconds_including_loading' in timings.get('worker_stages',{}),
            'quality':quality,'ocr_processing_seconds':timings.get('ocr_processing_seconds',0.),'composition_seconds':d.get('composition_seconds',0.),
            'relevance_selection_seconds':d.get('relevance_selection_seconds',0.),
            'answer_composition_seconds':d.get('answer_composition_only_seconds',d.get('composition_seconds',0.)),
            'visual_cache_hit':visual_hit,'frozen_result_preserved':True,
            'answer_source':d.get('answer_source'),'description_kind':d.get('description_kind'),
            'purpose_fallback_used':d.get('purpose_fallback',{}).get('used',False)}
        d['assistant_trace']=trace;d['quality']=quality;values[3]=d;values[6]=dict(timings,routing_seconds=self.routing_seconds)
        if 'text_hierarchy' in d:self.context['text_hierarchy']=copy.deepcopy(d['text_hierarchy'])
        if 'selected_hierarchy_ids' in d:
            self.context['active_hierarchy_ids']=list(d['selected_hierarchy_ids'])
            self.context['active_hierarchy_kind']=d['hierarchy_request']['kind']
            steps=[n for n in d['text_hierarchy']['nodes'] if n['kind']=='step']
            selected=[n for n in steps if n['id'] in d['selected_hierarchy_ids']]
            if selected:
                last=selected[-1];sequence=[n for n in steps if n['sequence_id']==last['sequence_id']]
                self.context['step_cursor']={'id':last['id'],'index':next(i for i,n in enumerate(sequence) if n['id']==last['id']), 'sequence_id':last['sequence_id']}
        elif d.get('selected_region_ids'):self.context['active_hierarchy_ids']=[]
        if 'passage_candidates' in d:
            self.context['passage_candidates']=copy.deepcopy(d['passage_candidates'])
            self.context['clarification_pending']=bool(d.get('clarification_pending'))
            if d.get('selected_region_ids'):
                self.context['active_passage_ids']=list(d['selected_region_ids'])
        elif d.get('reading_intent')=='interface':
            self.context['active_passage_ids']=[]
        self.context['turn_number']=self.context.get('turn_number',0)+1
        if self.route.task!='history':self.context['question_number']=self.context.get('question_number',0)+1
        turn={'id':self.context['session_id']+'-turn-'+str(self.context['turn_number']),
              'question_number':self.context.get('question_number'),
              'step_ids':list(d.get('selected_hierarchy_ids',[])) if d.get('hierarchy_request',{}).get('kind')=='step' else [],
              'question':str(self.original),'resolved_question':str(self.question),'task':self.route.task,
              'scope':d.get('reading_intent'),'answer':str(values[0]),'support_ids':list(d.get('supporting_evidence_ids',[]))}
        self.context['turns']=(self.context.get('turns',[])+[turn])[-MAX_TURNS:]
        if self.route.task!='history':
            ledger={'id':turn['id'],'question_number':turn['question_number'],'question':turn['question']}
            self.context['question_history']=(self.context.get('question_history',[])+[ledger])[-60:]
        if turn['step_ids']:self.context['last_step']=copy.deepcopy(turn)
        available={r['id'] for bundle in self.context['ocr'].values() for r in bundle['result'].get('evidence',[])}
        self.context['read_ids']=sorted((set(self.context.get('read_ids',[]))|set(turn['support_ids']))&available)
        self.context['updated']=time.time()
        return tuple(values)
