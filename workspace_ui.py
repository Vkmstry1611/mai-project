"""Presentation-only VisionBridge workspace. No inference policy lives here."""
from html import escape
import math

VISUAL='Visual Q&A'
READING='Experimental OCR Reading'
ASSISTANT='VisionBridge Assistant'
MODES=[ASSISTANT,VISUAL,READING]

HTML_CSS='''
:host{font-family:"Segoe UI",Arial,sans-serif;color:#232820}*{box-sizing:border-box}h1,h2,h3,p{margin:0}a{color:inherit}.eyebrow{font:600 10px/1.5 Consolas,monospace;letter-spacing:.17em;text-transform:uppercase;color:#73776c}
.brand{display:flex;align-items:center;gap:12px}.brand strong{font-size:22px;letter-spacing:-1px;font-weight:650}.brand small{display:block;font:10px Consolas,monospace;letter-spacing:.15em;margin-top:3px;color:#777b70}.brand svg{flex-shrink:0}.nav{display:flex;gap:28px;align-items:center;height:48px}.nav a{white-space:nowrap;text-decoration:none;font-size:13px;color:#62675c;transition:color .2s}.nav a:first-child{color:#254ddd}.nav a:hover{color:#254ddd}.nav a:focus-visible{outline:2px solid #254ddd;outline-offset:6px}.nav svg{display:none}
.hero{display:flex;justify-content:space-between;align-items:flex-end;gap:32px;padding:46px 0 36px}.hero h1{font:normal clamp(44px,5.5vw,76px)/1.03 Georgia,serif;letter-spacing:-3.5px;margin-top:18px}.hero h1 em{color:#254ddd;font-weight:normal}.hero .intro{max-width:290px;font-size:14px;line-height:1.8;color:#73776c;padding-bottom:4px}.hero .intro span{display:block;margin-top:15px;font:10px Consolas,monospace;letter-spacing:.08em;color:#48785b}.hero .intro span:before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:#48785b;margin-right:8px}
.section{display:flex;align-items:center;gap:12px;margin:0 0 10px}.number{font:11px Consolas,monospace;color:#254ddd;align-self:flex-start;padding-top:4px}.section h2{font-size:16px;font-weight:600;letter-spacing:-.3px}.section p{font-size:12px;color:#777b70;margin-top:4px;line-height:1.6}
.status{display:flex;gap:10px;padding:14px 0;border-top:1px solid #d9ddd0;margin:4px 0}.dot{width:6px;height:6px;background:#929888;border-radius:50%;margin-top:6px;flex-shrink:0}.status strong{font-size:12px;font-weight:600}.status p{color:#73776c;font-size:12px;line-height:1.65;margin-top:4px}.status.done .dot{background:#48785b}.status.working .dot{background:#254ddd;animation:breathe 1s ease-in-out infinite alternate}.status.error{border-color:#c67d67}.status.error .dot{background:#a53e25}.response-empty{padding:28px 0 6px}.response-empty .response-mark{font:normal 84px/.8 Georgia,serif;color:#c6ceba;display:block;margin-bottom:24px}.response-empty h3{font:normal 30px/1.15 Georgia,serif;letter-spacing:-.8px;max-width:270px}.response-empty p{font-size:13px;color:#73776c;line-height:1.8;max-width:285px;margin-top:14px}.meta{display:flex;gap:8px;flex-wrap:wrap;padding:6px 0}.chip{font:10px/1.6 Consolas,monospace;border-bottom:1px solid #b8c3a7;padding:4px 0;color:#586b45}.evidence-list{display:grid;gap:8px}.evidence-item{padding:16px;border-left:2px solid #a8b999;background:#f6f7f1}.evidence-top{display:flex;justify-content:space-between;font:10px Consolas,monospace;text-transform:uppercase;color:#586b45;gap:10px}.evidence-item p{font-size:14px;line-height:1.7;margin-top:10px;overflow-wrap:anywhere;white-space:pre-wrap}.evidence-item small{display:block;font-size:11px;color:#777b70;margin-top:8px}.empty-evidence{font-size:13px;line-height:1.8;color:#73776c;padding:22px;border:1px dashed #c8cdbf}.guide{display:grid;grid-template-columns:1fr 2fr;gap:40px;padding:28px 0 16px;border-top:1px solid #d5d8cd;margin-top:18px}.guide h2{font:normal 27px/1.25 Georgia,serif;letter-spacing:-.6px;max-width:260px}.guide p{font-size:12px;color:#777b70;line-height:1.9;max-width:700px}#question-notice,p[role=status]{font-size:11px;line-height:1.6;color:#73776c}@keyframes breathe{to{opacity:.3}}@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}@media(max-width:760px){.hero{padding:30px 0;align-items:flex-start;flex-direction:column;gap:20px}.hero h1{letter-spacing:-2px}.hero .intro{max-width:100%}.hero .intro span{margin-top:8px}.nav{gap:24px;height:34px}.brand strong{font-size:20px}.guide{grid-template-columns:1fr;gap:18px}.response-empty{padding-top:14px}}
'''
LOGO='''<svg width="38" height="38" viewBox="0 0 38 38" fill="none" aria-hidden="true"><path d="M3 9V3h10M25 3h10v10M35 25v10H25M13 35H3V25" stroke="#254ddd" stroke-width="2"/><path d="m10 14 9 12 9-12M19 9v9" stroke="#254ddd" stroke-width="2"/></svg>'''
BRAND=f'<a class="brand" href="#vb-workspace" style="text-decoration:none">{LOGO}<div><h1>VisionBridge</h1><small>See more. Understand more.</small></div></a>'
NAV='''<nav class="nav" aria-label="Workspace navigation"><a href="#vb-workspace">Workspace ↗</a><a href="#vb-evidence">Evidence</a><a href="#vb-guide">How it works</a></nav>'''
HERO='''<header class="hero"><div><p class="eyebrow">VISIONBRIDGE · YOUR VISUAL ASSISTANT</p><h1>A clearer <em>view.</em></h1></div><p class="intro">Upload or take a photo.<br>Ask by voice or text.<br>Read or listen to the answer.<span>PROCESSED ON YOUR COMPUTER</span></p></header>'''
EMPTY_RESPONSE='<div class="response-empty" aria-hidden="true"></div>'
EMPTY_EVIDENCE='<div class="empty-evidence">Your analysis has a paper trail. Extracted candidates and supporting regions will appear here when available.</div>'


HTML_CSS+='''
.nav{height:auto;min-height:44px;flex-wrap:wrap;gap:20px}.nav a{font-size:16px;color:#424c3e}.hero{padding:28px 0;align-items:center}.hero h1{font-size:clamp(38px,5vw,60px);letter-spacing:-2px}.hero .intro{font-size:17px;color:#424c3e;line-height:1.7}.eyebrow{font-size:12px;color:#505c49}.section h2{font-size:20px}.section p{font-size:16px;color:#424c3e}.number{font-size:13px}.status strong{font-size:17px}.status p{font-size:16px;color:#424c3e}.response-empty p,.empty-evidence{font-size:16px;color:#424c3e}.guide p{font-size:16px;color:#424c3e}.chip{font-size:13px;color:#3c5031}.skip-links a{position:absolute;top:-9999px;left:16px;background:#232820;color:white;padding:16px;z-index:9999;text-decoration:underline;font-size:18px}.skip-links a:focus{top:12px;outline:3px solid #254ddd;outline-offset:3px}.skip-links a:nth-child(2){left:220px}@media(max-width:760px){.hero{gap:16px;align-items:flex-start}.hero h1{margin-top:8px}.hero .intro{max-width:none}.guide{gap:16px}.nav{gap:16px}.skip-links a:nth-child(2){left:16px}.skip-links a:nth-child(2):focus{top:78px}}
'''
HTML_CSS+='''
.brand h1{font:750 32px/1.12 "Segoe UI",Arial,sans-serif;letter-spacing:-.045em;color:#101828;margin:0}.brand small{font:500 18px/1.45 "Segoe UI",Arial,sans-serif;letter-spacing:0;color:#344054;margin-top:3px}.brand svg{width:42px;height:42px}.status{margin:0;padding:12px 0;border:0}.status strong{font-size:18px;color:#101828}.status p{font-size:18px;color:#344054}.status.error{background:#fff3f0;border-left:4px solid #ad2f17;padding:12px}.status.working{background:#edf4ff;border-left:4px solid #1644cc;padding:12px}.response-empty{display:none}.chip{font:500 15px/1.4 "Segoe UI",Arial,sans-serif;color:#344054;border:0;background:#edf4ff;border-radius:99px;padding:5px 10px}.empty-evidence,.evidence-item p{font-size:18px;line-height:1.6;color:#101828}.evidence-item small,.evidence-top{font-size:15px;color:#344054}.skip-links a{background:#101828;color:#fff}
'''


def section(number,title,subtitle=''):
    return f'<div class="section"><span class="number">{escape(number)}</span><div><h2>{escape(title)}</h2><p>{escape(subtitle)}</p></div></div>'


def status_html(title,description='',kind='idle'):
    kind=kind if kind in ('idle','working','done','error') else 'idle'
    return f'<div class="status {kind}" role="status" aria-live="polite"><span class="dot" aria-hidden="true"></span><div><strong>{escape(title)}</strong><p>{escape(description)}</p></div></div>'


def evidence_html(details):
    if details.get('answer_source')=='local_vision_model':
        return '<div class="empty-evidence">The local vision model interpreted the supplied image directly. This response has no independently verified OCR boxes or correctness score. Check important details against the image.</div>'
    rows=details.get('selected_evidence',[]);support=set(details.get('supporting_evidence_ids',[]))
    if not rows:return '<div class="empty-evidence">No evidence candidates were selected for this request. A generated answer does not establish visual support.</div>'
    cards=[]
    for item in rows[:12]:
        used=item.get('id') in support
        source=item.get('source',{})
        source=(source.get('extractor') or source.get('engine','')) if isinstance(source,dict) else str(source)
        cards.append('<article class="evidence-item"><div class="evidence-top"><span>'+escape(str(item.get('kind','Evidence')))+
            '</span><span>'+('Used in answer' if used else 'Candidate only')+'</span></div><p>'+escape(str(item.get('text','')))+
            '</p><small>'+escape(source or 'Source not recorded')+'</small></article>')
    if len(rows)>12:cards.append('<div class="empty-evidence">Showing 12 of '+str(len(rows))+' candidates. The text alternative and advanced details contain the complete selection.</div>')
    excluded=details.get('excluded_evidence',[])
    if excluded:
        cards.append('<details><summary>Excluded candidates · '+str(len(excluded))+' (not used in this answer)</summary><p>'+escape('\n'.join(str(r.get('text','')) for r in excluded))+'</p><p>All confidence, region-selection reasons, boxes and source IDs remain in the decision JSON.</p></details>')
    return '<div class="evidence-list">'+''.join(cards)+'</div>'


def response_meta(details,timings,show_task=True):
    source=details.get('answer_source','Not recorded')
    mode=READING if timings.get('experimental_reading_active') else VISUAL
    tags=[mode,'Answer source: '+str(source)] if show_task else [ASSISTANT]
    if details.get('partial'):tags.append('Partial extraction')
    seconds=timings.get('callback_seconds')
    if isinstance(seconds,(float,int)) and math.isfinite(seconds):tags.append(f'Callback {seconds:.2f}s')
    return '<div class="meta">'+''.join('<span class="chip">'+escape(t)+'</span>' for t in tags)+'</div>'


def reading_mode(mode):
    if mode not in MODES:raise ValueError('Select a supported assistant mode.')
    return mode==READING


def routing_html(task,reason,automatic=True):
    return status_html('Task selection · '+('automatic' if automatic else 'developer override'),task+'. '+reason)


def speak_notice(message):
    """Optional speech for a capability/clarification notice, never a visual claim."""
    from dataclasses import replace
    from pathlib import Path
    from uuid import uuid4
    from config import Config
    from modules.text_to_speech import synthesize
    folder=Path(__file__).resolve().parent/'reports/workspace_ui/assistant-responses'/uuid4().hex
    return synthesize(message,replace(Config(),output_dir=folder))


class WorkspaceController:
    def __init__(self,session,transcriber=None,speaker=None):
        self.session=session
        self.speaker=speaker or speak_notice
        if transcriber is None:
            from workspace_voice import transcribe_local
            transcriber=transcribe_local
        self.transcriber=transcriber
        from workspace_speech import ResponseSpeech
        self.response_speech=ResponseSpeech()

    def analyze_ui(self,image,question,mute=False,mode=ASSISTANT,rotation='0',context=None):
        """Speech-first UI adapter. Frozen inference always receives speak=False.

        Publish text BEFORE synthesis, then attach one audio artifact. The legacy
        analyze contract stays available to original tests and API adapters.
        """
        final=None;active=self
        if context is not None and mode==ASSISTANT and hasattr(self.session,'for_conversation'):
            active=object.__new__(WorkspaceController)
            active.session=self.session.for_conversation(context,image);active.speaker=self.speaker
            active.transcriber=self.transcriber;active.response_speech=self.response_speech
        for value in active.analyze(image,question,False,mode,rotation):
            final=value
            yield value
        if final is None or not final[0] or mute:return
        import time
        tick=time.perf_counter()
        values=list(final)
        try:
            values[1]=self.response_speech(final[0])
            note='Spoken answer ready. Select Read Aloud to play it.'
        except Exception as error:
            from workspace_diagnostics import report_failure
            reference=report_failure('tts',error)
            note='Speech generation failed; your text answer is still available. Diagnostic reference: '+reference
        values[2]+=status_html('Spoken answer',note,'done' if values[1] else 'error')
        values[6]=dict(values[6],speech_seconds=time.perf_counter()-tick,speech_enabled=True)
        if 'assistant_trace' in values[3]:
            values[3]=dict(values[3],assistant_trace=dict(values[3]['assistant_trace'],tts_seconds=values[6]['speech_seconds'],speech_generated=bool(values[1])))
        yield tuple(values)

    def analyze(self,image,question,speak,mode,rotation):
        import gradio as gr
        from submission_ui_helpers import input_error
        from experimental_reading import is_reading_request
        error=input_error(image,question)
        from workspace_intent import classify,VERSION
        automatic=mode==ASSISTANT
        route=None
        manual_full=mode=='Full Text Reading (experimental)' and hasattr(self.session,'full_request')
        try:reading=True if manual_full else reading_mode(mode)
        except ValueError as problem:error=str(problem);reading=False
        if not error and len(question)>1000:error='Please keep your question under 1000 characters.'
        if not error and automatic:
            route=(self.session.classify(question) if hasattr(self.session,'classify') else classify(question));reading=route.task in ('reading','targeted')
        if not error and reading and not manual_full and not (hasattr(self.session,'full_request') and route and route.task in ('reading','targeted')) and not (route and route.task=='targeted') and not is_reading_request(question):
            error='Ask a literal reading question, such as “What words are written here?”, or select Visual Q&A.'
        task=route.task if route else ('reading' if reading else 'vqa')
        explanation=routing_html(task,route.reason if route else 'Use the explicitly selected existing pipeline.',automatic)
        if error:
            yield ('',None,status_html('Check your input',error,'error'),{},None,'',{},EMPTY_RESPONSE,gr.update(interactive=True),EMPTY_EVIDENCE,routing_html('Not run','Input needs correction.'))
            return
        if route and route.task in ('clarify','unsupported_field'):
            audio=None;note='No image inference was performed.'
            if speak:
                try:audio=self.speaker(route.message)
                except Exception as problem:note+=' Spoken response unavailable: '+str(problem)
            details={'final_answer':route.message,'answer_source':'assistant_notice','decision':'workspace_'+route.task,
                     'selected_evidence':[],'supporting_evidence_ids':[],'workspace_routing':{'task':route.task,'reason':route.reason,'version':VERSION}}
            notice=(route.message,audio,status_html('Please clarify' if task=='clarify' else 'Current limitation',note),details,None,
                   note,{'model_inference_performed':False},'<div class="meta"><span class="chip">VisionBridge Assistant</span></div>',gr.update(interactive=True),
                   '<div class="empty-evidence">No evidence was extracted for this clarification or capability notice.</div>',explanation)
            if hasattr(self.session,'record'):
                raw=self.session.record((notice[0],notice[1],note,notice[3],None,note,notice[6]))
                notice=(*notice[:3],raw[3],*notice[4:6],raw[6],*notice[7:])
            yield notice
            return
        yield ('',None,status_html('Analyzing your image','The first request may initialize models. Keep this tab open.','working'),{},None,'',{},
               '<div class="empty-evidence">Preparing your response from the supplied image and question.</div>',gr.update(interactive=False),EMPTY_EVIDENCE,explanation)
        try:
            # Only the separate opt-in orientation session implements this hook.
            # Existing sessions/manual modes retain the original call unchanged.
            if manual_full:
                answer,audio,status,details,annotated,text,timings=self.session.full_request(image,question,speak,rotation)
            else:
                request=(self.session.request_auto if automatic and reading and hasattr(self.session,'request_auto') else self.session.request)
                answer,audio,status,details,annotated,text,timings=request(image,question,speak,reading,rotation if reading and not automatic else '0')
            failed=not answer or bool(details.get('phase3_error'))
            title='Needs attention' if failed else ('More information needed' if details.get('answer_source')=='abstention' else 'Analysis complete')
            visible_status=status.replace(' See warnings in the evidence panel.',' Check important details against the image.')
            yield (answer,audio,status_html(title,visible_status,'error' if failed else 'done'),details,annotated,text,timings,
                   response_meta(details,timings,show_task=not automatic),gr.update(interactive=True),evidence_html(details),explanation)
        except Exception as error:
            from workspace_diagnostics import report_failure
            from modules.local_vision import LocalVisionError
            reference=report_failure('analysis-worker',error)
            updated='Session artifact changed; restart and reverify:' in str(error)
            message=(str(error) if isinstance(error,LocalVisionError) else
                     'The application was updated while it was running. Restart VisionBridge, refresh this page, and try again.' if updated else
                     'I could not complete this request. Please try again. If it continues, restart VisionBridge and share the diagnostic reference.')
            yield ('',None,status_html('Analysis could not finish',message+' No substitute answer was generated. Diagnostic reference: '+reference,'error'),{'application_error':{'stage':'analysis-worker','reference':reference,'code':'worker_source_changed' if updated else 'analysis_failed'}},None,'',{},EMPTY_RESPONSE,
                   gr.update(interactive=True),EMPTY_EVIDENCE,explanation)

    def transcribe(self,recording):
        import gradio as gr
        if not recording:
            yield gr.skip(),status_html('Record a question first','You can also type directly into the question box.','error');return
        yield gr.skip(),status_html('Transcribing your recording','Review and edit the text before analyzing.','working')
        try:text,status=self.transcriber(recording)
        except Exception as error:
            yield gr.skip(),status_html('Transcription unavailable',str(error)+' You can type your question instead.','error');return
        yield (text if text else gr.skip()),status_html('Review your question' if text else 'Transcription unavailable',status,'done' if text else 'error')
