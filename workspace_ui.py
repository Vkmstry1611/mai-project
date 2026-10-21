"""Presentation-only VisionBridge workspace. No inference policy lives here."""
from html import escape
import math

VISUAL='Visual Q&A'
READING='Experimental OCR Reading'
ASSISTANT='VisionBridge Assistant'
MODES=[ASSISTANT,VISUAL,READING]

HTML_CSS='''
:host{font-family:"Segoe UI",Arial,sans-serif;color:#e4edf7}
*{box-sizing:border-box} h1,h2,h3,p{margin:0} a{color:inherit}
.eyebrow{font-size:11px;font-weight:650;letter-spacing:.15em;color:#83a5bd;text-transform:uppercase}
.brand{display:flex;align-items:center;gap:10px;padding:2px 0 30px}.brand svg{flex-shrink:0}
.brand strong{display:block;font-size:17px;letter-spacing:-.5px}.brand small{display:block;color:#8fa5bc;font-size:10px;letter-spacing:.12em;margin-top:4px}
.nav{display:grid;gap:8px}.nav a{display:flex;align-items:center;gap:12px;padding:12px 11px;border-radius:9px;text-decoration:none;font-size:13px;color:#9fb3c9;border:1px solid transparent}
.nav a:hover,.nav a:focus-visible{background:#15283c;color:#d8f3f6;border-color:#34536c;outline:2px solid #7ed2df;outline-offset:2px}.nav svg{width:18px;height:18px}
.rail-note{margin-top:100px;font-size:12px;line-height:1.7;color:#829bb4;border-top:1px solid #24354a;padding-top:20px}
.hero{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;padding:0 0 18px}
.hero h1{font-size:32px;line-height:1.2;letter-spacing:-1.15px;font-weight:650;margin:10px 0;color:#edf4fb}.hero p:not(.eyebrow){font-size:14px;color:#a2b5ca;line-height:1.6}
.project-tag{border:1px solid #2b4056;border-radius:7px;padding:7px 10px;font-size:10px;color:#93adbf;white-space:nowrap;margin-top:3px;letter-spacing:.06em}
.section{display:flex;align-items:center;gap:12px;margin-bottom:7px}.number{font-size:11px;color:#74c6d0;border:1px solid #315165;border-radius:6px;padding:5px 7px;font-variant-numeric:tabular-nums}.section h2{font-size:16px;font-weight:600;letter-spacing:-.3px}.section p{font-size:12px;color:#91a8be;margin-top:4px;line-height:1.5}
.status{display:flex;align-items:flex-start;gap:10px;background:#0d192a;border:1px solid #2b4057;border-radius:9px;padding:12px 14px;margin:2px 0}.dot{width:7px;height:7px;background:#6c94b1;border-radius:50%;margin-top:5px;flex-shrink:0}.status strong{font-size:12px;font-weight:600}.status p{color:#9bb0c5;font-size:12px;line-height:1.6;margin-top:3px}.status.done .dot{background:#8bd0c6}.status.working .dot{background:#83bbdf}.status.error{border-color:#8b6561}.status.error .dot{background:#d5a69c}
.response-empty{padding:20px 0 9px}.response-empty svg{color:#6e9eb8;margin-bottom:15px}.response-empty h3{font-size:18px;letter-spacing:-.4px;margin-bottom:8px}.response-empty p{font-size:13px;color:#93aac1;line-height:1.65;max-width:330px}
.meta{display:flex;gap:7px;flex-wrap:wrap;padding:7px 0}.chip{font-size:11px;line-height:1.6;background:#192d42;border:1px solid #314d65;padding:4px 9px;border-radius:6px;color:#b6dce4}
.evidence-list{display:grid;gap:8px}.evidence-item{padding:13px 15px;border:1px solid #2a4058;border-radius:9px;background:#0d192a}.evidence-top{display:flex;justify-content:space-between;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#80bdca;gap:9px}.evidence-item p{font-size:14px;line-height:1.6;margin-top:8px;overflow-wrap:anywhere;white-space:pre-wrap}.evidence-item small{display:block;color:#92a8bd;font-size:11px;line-height:1.6;margin-top:6px}.empty-evidence{font-size:13px;line-height:1.7;color:#91a8be;padding:20px;border:1px dashed #2e455d;border-radius:9px}.guide{border-top:1px solid #263b51;padding-top:22px;margin-top:10px}.guide h2{font-size:15px;margin-bottom:10px}.guide p{font-size:12px;color:#96acc2;line-height:1.8;max-width:880px}.guide strong{color:#c1d2e4}
@media(max-width:760px){.brand{padding-bottom:12px}.nav{display:flex;flex-wrap:wrap;gap:2px}.nav a{font-size:12px;padding:9px}.rail-note{display:none}.hero h1{font-size:26px}.project-tag{display:none}}
'''

LOGO='''<svg width="36" height="36" viewBox="0 0 36 36" fill="none" aria-hidden="true"><rect x="1" y="1" width="34" height="34" rx="10" fill="#142d42" stroke="#3d677e"/><path d="M10 25V17a8 8 0 0 1 16 0v8M10 20h16M18 12v15" stroke="#91d8df" stroke-width="2.1" stroke-linecap="round"/><circle cx="18" cy="10" r="2.3" fill="#91d8df"/></svg>'''
BRAND=f'<div class="brand">{LOGO}<div><strong>VisionBridge</strong><small>VISUAL ASSISTANT</small></div></div>'
NAV='''<nav class="nav" aria-label="Workspace navigation"><a href="#vb-workspace"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="3" width="18" height="18" rx="4"/><path d="M9 3v18M9 10h12"/></svg>Workspace</a><a href="#vb-evidence"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 8V4h4m10 0h4v4M3 16v4h4m10 0h4v-4"/><circle cx="12" cy="12" r="4"/></svg>Evidence</a><a href="#vb-guide"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M5 4h10a4 4 0 0 1 4 4v12H9a4 4 0 0 1-4-4Z"/><path d="M9 4v16m3-11h4m-4 4h4"/></svg>Quick guide</a></nav>'''
HERO='''<header class="hero"><div><p class="eyebrow">Accessible visual assistance · on your laptop</p><h1>VISIONBRIDGE</h1><p>Your AI Assistant for Understanding Visual Information</p><p>Upload a photo and ask in your own words.</p></div><span class="project-tag">UNIVERSITY PROJECT · EXPERIMENTAL ASSISTANT</span></header>'''
EMPTY_RESPONSE=f'<div class="response-empty">{LOGO}<h3>Room for a clearer answer.</h3><p>Upload an image and ask a question. Your answer and optional spoken response will appear here.</p></div>'
EMPTY_EVIDENCE='<div class="empty-evidence">No evidence yet. After analysis, this space will show the actual extracted candidates and which ones supported the answer.</div>'


def section(number,title,subtitle=''):
    return f'<div class="section"><span class="number">{escape(number)}</span><div><h2>{escape(title)}</h2><p>{escape(subtitle)}</p></div></div>'


def status_html(title,description='',kind='idle'):
    kind=kind if kind in ('idle','working','done','error') else 'idle'
    return f'<div class="status {kind}" role="status" aria-live="polite"><span class="dot" aria-hidden="true"></span><div><strong>{escape(title)}</strong><p>{escape(description)}</p></div></div>'


def evidence_html(details):
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
            note='Spoken answer ready. If autoplay is blocked, press Play Answer or the audio player Play control.'
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
            yield (answer,audio,status_html(title,status,'error' if failed else 'done'),details,annotated,text,timings,
                   response_meta(details,timings,show_task=not automatic),gr.update(interactive=True),evidence_html(details),explanation)
        except Exception as error:
            from workspace_diagnostics import report_failure
            reference=report_failure('analysis-worker',error)
            updated='Session artifact changed; restart and reverify:' in str(error)
            message=('The application was updated while it was running. Restart VisionBridge, refresh this page, and try again.' if updated else
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
