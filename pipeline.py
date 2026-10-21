import logging
from dataclasses import replace
from config import Config
from modules import vqa, speech_to_text, text_to_speech, object_detection, ocr, visual_features, qges, fusion, answerability, reliability, evidence_store, evidence_decision
from modules.question_intent import classify
from modules.custom_models import load_adapter
from utils.image_utils import normalize_image, annotate
from utils.model_loader import resolve_device

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, config=None):
        self.config = config or Config()
        self.adapters = {}

    def adapter(self, name):
        if name not in self.adapters:
            folder = getattr(self.config, name + '_checkpoint')
            self.adapters[name] = load_adapter(folder, resolve_device(self.config.device))
        return self.adapters[name]

    def transcribe(self, audio):
        try:
            text = speech_to_text.transcribe(audio, self.config)
            return text, text, 'Transcribed. Correct the question before asking.'
        except Exception as error:
            logger.exception('Transcription failed')
            import gradio as gr
            return gr.skip(), gr.skip(), f'Transcription failed: {error}. You can type your question.'

    def run(self, image, question, use_objects=False, use_ocr=False, use_clip=False, speak=True, enhanced=False, vqa_mode=None, reliability_checks=None, evidence_aware=False):
        debug = {'mode': 'BLIP baseline', 'grounded': False, 'answerability': answerability.unavailable().__dict__,
                 'components': {}, 'evidence': [], 'selected_evidence': [], 'warnings': [],
                 'baseline_answer': None, 'final_answer': None, 'decision_reason': '', 'supporting_evidence_ids': [], 'extraction_status': {}}
        try:
            image = normalize_image(image, thumbnail=not (evidence_aware or (vqa_mode or self.config.vqa_mode) == 'v2'))
            question = (question or '').strip()
            if not question:
                raise ValueError('Please type a question or transcribe a recording.')
            if len(question) > 1000:
                raise ValueError('Please keep the question under 1000 characters.')
            request_config = replace(self.config, vqa_mode=vqa_mode or self.config.vqa_mode)
            if evidence_aware and request_config.vqa_mode != 'v2':
                raise ValueError('Phase 2 requires explicit Improved BLIP v2 selection; no fallback is permitted.')
            if evidence_aware and enhanced:
                raise ValueError('Phase 2 Stages 1–4 cannot be combined with custom learned adapters.')
            checks_enabled = self.config.enable_reliability_checks if reliability_checks is None else reliability_checks
            if evidence_aware:
                checks_enabled = False  # No pre-generation legacy gate in the Phase 2 comparison.
                debug['warnings'].append('Phase 2 OCR decisions active; legacy pre-answer safeguards bypassed. Learned answerability is not implemented.')
            safeguard = reliability.question_check(question) if checks_enabled else reliability.Check('disabled')
            source, label = vqa.model_source(request_config)
            debug['mode'] = label
            debug['components']['vqa'] = {'model': source, 'status': 'selected'}
            check_presence = checks_enabled and safeguard.response is None and bool(safeguard.targets)
            intent = classify(question)
            needs_objects = evidence_aware and intent.supported and bool(reliability.mentioned_objects(question))
            for name, enabled, component in [('objects', use_objects or check_presence or needs_objects, object_detection), ('ocr', use_ocr or (evidence_aware and intent.supported), ocr)]:
                debug['components'][name] = 'disabled'
                debug['extraction_status'][name] = {'status': 'disabled'}
                if enabled:
                    try:
                        if evidence_aware:
                            items, extraction = evidence_store.extract(image, 'object' if name == 'objects' else 'ocr', component, self.config)
                            debug['evidence'].extend(items)
                            debug['extraction_status'][name] = {key: value for key, value in extraction.items() if key != 'items'}
                            debug['components'][name] = 'active' if extraction['status'] in ('complete', 'empty') else extraction['status']
                        else:
                            debug['evidence'].extend(component.extract(image, self.config))
                            debug['components'][name] = 'active'
                            debug['extraction_status'][name] = {'status': 'complete'}
                    except Exception as error:
                        debug['components'][name] = f'unavailable: {error}'
                        debug['extraction_status'][name] = {'status': 'unavailable', 'error': str(error)}
            if check_presence:
                safeguard = reliability.check_objects(safeguard.targets, debug['evidence'],
                    debug['components']['objects'] == 'active', self.config.object_presence_threshold)
            debug['reliability_checks'] = safeguard.__dict__
            features = {}
            debug['components']['clip'] = 'disabled'
            if use_clip:
                try:
                    features = visual_features.extract(image, question, debug['evidence'], self.config)
                    debug['components']['clip'] = 'active'
                except Exception as error:
                    debug['components']['clip'] = f'unavailable: {error}'
            selected = qges.select(question, debug['evidence'], top_k=None if evidence_aware else 5) if self.config.enable_qges or evidence_aware else debug['evidence']
            debug['selected_evidence'] = selected
            debug['visual_features'] = features
            inputs = fusion.FusionInput(question, '', selected, features)
            inputs.extraction_status = debug['extraction_status']
            inputs.ocr_min_score = self.config.ocr_min_score
            gate = answerability.unavailable()
            if enhanced and self.config.enable_trained:
                try:
                    inputs.evidence = self.adapter('qges').predict(inputs)
                except Exception as error:
                    debug['warnings'].append(f'Learned QGES unavailable: {error}')
                try:
                    gate = self.adapter('answerability').predict(inputs)
                except Exception as error:
                    debug['warnings'].append(f'Answerability unavailable: {error}')
            debug['answerability'] = gate.__dict__
            if safeguard.response:
                final = safeguard.response
            elif gate.answerable is False:
                final = 'I cannot answer this image reliably. Please take a clearer photo or ask another question.'
            else:
                inputs.baseline_answer = vqa.answer(image, question, request_config)
                debug['components']['vqa']['status'] = 'active'
                result = fusion.baseline(inputs)
                if evidence_aware:
                    result = evidence_decision.decide(inputs)
                    debug['mode'] = 'BLIP v2 + restricted OCR evidence decision'
                    debug['grounded'] = result.grounded
                if enhanced and self.config.enable_trained:
                    try:
                        result = self.adapter('fusion').predict(inputs)
                        debug['mode'] = 'Custom fusion adapter'
                        debug['grounded'] = result.grounded
                    except Exception as error:
                        debug['warnings'].append(f'Fusion unavailable; using BLIP: {error}')
                elif enhanced:
                    debug['warnings'].append('Trained components disabled; using BLIP baseline.')
                final = result.answer or 'I could not generate an answer. Try a clearer image or a different question.'
                debug.update(baseline_answer=inputs.baseline_answer, final_answer=result.answer,
                             decision_reason=result.reason, supporting_evidence_ids=result.supporting_evidence_ids,
                             decision=result.decision, answer_source=result.source,
                             candidate_trace=result.candidate_trace)
                if result.answer.strip().lower() == 'unanswerable':
                    final = 'The VQA model returned unanswerable. Please try a clearer image or a different question.'
            if debug['final_answer'] is None:
                debug['final_answer'] = final
                debug['decision_reason'] = safeguard.reason or gate.reason
            audio = None
            if speak:
                try:
                    audio = text_to_speech.synthesize(final, self.config)
                except Exception as error:
                    debug['warnings'].append(f'Speech output unavailable: {error}')
            status = 'Complete.' + (' See warnings in the evidence panel.' if debug['warnings'] else '')
            return final, audio, status, debug, annotate(image, debug['evidence'])
        except Exception as error:
            logger.exception('Question processing failed')
            return '', None, f'Unable to process: {error}', debug, None
