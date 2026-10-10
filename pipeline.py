import logging
from dataclasses import replace
from config import Config
from modules import vqa, speech_to_text, text_to_speech, object_detection, ocr, visual_features, qges, fusion, answerability, reliability
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

    def run(self, image, question, use_objects=False, use_ocr=False, use_clip=False, speak=True, enhanced=False, vqa_mode=None, reliability_checks=None):
        debug = {'mode': 'BLIP baseline', 'grounded': False, 'answerability': answerability.unavailable().__dict__,
                 'components': {}, 'evidence': [], 'selected_evidence': [], 'warnings': []}
        try:
            image = normalize_image(image)
            question = (question or '').strip()
            if not question:
                raise ValueError('Please type a question or transcribe a recording.')
            if len(question) > 1000:
                raise ValueError('Please keep the question under 1000 characters.')
            request_config = replace(self.config, vqa_mode=vqa_mode or self.config.vqa_mode)
            checks_enabled = self.config.enable_reliability_checks if reliability_checks is None else reliability_checks
            safeguard = reliability.question_check(question) if checks_enabled else reliability.Check('disabled')
            source, label = vqa.model_source(request_config)
            debug['mode'] = label
            debug['components']['vqa'] = {'model': source, 'status': 'selected'}
            check_presence = checks_enabled and safeguard.response is None and bool(safeguard.targets)
            for name, enabled, component in [('objects', use_objects or check_presence, object_detection), ('ocr', use_ocr, ocr)]:
                debug['components'][name] = 'disabled'
                if enabled:
                    try:
                        debug['evidence'].extend(component.extract(image, self.config))
                        debug['components'][name] = 'active'
                    except Exception as error:
                        debug['components'][name] = f'unavailable: {error}'
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
            selected = qges.select(question, debug['evidence']) if self.config.enable_qges else debug['evidence']
            debug['selected_evidence'] = selected
            debug['visual_features'] = features
            inputs = fusion.FusionInput(question, '', selected, features)
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
                if result.answer.strip().lower() == 'unanswerable':
                    final = 'The VQA model returned unanswerable. Please try a clearer image or a different question.'
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
