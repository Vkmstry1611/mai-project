"""Presentation tests using the installed UI environment; no model inference."""
import copy
import inspect
import unittest
from types import SimpleNamespace

from PIL import Image
from visionbridge.workspace_ui import WorkspaceController,ASSISTANT,VISUAL,READING,evidence_html,status_html,response_meta


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        self.photo=Image.new('RGB',(120,60),'white')
        self.details={'baseline_answer':'hello','final_answer':'HELLO','answer_source':'ocr',
            'selected_evidence':[dict(id='ocr-1',text='HELLO',kind='ocr',source={'extractor':'easyocr'})],
            'supporting_evidence_ids':['ocr-1']}
        self.timings={'callback_seconds':1.25,'experimental_reading_active':False}
        def request(*args):
            self.calls.append(args)
            return ('HELLO','generated.wav','Complete',self.details,self.photo,'HELLO evidence',self.timings)
        self.controller=WorkspaceController(SimpleNamespace(request=request),lambda path:('Read this','Transcribed; review.'))

    def test_visual_route_preserves_input_outputs_and_ignores_rotation(self):
        before=copy.deepcopy(self.details)
        steps=list(self.controller.analyze(self.photo,'What does this say?',True,VISUAL,'180'))
        self.assertEqual(len(steps),2)
        self.assertIn('Analyzing your image',steps[0][2]);self.assertFalse(steps[0][8]['interactive'])
        final=steps[-1]
        self.assertEqual(self.calls,[(self.photo,'What does this say?',True,False,'0')])
        self.assertEqual(final[:2],('HELLO','generated.wav'))
        self.assertIs(final[3],self.details);self.assertIs(final[4],self.photo);self.assertIs(final[6],self.timings)
        self.assertEqual(self.details,before);self.assertTrue(final[8]['interactive'])
        self.assertIn('HELLO',final[9]);self.assertIn('Used in answer',final[9])

    def test_reading_route_preserves_question_rotation_and_speech_flag(self):
        final=list(self.controller.analyze(self.photo,'What words are written here?',False,READING,'180'))[-1]
        self.assertEqual(self.calls,[(self.photo,'What words are written here?',False,True,'180')])
        self.assertEqual(final[0],'HELLO')

    def test_invalid_inputs_never_start_worker(self):
        for args in [(None,'Read this',False,VISUAL,'0'),(self.photo,' ',False,VISUAL,'0'),
                     (self.photo,'What is this?',False,READING,'0'),(self.photo,'Read this',False,'unknown','0'),
                     (self.photo,'x'*1001,False,VISUAL,'0')]:
            with self.subTest(args=args[1:]):
                result=list(self.controller.analyze(*args))
                self.assertEqual(len(result),1);self.assertIn('Check your input',result[0][2])
        self.assertFalse(self.calls)

    def test_worker_failure_does_not_generate_replacement_answer(self):
        def fail(*args):raise RuntimeError('test worker failed')
        self.controller.session=SimpleNamespace(request=fail)
        final=list(self.controller.analyze(self.photo,'Read this',False,VISUAL,'0'))[-1]
        self.assertEqual(final[0],'');self.assertIsNone(final[1]);self.assertIn('No substitute answer',final[2]);self.assertTrue(final[8]['interactive'])

    def test_abstention_and_error_labels_follow_actual_details(self):
        self.details['answer_source']='abstention'
        final=list(self.controller.analyze(self.photo,'Read this',False,VISUAL,'0'))[-1]
        self.assertIn('More information needed',final[2])
        self.details['phase3_error']='failure'
        final=list(self.controller.analyze(self.photo,'Read this',False,VISUAL,'0'))[-1]
        self.assertIn('Needs attention',final[2])

    def test_transcription_review_and_missing_recording(self):
        missing=list(self.controller.transcribe(None));self.assertIn('Record a question first',missing[0][1])
        steps=list(self.controller.transcribe('recording.wav'))
        self.assertIn('Transcribing',steps[0][1]);self.assertEqual(steps[-1][0],'Read this')
        self.assertIn('Review your question',steps[-1][1])

    def test_failed_transcription_preserves_typed_question(self):
        for transcriber in [lambda p:('','No speech'),lambda p:(_ for _ in ()).throw(RuntimeError('decoder unavailable'))]:
            self.controller.transcriber=transcriber
            final=list(self.controller.transcribe('recording.wav'))[-1]
            self.assertEqual(final[0],{'__type__':'update'})
            self.assertIn('Transcription unavailable',final[1])

    def test_evidence_and_status_escape_untrusted_text(self):
        details={'selected_evidence':[dict(id='x',kind='ocr',text='<script>bad</script>',source={'extractor':'<img>'})]}
        self.assertNotIn('<script>',evidence_html(details));self.assertIn('&lt;script&gt;',evidence_html(details))
        self.assertIn('Candidate only',evidence_html(details))
        self.assertNotIn('<script>',status_html('<script>','<script>','bad'))

    def test_evidence_limit_is_explicit_and_missing_timing_not_fabricated(self):
        details={'selected_evidence':[dict(id=str(i),kind='ocr',text=str(i)) for i in range(14)]}
        self.assertIn('12 of 14',evidence_html(details))
        self.assertNotIn('Callback',response_meta({},{}))


class GradioContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from launch_workspace import build_app,launch_options
        cls.demo,cls.session=build_app('reports/phase4_policy/run-local-20261013-v1',SimpleNamespace(close=lambda:None))
        cls.options=launch_options()
        cls.config=cls.demo.config
        cls.components={c['id']:c for c in cls.config['components']}
        cls.by_elem={c['props'].get('elem_id'):c for c in cls.config['components'] if c['props'].get('elem_id')}

    @classmethod
    def tearDownClass(cls):cls.session.close()

    def test_theme_and_launch_parameters_are_supported(self):
        import gradio as gr
        self.assertNotIn('input_text_color',inspect.signature(gr.themes.Base.set).parameters)
        self.assertEqual(self.options['theme'].body_text_color,'#232820')
        self.assertTrue(set(self.options)<=set(inspect.signature(gr.Blocks.launch).parameters))

    def test_modes_default_and_rotation_handler(self):
        self.assertEqual(self.by_elem['vb-mode']['props']['value'],ASSISTANT)
        self.assertFalse(self.by_elem['vb-rotation']['props']['visible'])
        switch=next(f.fn for f in self.demo.fns.values() if f.api_name=='set_mode')
        self.assertTrue(switch(READING)[0]['visible']);self.assertFalse(switch(VISUAL)[0]['visible'])
        self.assertNotIn('value',switch(READING)[1])  # Keep the student's typed question.

    def test_modes_are_only_in_closed_developer_section(self):
        developer=self.by_elem['vb-developer']
        self.assertEqual(developer['type'],'accordion');self.assertFalse(developer['props']['open'])
        mode_id=self.by_elem['vb-mode']['id']
        found=[]
        def visit(node,ancestors):
            if node.get('id')==mode_id:found.extend(ancestors)
            for child in node.get('children',[]):visit(child,ancestors+[node.get('id')])
        visit(self.config['layout'],[])
        self.assertIn(developer['id'],found)

    def test_analyze_inputs_and_panel_outputs(self):
        event=next(d for d in self.config['dependencies'] if d['api_name']=='analyze')
        self.assertEqual(len(event['inputs']),6);self.assertEqual(len(event['outputs']),13)
        self.assertEqual(event['inputs'][0],self.by_elem['vb-photo']['id'])
        self.assertEqual(event['outputs'][0],self.by_elem['vb-answer']['id'])
        self.assertEqual(self.components[event['inputs'][2]]['props']['label'],'Mute Spoken Answers')
        self.assertFalse(self.components[event['inputs'][2]]['props']['value'])
        self.assertTrue(self.by_elem['vb-playback']['props']['autoplay'])
        self.assertTrue(event['types']['generator'])

    def test_upload_recording_and_accessibility_handlers_are_wired(self):
        photo_id=self.by_elem['vb-photo']['id'];larger_id=self.by_elem['vb-access']['id']
        self.assertTrue(any(tuple(t)==(photo_id,'change') for d in self.config['dependencies'] for t in d['targets']))
        events={target[1] for d in self.config['dependencies'] for target in d['targets']}
        self.assertTrue({'start_recording','stop_recording'}<=events)
        larger=next(d for d in self.config['dependencies'] if any(tuple(t)==(larger_id,'change') for t in d['targets']))
        self.assertIn('vb-large-text',larger['js']);self.assertFalse(larger['backend_fn'])
        transcribe=next(d for d in self.config['dependencies'] if d['api_name']=='transcribe_question')
        audio=self.components[transcribe['inputs'][0]]
        self.assertEqual(audio['props']['sources'],['microphone','upload'])

    def test_reset_clears_old_response_without_model_call(self):
        reset=next(f.fn for f in self.demo.fns.values() if f.api_name=='reset')
        result=reset(Image.new('RGB',(10,10)))
        self.assertEqual(result[0],'');self.assertIsNone(result[1]);self.assertEqual(result[3],{});self.assertIn('Image ready',result[2])


if __name__=='__main__':unittest.main()
