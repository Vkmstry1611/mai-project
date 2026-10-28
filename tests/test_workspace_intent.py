"""English rule routing and delegation contracts; no neural inference."""
import unittest
from types import SimpleNamespace
from PIL import Image
from visionbridge.workspace_intent import classify


class IntentTests(unittest.TestCase):
    def test_general_visual_questions(self):
        for question in ['What is this object?','What color is the bottle?','Describe the room.',
                         'How many dogs are there?','What is the person doing?','What is the person reading?',
                         'What shape is this?','What color is the price tag?','Is there a cat?']:
            with self.subTest(question=question):self.assertEqual(classify(question).task,'vqa')

    def test_explicit_reading_questions(self):
        for question in ['Read this.','Can you please read this sign?','What is written in this image?',
                         'What words are printed on this label?','What does this say?',
                         'What does the sign say?','Read the visible text aloud.','Please read the label to me.',
                         'Tell me what the label says.']:
            with self.subTest(question=question):self.assertEqual(classify(question).task,'reading')

    def test_specific_fields_are_explicitly_unsupported(self):
        for question in ['What is the expiry date?','When does this expire?','Read the expiration date.',
                         'What is the manufacturing date?','What is the price?','How much is this?',
                         'What is the product name?','What date is on this label?','When was this made?',
                         'How many days until the expiry date?','Is this expired?','Tell me what this costs.']:
            with self.subTest(question=question):
                route=classify(question)
                self.assertEqual(route.task,'unsupported_field');self.assertIn('cannot reliably',route.message)

    def test_ambiguous_negated_and_mixed_requests_clarify(self):
        for question in ['Help','Can you help me?','What about this?','Tell me about this',"Don't read this.",
                         'Read the text and tell me its color.','Translate this text.','What does this label mean?']:
            with self.subTest(question=question):
                route=classify(question)
                self.assertEqual(route.task,'clarify');self.assertTrue(route.message)

    def test_normalization_is_deterministic(self):
        self.assertEqual(classify('  READ   THIS. '),classify('read this.'))
        self.assertEqual(classify('What is this object?'),classify('What is this object?'))

    def test_unsupported_reader_wording_does_not_silently_use_blip(self):
        self.assertEqual(classify('Transcribe the sign.').task,'clarify')


class AssistantControllerTests(unittest.TestCase):
    def setUp(self):
        from visionbridge.workspace_ui import WorkspaceController
        self.calls=[];self.spoken=[];self.photo=Image.new('RGB',(120,60),'white')
        self.details={'final_answer':'HELLO','baseline_answer':'hello','answer_source':'ocr','selected_evidence':[]}
        self.timings={'callback_seconds':1,'experimental_reading_active':True}
        def request(*args):self.calls.append(args);return ('HELLO',None,'Complete',self.details,self.photo,'evidence',self.timings)
        def speaker(text):self.spoken.append(text);return 'notice.wav'
        self.controller=WorkspaceController(SimpleNamespace(request=request),lambda p:('Read this','Review'),speaker)

    def test_both_automatic_routes_preserve_outputs(self):
        from visionbridge.workspace_ui import ASSISTANT
        for question,reading in [('What color is this?',False),('Read this',True)]:
            with self.subTest(question=question):
                result=list(self.controller.analyze(self.photo,question,False,ASSISTANT,'180'))[-1]
                self.assertEqual(self.calls[-1],(self.photo,question,False,reading,'0'))
                self.assertEqual(result[0],'HELLO');self.assertIs(result[3],self.details);self.assertIs(result[6],self.timings)
                self.assertIn('automatic',result[10]);self.assertNotIn('Experimental OCR',result[7])
                self.assertNotIn('workspace_routing',self.details)

    def test_limitation_and_clarification_do_not_run_inference(self):
        from visionbridge.workspace_ui import ASSISTANT
        for question,task in [('What is the expiry date?','unsupported_field'),('Help','clarify')]:
            result=list(self.controller.analyze(self.photo,question,True,ASSISTANT,'0'))[-1]
            self.assertFalse(self.calls);self.assertEqual(result[1],'notice.wav')
            self.assertEqual(result[3]['decision'],'workspace_'+task)
            self.assertEqual(result[3]['answer_source'],'assistant_notice')
            self.assertFalse(result[6]['model_inference_performed']);self.assertEqual(result[3]['selected_evidence'],[])
            self.assertEqual(self.spoken[-1],result[0])

    def test_manual_override_keeps_existing_frozen_behavior(self):
        from visionbridge.workspace_ui import VISUAL,READING
        for mode,question,reading,rotation in [(VISUAL,'What is the expiry date?',False,'0'),(READING,'Read this',True,'180')]:
            result=list(self.controller.analyze(self.photo,question,False,mode,'180'))[-1]
            self.assertEqual(self.calls[-1],(self.photo,question,False,reading,rotation))
            self.assertIs(result[3],self.details);self.assertIn('developer override',result[10])

    def test_notice_voice_failure_preserves_text(self):
        from visionbridge.workspace_ui import ASSISTANT
        def fail(text):raise RuntimeError('voice unavailable')
        self.controller.speaker=fail
        result=list(self.controller.analyze(self.photo,'Help',True,ASSISTANT,'0'))[-1]
        self.assertTrue(result[0]);self.assertIsNone(result[1]);self.assertIn('Spoken response unavailable',result[2])


if __name__=='__main__':unittest.main()
