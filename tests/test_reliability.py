from unittest.mock import patch
from PIL import Image
from pipeline import Pipeline
from modules.reliability import mentioned_objects, check_objects, question_check

IMAGE = Image.new('RGB', (30, 30))

def test_weather_abstains_before_vqa_and_speaks_response():
    with patch('modules.vqa.answer') as model, patch('modules.text_to_speech.synthesize', return_value='speech.wav') as speech:
        result = Pipeline().run(IMAGE, 'What is the weather?')
    assert result[0] == "I can't determine the weather from this image."
    model.assert_not_called()
    assert speech.call_args.args[0] == result[0]
    assert result[3]['answerability']['probability'] is None

def test_missing_object_does_not_generate_attributes():
    with patch('modules.object_detection.extract', return_value=[{'kind': 'object', 'text': 'cat', 'confidence': 0.9}]), patch('modules.vqa.answer') as model:
        result = Pipeline().run(IMAGE, 'What color is the bottle?', speak=False)
    assert "I can't see that item" in result[0]
    assert 'bottle' in result[0]
    model.assert_not_called()
    assert result[3]['reliability_checks']['decision'] == 'abstain'

def test_present_object_still_has_no_answer_confidence():
    with patch('modules.object_detection.extract', return_value=[{'kind': 'object', 'text': 'bottle', 'confidence': 0.9}]), patch('modules.vqa.answer', return_value='blue'):
        result = Pipeline().run(IMAGE, 'What color is the bottle?', speak=False)
    assert result[0] == 'blue'
    assert result[3]['reliability_checks']['decision'] == 'supported_objects'
    assert result[3]['reliability_checks']['probability'] is None

def test_detector_failure_is_not_claimed_absence():
    with patch('modules.object_detection.extract', side_effect=RuntimeError('not installed')), patch('modules.vqa.answer') as model:
        result = Pipeline().run(IMAGE, 'Is there a dog?', speak=False)
    assert "can't verify" in result[0]
    assert "can't see" not in result[0]
    model.assert_not_called()

def test_long_phrases_aliases_and_word_boundaries():
    assert mentioned_objects('What color is the hot dog?') == ['hot dog']
    assert mentioned_objects('How many people are next to the sofa?') == ['person', 'couch']
    assert mentioned_objects('What is the category?') == []
    assert mentioned_objects('Are there cats?') == ['cat']

def test_low_detection_score_is_not_enough():
    result = check_objects(['bottle'], [{'kind': 'object', 'text': 'bottle', 'confidence': 0.2}], True)
    assert result.decision == 'abstain'

def test_checks_can_be_disabled_for_comparison():
    with patch('modules.object_detection.extract') as detector, patch('modules.vqa.answer', return_value='sunny'):
        result = Pipeline().run(IMAGE, 'What is the weather?', speak=False, reliability_checks=False)
    assert 'sunny' in result[0]
    detector.assert_not_called()

def test_name_of_object_is_not_personal_identity():
    assert question_check('What is the name of this object?').response is None
    assert question_check('What are the names of these cats?').decision == 'abstain'
