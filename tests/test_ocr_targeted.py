"""Diagnostic regressions: synthetic OCR only, no references in selection."""
from modules import qges
from modules.evidence_decision import decide
from modules.fusion import FusionInput
from modules.question_intent import classify


def item(text, identity, box, confidence=.99):
    return dict(kind='ocr', id=identity, text=text, box=box,
                confidence=confidence, status='complete')


def answer(question, items, **kwargs):
    selected = qges.select(question, items, top_k=None)
    return decide(FusionInput(question, 'wrong', selected, {}, **kwargs))


def test_noise_does_not_globally_veto_readable_candidate():
    result = answer('What does this message say?', [
        item('Restoration Complete', 'good', [0, 0, 180, 20]),
        item('noise', 'bad', [300, 300, 320, 310], .1)])
    assert result.answer == 'Restoration Complete'
    assert result.supporting_evidence_ids == ['good']


def test_phrase_order_uses_geometry_and_preserves_all_ids():
    items = [item('Complete', 'b', [85, 0, 165, 20]),
             item('Restoration', 'a', [0, 0, 80, 20])]
    result = answer('What does this say?', items)
    assert result.answer == 'Restoration Complete'
    assert result.supporting_evidence_ids == ['a', 'b']
    assert answer('What does this say?', list(reversed(items))).answer == result.answer


def test_slight_vertical_offset_does_not_reverse_reading_order():
    items = [item('Restoration', 'a', [0, 3, 80, 23]),
             item('Complete', 'b', [85, 0, 165, 20])]
    assert answer('What does this say?', items).answer == 'Restoration Complete'


def test_compact_multiline_phrase_and_separate_regions():
    compact = [item('Spring Branch', 'a', [0, 0, 140, 20]),
               item('Presbyterian Academy', 'b', [0, 25, 150, 45])]
    assert answer('What is written on this shirt?', compact).answer == 'Spring Branch Presbyterian Academy'
    separate = [item('STOP', 'a', [0, 0, 60, 20]),
                item('NOTICE', 'b', [300, 200, 380, 220])]
    assert answer('What does this say?', separate).source == 'blip'


def test_curved_shirt_layout_is_not_forced_into_a_reference_order():
    # Diagnosed geometry: uncertain curved/overlapping boxes must stay ambiguous.
    items = [item('Presbyterian', 'a', [285, 581, 631, 691], .997),
             item('Spring', 'b', [383, 521, 577, 614], .960),
             item('Branch', 'c', [576, 484, 787, 577], .967),
             item('Academy', 'd', [629, 534, 897, 631], .984)]
    assert answer('What is written on this shirt?', items).source == 'blip'


def test_dropped_middle_word_is_not_silently_assembled():
    items = [item('KEEP', 'a', [0, 0, 40, 20]),
             item('NOT', 'b', [45, 0, 75, 20], .2),
             item('ENTER', 'c', [80, 0, 130, 20])]
    assert answer('What does this sign say?', items).source == 'blip'


def test_polite_reading_and_brand_routing_not_verification_or_quantities():
    assert classify('Can you tell me what this says, please?').kind == 'reading'
    assert classify('Can you read this to me please?').kind == 'reading'
    assert classify('Could you identify the brand of wine?').kind == 'brand'
    assert not classify('Can you read this?').supported
    assert not classify('Is the brand Acme?').supported
    assert classify('How much fat does it contain?').kind == 'general'


def test_prominent_brand_literal_not_any_arbitrary_word():
    items = [item('ExampleBrand', 'a', [0, 0, 180, 50]),
             item('bottle noise', 'b', [0, 100, 80, 110], .2)]
    assert answer('Could you identify the brand?', items).answer == 'ExampleBrand'
    assert answer('What brand is this?', [items[0]]).source == 'blip'
    competing = items + [item('AnotherBrand', 'c', [220, 0, 400, 50])]
    assert answer('What brand is this?', competing).source == 'blip'


def test_unknown_partial_and_blip_abstention_still_retain():
    selected = qges.select('What does this say?', [item('WELCOME', 'a', [0, 0, 100, 20])], None)
    result = decide(FusionInput('What does this say?', 'wrong', selected, {},
                               extraction_status={'ocr': {'status': 'partial', 'rejected_items': 1}}))
    assert result.source == 'blip'
    assert decide(FusionInput('What does this say?', 'unanswerable', selected, {})).answer == 'unanswerable'


def test_known_partial_outside_detected_target_is_safe_but_overlap_is_not():
    bottle = dict(kind='object', id='object-a', text='bottle', box=[0, 0, 200, 200], confidence=.99)
    items = [bottle, item('Brand: Acme', 'a', [10, 10, 140, 30])]
    statuses = {'ocr': {'status': 'partial', 'rejected_items': 1,
                       'rejections': [{'box': [300, 300, 340, 320], 'reason': 'outside_image'}]},
                'object': {'status': 'complete'}}
    assert answer('What brand is on the bottle?', items, extraction_status=statuses).answer == 'Acme'
    statuses['ocr']['rejections'][0]['box'] = [0, 0, 100, 30]
    assert answer('What brand is on the bottle?', items, extraction_status=statuses).source == 'blip'
