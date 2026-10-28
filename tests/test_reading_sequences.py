"""Synthetic layout regressions; never load evaluation datasets or models."""
import copy
import pytest
from PIL import Image
from test_reading_relevance import row
from test_image_conversation import fake_base,turn
from visionbridge.reading_relevance import compose,intent
from visionbridge.full_text_session import FullTextSession
from visionbridge.image_conversation import prepare


def cooking():
    return [row('You Will Need',0,0),row('1 Lb ground beef',600,0),
            row('Brown ground beef',10,180),row('in a skillet; drain.',10,220),
            row('Stir in hot water,',10,280),row('milk and pasta.',10,320),
            row('Reduce heat.',10,380),row('Cover and simmer.',10,420),
            row('unreadable tail',10,450,score=.4)]


@pytest.mark.parametrize('q',['Tell me the cooking instructions mentioned in the image?',
    'Read the cooking steps','Read the recipe','Tell me the preparation directions'])
def test_explicit_instruction_routing(q):
    assert intent(q)=='instructions'
    assert object.__new__(FullTextSession).classify(q).task=='reading'


def test_directions_exclude_ingredients_preserve_words_and_uncertainty():
    data={'evidence':cooking()};before=copy.deepcopy(data)
    result=compose('Tell me the cooking instructions',data)
    assert 'Brown ground beef' in result['answer'] and 'Cover and simmer.' in result['answer']
    assert '1 Lb' not in result['answer'] and 'You Will Need' not in result['answer']
    assert 'unreadable tail' not in result['answer'] and result['partial']
    assert '1.' not in result['answer']  # Never invent undetected step numbers.
    assert data==before and result['evidence']==before['evidence']
    assert any(g['kind']=='instruction_sequence' for g in result['regions'])


def test_numbered_steps_and_wraps_keep_source_order():
    rows=[row('Cooking directions',0,0),row('1.',0,60,w=20),row('Brown the meat.',35,60),
          row('Drain carefully.',35,100),row('2.',0,160,w=20),row('Stir in water.',35,160),
          row('3.',0,250,w=20),row('Reduce heat.',35,250),row('Ingredients: salt',650,50)]
    result=compose('Read cooking steps',{'evidence':list(reversed(rows))})
    answer=result['answer']
    assert answer.index('1.')<answer.index('Brown')<answer.index('Drain')<answer.index('2.')<answer.index('Stir')<answer.index('3.')
    assert 'Ingredients' not in answer


def passages():
    return [row('This is the left passage.',0,y) for y in (0,40,80)]+[row('This is the right passage.',1000,y) for y in (0,40,80)]


@pytest.mark.parametrize('reference',['First passage','1st passage','Read the first one','Start with first passage.'])
def test_ordinal_resolves_pending_passages_with_one_extraction(tmp_path,reference):
    base,calls=fake_base(tmp_path,passages());image=Image.new('RGB',(1600,300));state=prepare(None,image)
    initial=turn(base,state,image,'Could you read the label?')
    assert 'multiple possible' in initial[0] and len(state['conversation']['passage_candidates'])==2
    result=turn(base,state,image,reference)
    assert 'left passage' in result[0] and 'right passage' not in result[0]
    assert calls==['ocr'] and result[3]['assistant_trace']['ocr_cache_hit']
    rest=turn(base,state,image,'Continue')
    assert 'no additional' in rest[0] and 'multiple possible' not in rest[0]


def test_second_all_and_beginning_no_repeated_clarification(tmp_path):
    base,calls=fake_base(tmp_path,passages());image=Image.new('RGB',(1600,300));state=prepare(None,image)
    turn(base,state,image,'Read the label')
    second=turn(base,state,image,'Second passage');assert 'right passage' in second[0] and 'left passage' not in second[0]
    beginning=turn(base,state,image,'Start from the beginning');assert 'right passage' in beginning[0]
    every=turn(base,state,image,'Read all passages');assert 'left passage' in every[0] and 'right passage' in every[0]
    assert calls==['ocr']


def test_cooking_scope_is_reused_and_new_image_resets_candidates(tmp_path):
    base,calls=fake_base(tmp_path,cooking());image=Image.new('RGB',(1200,600));state=prepare(None,image)
    result=turn(base,state,image,'Tell me the cooking instructions mentioned in the image?')
    assert 'Brown ground beef' in result[0] and calls==['ocr']
    assert 'multiple possible' not in turn(base,state,image,'Could you read the label?')[0]
    reset=prepare(state,Image.new('RGB',(1200,600),'blue'))
    assert not reset['conversation'].get('passage_candidates')


def test_missing_and_out_of_range_ordinal_do_not_infer_visual(tmp_path):
    base,calls=fake_base(tmp_path,passages());image=Image.new('RGB',(1600,300));state=prepare(None,image)
    facade=base.for_conversation(state,image);route=facade.classify('First passage')
    assert route.task=='clarify' and 'passage' in route.message and not calls
    turn(base,state,image,'Read the label')
    facade=base.for_conversation(state,image);route=facade.classify('Third passage')
    assert route.task=='clarify' and '2' in route.message and calls==['ocr']


def test_read_every_visible_word_still_includes_other_regions():
    result=compose('Read every visible word',{'evidence':cooking()})
    assert '1 Lb' in result['answer'] and 'Brown ground beef' in result['answer']


def test_weak_markers_and_conflicting_columns_not_invented():
    rows=[row('1.',0,0,w=20,score=.4),row('Brown beef',40,0),row('Stir water',40,70),
          row('1.',1000,0,w=20),row('Bake bread',1040,0),row('2.',1000,70,w=20),row('Cool bread',1040,70)]
    result=compose('Read cooking instructions',{'evidence':rows})
    assert 'multiple possible' in result['answer']


def test_generic_inline_ordered_list_and_separate_recipe_headings():
    rows=[row('1. Open package.',0,0),row('2. Check the label.',0,60)]
    result=compose('Read the numbered list',{'evidence':rows})
    assert result['answer'].index('1.')<result['answer'].index('2.')
    recipes=[row('Recipe one',0,0),row('Bake the bread.',0,60),row('Cool the bread.',0,100),
             row('Recipe two',0,190),row('Brown the beef.',0,250),row('Stir in water.',0,290)]
    result=compose('Read cooking instructions',{'evidence':recipes})
    assert 'multiple possible' in result['answer']


def test_side_by_side_candidates_ignore_small_detector_y_jitter():
    rows=[row('This is the left passage.',0,y+2) for y in (0,40,80)]+[row('This is the right passage.',1000,y) for y in (0,40,80)]
    result=compose('Read this passage',{'evidence':rows})
    assert 'left passage' in result['passage_candidates'][0]['preview']


def test_explicit_website_question_keeps_priority_over_recipe_context():
    assert intent('What website is shown for this cooking recipe?')=='interface'


def test_continue_after_summary_reads_only_unread_source_lines(tmp_path):
    rows=[row('This is source line '+str(i)+'.',0,40*i) for i in range(7)]
    base,calls=fake_base(tmp_path,rows);image=Image.new('RGB',(800,300));state=prepare(None,image)
    first=turn(base,state,image,'Summarize this passage')
    assert 'source line 5' in first[0] and 'source line 6' not in first[0]
    rest=turn(base,state,image,'Continue')
    assert 'source line 6' in rest[0] and 'source line 5' not in rest[0] and calls==['ocr']
