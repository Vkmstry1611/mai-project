import copy
from PIL import Image
from modules.answerability_features import features, image_quality


def sample():
    return {'question':'Please read the label','baseline_answer':'milk','final_answer':'MILK','answer_source':'ocr',
            'ocr_question_intent':'reading','decision':'ocr_supported','ocr_evidence':[{'kind':'ocr','text':'MILK','confidence':.97,'box':[1,1,20,10]}],
            'selected_evidence':[{'kind':'ocr','text':'MILK','confidence':.97,'box':[1,1,20,10]}],
            'extraction_status':{'ocr':{'status':'complete'},'object':{'status':'disabled'}},
            'candidate_trace':{'candidates':[{'text':'MILK'}]},'supporting_evidence_ids':['ocr-1']}


def test_reference_label_score_changes_cannot_change_features():
    row=sample();quality=image_quality(Image.new('RGB',(32,24),'white'))
    before=features(row,quality)
    row.update(answers=[{'answer':'unanswerable'}]*10,answerable=0,answer_type='unanswerable',final_vqa_score=0,
               baseline_vqa_score=1,score_delta=-1,ocr_corrected_blip=True,image='different-id',peak_vram_mb=9999,processing_seconds=123)
    assert features(row,quality)==before
    assert not any(k in before for k in ('answerable','answers','image','final_vqa_score','answer_type'))


def test_extraction_disabled_empty_unavailable_are_distinct():
    row=sample();quality=image_quality(Image.new('RGB',(16,16)))
    for status in ('disabled','empty','unavailable'):
        row['extraction_status']['ocr']['status']=status
        value=features(row,quality)
        assert value['ocr_status_'+status]==1
        assert sum(value['ocr_status_'+s] for s in ('disabled','empty','complete','partial','unavailable','unknown'))==1


def test_missing_ocr_is_not_forced_abstention_and_schema_stable():
    row=sample();q=image_quality(Image.new('RGB',(16,16)))
    baseline=features(row,q);row['ocr_evidence']=[];row['selected_evidence']=[]
    value=features(row,q)
    assert set(value)==set(baseline)
    assert value['has_ocr']==0 and value['candidate_abstained']==0


def test_image_quality_is_deterministic():
    image=Image.new('RGB',(40,20),(24,180,70))
    assert image_quality(image)==image_quality(image.copy())


def test_candidate_matches_selected_high_confidence_evidence_only():
    row=sample();q=image_quality(Image.new('RGB',(32,24)))
    value=features(row,q)
    assert value['candidate_matches_selected_ocr']==1 and value['candidate_selected_ocr_token_overlap']==1
    row['selected_evidence'][0]['confidence']=.89
    value=features(row,q)
    assert value['candidate_matches_selected_ocr']==0 and value['candidate_selected_ocr_token_overlap']==0
