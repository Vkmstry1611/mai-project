import copy

import pytest

from benchmark_warm_requests import stats
from submission_ui_helpers import evidence_text, input_error, with_evidence_text


@pytest.mark.parametrize('image,question',[(None,'Read this'),(object(),''),(object(),'  '),(object(),None)])
def test_missing_input_is_explained(image,question):
    assert input_error(image,question)


def test_valid_input_passes():
    assert input_error(object(),'Read this') is None


def test_presentation_preserves_every_frozen_output():
    details={'final_answer':'STOP','baseline_answer':'stop','selected_evidence':[{'id':'ocr-1','text':'STOP','kind':'ocr','box':[1,2,3,4]}],
             'supporting_evidence_ids':[],'extraction_status':{'ocr':{'status':'complete'}}}
    before=copy.deepcopy(details)
    response=('STOP',None,'Complete',details,object())
    presented=with_evidence_text(response)
    assert presented[:5]==response
    assert presented[3] is details
    assert details==before
    assert 'used as final support: no' in presented[5]
    assert 'ocr extraction: complete' in presented[5]


def test_support_and_no_evidence_text_are_distinct():
    assert 'No selected evidence' in evidence_text({})
    assert 'used as final support: yes' in evidence_text({'selected_evidence':[{'id':'ocr-1'}],'supporting_evidence_ids':['ocr-1']})


def test_latency_statistics_include_only_supplied_times():
    result=stats([2.,4.,6.])
    assert result['median_seconds']==4
    assert result['mean_seconds']==4
    assert result['p95_seconds_nearest_rank']==6
