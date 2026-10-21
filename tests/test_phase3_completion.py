import pytest
from verify_phase3_completion import reference_identity, risk_points, same


def test_reference_frequency_and_order_must_be_preserved():
    row={'image':'development.jpg','question':'What is written?', 'answerable':1,
         'answers':[{'answer':'milk'}]*6+[{'answer':'water'}]*4}
    reference_identity(row,row)
    altered={**row,'answers':[{'answer':'milk'}]*5+[{'answer':'water'}]*5}
    with pytest.raises(ValueError,match='answers changed'):
        reference_identity(altered,row)


def test_metric_audit_accepts_roundoff_but_rejects_false_gain():
    same({'score':60.70000000001},{'score':60.7},'result')
    with pytest.raises(ValueError,match='differs'):
        same({'score':61.7},{'score':60.7},'result')


def test_zero_answer_risk_is_not_plotted_as_observed_error():
    values=[{'name':'both_heads','answered_examples':0,'answered_subset_risk':1.},
            {'name':'both_heads','answered_examples':1,'answered_subset_risk':1.}]
    assert risk_points(values,'both_heads')==values[1:]
