from types import SimpleNamespace

import pytest
import torch

from modules.vqa import validate_blip_loading


def model_fixture(tie=True, output_shape=(5, 3)):
    embeddings = SimpleNamespace(weight=torch.nn.Parameter(torch.arange(15, dtype=torch.float32).reshape(5, 3)))
    decoder = SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(output_shape)))
    model = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(tie_word_embeddings=tie)),
        text_decoder=SimpleNamespace(bert=SimpleNamespace(embeddings=SimpleNamespace(word_embeddings=embeddings)),
                                     cls=SimpleNamespace(predictions=SimpleNamespace(decoder=decoder))))
    return model, embeddings, decoder


def test_restores_only_expected_blip_tied_decoder_projection():
    model, embeddings, decoder = model_fixture()
    returned = validate_blip_loading(model, {'missing_keys': ['text_decoder.cls.predictions.decoder.weight'],
                                             'unexpected_keys': [], 'mismatched_keys': [], 'error_msgs': []})
    assert returned is model
    assert decoder.weight is embeddings.weight


@pytest.mark.parametrize('loading_info', [
    {'missing_keys': ['vision_model.some_weight']},
    {'missing_keys': [], 'unexpected_keys': ['unexpected.weight']},
    {'missing_keys': [], 'mismatched_keys': ['vision.weight']},
    {'missing_keys': [], 'error_msgs': ['bad checkpoint']},
])
def test_rejects_any_other_incomplete_checkpoint(loading_info):
    model, _, _ = model_fixture()
    with pytest.raises(RuntimeError):
        validate_blip_loading(model, loading_info)


def test_missing_projection_requires_declared_tie_and_matching_shape():
    missing = {'missing_keys': ['text_decoder.cls.predictions.decoder.weight']}
    model, _, _ = model_fixture(tie=False)
    with pytest.raises(RuntimeError, match='does not declare'):
        validate_blip_loading(model, missing)
    model, _, _ = model_fixture(output_shape=(4, 3))
    with pytest.raises(RuntimeError, match='incompatible shapes'):
        validate_blip_loading(model, missing)
