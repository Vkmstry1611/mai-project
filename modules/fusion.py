from dataclasses import dataclass, field
from typing import Protocol

@dataclass
class FusionInput:
    question: str
    baseline_answer: str
    evidence: list
    visual_features: dict
    selection_truncated: bool = False
    ocr_min_score: float = 0.9
    extraction_status: dict = field(default_factory=dict)

@dataclass
class FusionOutput:
    answer: str
    grounded: bool = False
    baseline_answer: str = ''
    reason: str = 'BLIP retained; evidence-aware decisions disabled.'
    supporting_evidence_ids: list = field(default_factory=list)
    decision: str = 'blip_retained'
    source: str = 'blip'
    candidate_trace: dict = field(default_factory=dict)

class FusionAdapter(Protocol):
    def predict(self, inputs: FusionInput) -> FusionOutput: ...

def baseline(inputs):
    return FusionOutput(inputs.baseline_answer, grounded=False, baseline_answer=inputs.baseline_answer)
