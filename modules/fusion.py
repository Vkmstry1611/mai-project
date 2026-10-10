from dataclasses import dataclass
from typing import Protocol

@dataclass
class FusionInput:
    question: str
    baseline_answer: str
    evidence: list
    visual_features: dict

@dataclass
class FusionOutput:
    answer: str
    grounded: bool = False

class FusionAdapter(Protocol):
    def predict(self, inputs: FusionInput) -> FusionOutput: ...

def baseline(inputs):
    return FusionOutput(inputs.baseline_answer, grounded=False)
