from dataclasses import dataclass
from typing import Protocol

@dataclass
class GateResult:
    answerable: bool | None
    probability: float | None = None
    reason: str = 'Trained answerability gate unavailable; reliability is not assessed.'

class AnswerabilityAdapter(Protocol):
    def predict(self, inputs) -> GateResult: ...

def unavailable():
    return GateResult(None)
