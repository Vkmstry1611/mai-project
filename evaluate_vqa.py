"""VizWiz VQA accuracy using normalization tables from its official API.

The API normalizes predictions, compares them to the supplied annotations,
and averages over the ten leave-one-annotator-out sets. This module does the
same without importing the API's optional Java/caption/scikit-learn tools.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

API_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/API.zip"
API_MEMBER = "PythonEvaluationTools/vqaEvaluation/vqaEval.py"
REFERENCE_POLICY = "vizwiz-raw-ten-slots-valid-targets-v1"


def reference_answer(answer):
    """Return an original usable string, never a fabricated/coerced label."""
    if isinstance(answer, dict):
        value = answer.get("answer")
        if isinstance(value, str) and value.strip():
            return value
    return None


def usable_answers(answers):
    """Preserve order and duplicate frequencies; do not mutate raw references."""
    return [value for answer in answers if (value := reference_answer(answer)) is not None]


def extract_rules(archive, destination):
    """Read literal tables only: no downloaded Python code is executed."""
    with zipfile.ZipFile(archive) as bundle:
        source = bundle.read(API_MEMBER)
    tree = ast.parse(source.decode("utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VQAEval")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    tables = {}
    for node in init.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
            name = node.targets[0].attr
            if name in {"contractions", "manualMap", "articles", "punct"}:
                tables[name] = ast.literal_eval(node.value)
    if set(tables) != {"contractions", "manualMap", "articles", "punct"}:
        raise ValueError("Official API tables changed; review before evaluating")
    rules = {"source": API_URL, "source_sha256": hashlib.sha256(source).hexdigest(), **tables}
    path = Path(destination)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != rules:
            raise ValueError("Existing evaluation rules differ; refusing overwrite")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(rules, stream, indent=2, ensure_ascii=True)
    return OfficialScorer(rules)


class OfficialScorer:
    def __init__(self, rules):
        self.rules = rules
        self.identity = {"metric": "official VizWiz VQA accuracy (0..1)",
                         "source": rules["source"], "source_sha256": rules["source_sha256"],
                         "reference_policy": REFERENCE_POLICY}
        self.period = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.comma = re.compile(r"(\d)(\,)(\d)")

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def normalize(self, text):
        text = str(text).replace("\n", " ").replace("\t", " ").strip()
        result = text
        for punctuation in self.rules["punct"]:
            replacement = "" if (punctuation + " " in text or " " + punctuation in text
                                   or self.comma.search(text)) else " "
            result = result.replace(punctuation, replacement)
        result = self.period.sub("", result, re.UNICODE)
        words = [self.rules["manualMap"].get(word, word) for word in result.lower().split()
                 if word not in self.rules["articles"]]
        return " ".join(self.rules["contractions"].get(word, word) for word in words)

    def score(self, prediction, answers):
        if len(answers) != 10:
            raise ValueError("Official VizWiz evaluation requires ten reference answers")
        prediction = self.normalize(prediction)
        # Keep ten annotator slots and the official denominator. Invalid slots
        # never match; an empty prediction never earns credit for missing data.
        # For nonempty predictions and official string references, this agrees
        # with the official API, including its raw-reference comparison.
        if not prediction:
            return 0.0
        refs = [reference_answer(answer) for answer in answers]
        return sum(min(1.0, sum(prediction == answer for j, answer in enumerate(refs) if j != i) / 3)
                   for i in range(10)) / 10


def summarize(rows, predictions, scorer):
    if not rows or len(rows) != len(predictions):
        raise ValueError("Every evaluation example must have a prediction; no silent skips")
    records, groups = [], {"answerable": {}, "answer_type": {}}
    for row, prediction in zip(rows, predictions):
        score = scorer.score(prediction, row["answers"])
        record = {"image": row["image"], "question": row["question"],
                  "prediction": prediction.strip(), "answers": row["answers"],
                  "answerable": row["answerable"], "answer_type": row["answer_type"], "score": score}
        records.append(record)
        for key in groups:
            groups[key].setdefault(str(row[key]), []).append(score)
    metrics = {"examples": len(rows), "vqa_score": sum(r["score"] for r in records) / len(rows)}
    for key, values in groups.items():
        metrics["by_" + key] = {label: {"examples": len(scores), "vqa_score": sum(scores) / len(scores)}
                               for label, scores in values.items()}
    # This is discrete output behavior, not a probability or trained gate metric.
    abstentions = [scorer.normalize(p) == "unanswerable" for p in predictions]
    metrics["unanswerable_output_rate"] = sum(abstentions) / len(rows)
    metrics["answered_coverage"] = 1 - metrics["unanswerable_output_rate"]
    return metrics, records
