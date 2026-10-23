"""Offline tests. Only tiny synthetic models receive optimizer updates."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

from evaluate_vqa import (API_MEMBER, OfficialScorer, extract_rules, summarize,
                         REFERENCE_POLICY, usable_answers)
from visionbridge_training import (TrainConfig, accumulation_update, annotation_rows,
    batches_for_epoch, chosen_answer, create_run, guard_output, load_snapshot_weights,
    prepare_manifest, read_json, save_snapshot, sha256, stratified_sample,
    validation_decision, verify_checkpoint, write_json, extract_named, download_once, final_compare)


def row(index, split="train", answerable=1, answer_type="other"):
    return {"image": f"{split}_{index}.jpg", "question": "What is this?",
            "answerable": answerable, "answer_type": answer_type,
            "answers": [{"answer": "cat"}] * 10}


class DataSafetyTests(unittest.TestCase):
    def test_exact_reproducible_stratification(self):
        rows = [row(i, answerable=int(i % 5 != 0)) for i in range(100)]
        first = stratified_sample(rows, 25, 42)
        self.assertEqual(first, stratified_sample(rows, 25, 42))
        self.assertEqual(len(first), 25)
        self.assertEqual(sum(r["answerable"] == 0 for r in first), 5)
        self.assertEqual(len({r["image"] for r in first}), 25)

    def test_manifest_is_disjoint_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, val = root / "train.json", root / "val.json"
            write_json(train, [row(i) for i in range(20)])
            write_json(val, [row(i, "val") for i in range(10)])
            config = TrainConfig(train_size=8, dev_size=4)
            manifest = prepare_manifest(config, root, train, val)
            self.assertEqual(len(manifest["holdout"]), 6)
            names = [{r["image"] for r in manifest[s]} for s in ("train", "dev", "holdout")]
            self.assertFalse(names[0] & names[1] or names[0] & names[2] or names[1] & names[2])
            self.assertEqual(manifest, prepare_manifest(config, root, train, val))
            write_json(val, [row(i, "different") for i in range(10)])
            with self.assertRaises(ValueError):
                prepare_manifest(config, root, train, val)

    def test_output_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input"
            for output in (source, source / "run", Path(tmp)):
                with self.assertRaises(ValueError):
                    guard_output(source, output)
            guard_output(source, Path(tmp) / "new_run")

    def test_lfs_pointer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.txt"):
                (root / name).write_text("{}")
            (root / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "LFS pointer"):
                verify_checkpoint(root)

    def test_archive_keeps_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("folder/image.jpg", b"original image bytes")
            extract_named(archive, ["image.jpg"], root / "images")
            extract_named(archive, ["image.jpg"], root / "images")
            path = root / "images/image.jpg"
            path.write_bytes(b"local changed bytes")
            with self.assertRaises(ValueError):
                extract_named(archive, ["image.jpg"], root / "images")
            self.assertEqual(path.read_bytes(), b"local changed bytes")
            with self.assertRaises(ValueError):
                extract_named(archive, ["../outside.jpg"], root / "images")

    def test_invalid_annotations_fail_instead_of_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ann.json"
            invalid = row(0)
            invalid["answers"] = []
            write_json(path, [invalid])
            with self.assertRaises(ValueError):
                annotation_rows(path)

    def test_individual_invalid_answers_keep_raw_references_and_target_frequencies(self):
        item = row(0, answerable=0, answer_type="unanswerable")
        item["answers"] = ([{"answer": "cat", "answer_confidence": "yes"}] * 3
                           + [{"answer": "kitten"}] * 2
                           + [{"answer": ""}, {"answer": " \t"}, {"answer": None}, {}, 42])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ann.json"
            write_json(path, [item])
            rows, audit = annotation_rows(path, return_audit=True)
            self.assertEqual(rows, [item])
            self.assertEqual(read_json(path), [item])
            self.assertEqual(usable_answers(rows[0]["answers"]), ["cat"] * 3 + ["kitten"] * 2)
            self.assertEqual(audit["invalid_answer_slots"], 5)
            self.assertEqual(rows[0]["answerable"], 0)
            self.assertEqual(chosen_answer(rows[0], 42, 0, False), "cat")
            import random
            for epoch in range(20):
                expected = random.Random(f"42:{epoch}:{item['image']}").choice(["cat"] * 3 + ["kitten"] * 2)
                self.assertEqual(chosen_answer(rows[0], 42, epoch, True), expected)

    def test_no_usable_answers_excluded_before_deterministic_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, val = root / "train.json", root / "val.json"
            missing_train, missing_val = row(100, answerable=0), row(100, "val", answerable=1)
            missing_train["answers"] = [{"answer": ""}] * 10
            missing_val["answers"] = [None, {}, {"answer": False}, {"answer": 5}, {"answer": " "}] * 2
            mixed = row(0, "val")
            mixed["answers"][-1] = {"answer": ""}
            write_json(train, [row(i) for i in range(20)] + [missing_train])
            write_json(val, [mixed] + [row(i, "val") for i in range(1, 10)] + [missing_val])
            config = TrainConfig(train_size=8, dev_size=4)
            first_dir, second_dir = root / "one", root / "two"
            first_dir.mkdir(); second_dir.mkdir()
            first = prepare_manifest(config, first_dir, train, val)
            second = prepare_manifest(config, second_dir, train, val)
            self.assertEqual(first, second)
            self.assertEqual(first, prepare_manifest(config, first_dir, train, val))
            self.assertEqual(first["reference_policy"], REFERENCE_POLICY)
            self.assertEqual([len(first[s]) for s in ("train", "dev", "holdout")], [8, 4, 6])
            names = [{r["image"] for r in first[s]} for s in ("train", "dev", "holdout")]
            self.assertFalse(names[0] & names[1] or names[0] & names[2] or names[1] & names[2])
            self.assertNotIn(missing_train["image"], set.union(*names))
            self.assertNotIn(missing_val["image"], set.union(*names))
            refs = {r["image"]: r for s in ("dev", "holdout") for r in first[s]}
            self.assertEqual(refs[mixed["image"]], mixed)
            for split, missing in (("train", missing_train), ("validation", missing_val)):
                self.assertEqual(first["annotation_audit"][split]["excluded_examples"],
                                 [{"image": missing["image"], "answerable": missing["answerable"],
                                   "answer_type": missing["answer_type"], "reason": "no_usable_reference_answers"}])
            with self.assertRaisesRegex(ValueError, "No usable reference target"):
                chosen_answer(missing_train, 42, 0, True)
            legacy = dict(first)
            legacy.pop("reference_policy")
            write_json(first_dir / "split_manifest.json", legacy)
            with self.assertRaisesRegex(ValueError, "Reference policy changed"):
                prepare_manifest(config, first_dir, train, val)

    def test_official_annotations_with_empty_answers_prepare_unchanged_counts(self):
        archive = Path(__file__).resolve().parents[1] / ".deps/vizwiz_annotations_review.zip"
        if not archive.exists():
            self.skipTest("Official annotation cache absent; test never downloads it")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extract_named(archive, ["train.json", "val.json"], root)
            first_dir, second_dir = root / "one", root / "two"
            first_dir.mkdir(); second_dir.mkdir()
            config = TrainConfig()
            manifest = prepare_manifest(config, first_dir, root / "train.json", root / "val.json")
            self.assertEqual(manifest, prepare_manifest(config, second_dir, root / "train.json", root / "val.json"))
            self.assertEqual([len(manifest[s]) for s in ("train", "dev", "holdout")], [5000, 1000, 3319])
            self.assertEqual(manifest["annotation_audit"]["train"]["invalid_answer_slots"], 2)
            self.assertEqual(manifest["annotation_audit"]["validation"]["invalid_answer_slots"], 0)
            self.assertFalse(manifest["annotation_audit"]["train"]["excluded_examples"])
            self.assertEqual(manifest["train"], stratified_sample(read_json(root / "train.json"), 5000, 42))
            self.assertEqual(manifest["dev"], stratified_sample(read_json(root / "val.json"), 1000, 43))
            actual = next(r for r in annotation_rows(root / "train.json") if r["image"] == "VizWiz_train_00023642.jpg")
            self.assertEqual(actual["answers"][3]["answer"], "")
            self.assertEqual(actual["answerable"], 1)
            self.assertEqual(usable_answers(actual["answers"]).count("unanswerable"), 4)

    def test_interrupted_extraction_does_not_publish_partial_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'images.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('folder/image.jpg', b'complete bytes')
            def interrupted(source, output):
                output.write(b'partial')
                raise OSError('simulated disconnect')
            with patch('visionbridge_training.shutil.copyfileobj', side_effect=interrupted):
                with self.assertRaises(OSError):
                    extract_named(archive, ['image.jpg'], root / 'images')
            self.assertFalse((root / 'images/image.jpg').exists())
            self.assertTrue(list((root / 'images').glob('*.partial')))
            extract_named(archive, ['image.jpg'], root / 'images')
            self.assertEqual((root / 'images/image.jpg').read_bytes(), b'complete bytes')

    def test_truncated_download_does_not_publish_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = io.BytesIO(b'truncated')
            stream.headers = {'Content-Length': '100'}
            path = Path(tmp) / 'images.zip'
            with patch('visionbridge_training.urllib.request.urlopen', return_value=stream):
                with self.assertRaisesRegex(ValueError, 'Incomplete download'):
                    download_once('https://fixture.invalid/images.zip', path)
            self.assertFalse(path.exists())
            self.assertTrue(list(Path(tmp).glob('*.partial')))

    def test_answer_sampling_and_epoch_order_resume(self):
        item = row(0)
        item["answers"] = [{"answer": "cat"}] * 7 + [{"answer": "kitten"}] * 3
        self.assertEqual(chosen_answer(item, 42, 3, True), chosen_answer(item, 42, 3, True))
        self.assertEqual(chosen_answer(item, 42, 3, False), "cat")
        config = TrainConfig(batch_size=3)
        rows = [row(i) for i in range(10)]
        batches = batches_for_epoch(rows, config, 2)
        self.assertEqual(batches, batches_for_epoch(rows, config, 2))
        self.assertEqual(len(batches[-1]), 1)

    def test_best_selection_and_patience(self):
        config = TrainConfig(patience=2, min_delta=0.002)
        state = {"best_score": 0.5, "patience_score": 0.5, "bad_checks": 0}
        self.assertEqual(validation_decision(state, 0.501, config), (True, False))
        self.assertEqual(validation_decision(state, 0.49, config), (False, True))
        self.assertEqual(state["best_score"], 0.501)
        self.assertEqual(validation_decision(state, 0.51, config), (True, False))
        self.assertEqual(state["bad_checks"], 0)


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = {"source": "fixture", "source_sha256": "fixture", "contractions": {"dont": "don't"},
                     "manualMap": {"two": "2"}, "articles": ["a", "an", "the"], "punct": [",", "!"]}

    def test_agreement_values_and_normalization(self):
        scorer = OfficialScorer(self.rules)
        self.assertEqual(scorer.normalize("The two!"), "2")
        self.assertEqual(scorer.normalize("dont"), "don't")
        self.assertEqual(scorer.normalize("1,000"), "1000")
        for matches, expected in ((0, 0), (1, .3), (2, .6), (3, .9), (4, 1)):
            answers = [{"answer": "2"}] * matches + [{"answer": "cat"}] * (10 - matches)
            self.assertAlmostEqual(scorer.score("two", answers), expected)

    def test_summarize_requires_every_prediction(self):
        with self.assertRaises(ValueError):
            summarize([row(0)], [], OfficialScorer(self.rules))

    def test_invalid_slots_never_become_labels_or_change_ten_slot_denominator(self):
        scorer = OfficialScorer(self.rules)
        invalid = [{"answer": ""}, {"answer": " "}, {"answer": None}, {}, 3]
        answers = [{"answer": "cat"}] * 3 + [{"answer": "kitten"}] * 2 + invalid
        self.assertAlmostEqual(scorer.score("cat", answers), .9)
        self.assertAlmostEqual(scorer.score("kitten", answers), .6)
        for prediction in ("", " ", "None", "3", "unanswerable"):
            self.assertEqual(scorer.score(prediction, answers), 0)
        self.assertEqual(scorer.score("cat", invalid * 2), 0)
        self.assertEqual(scorer.identity["reference_policy"], REFERENCE_POLICY)
        self.assertEqual(answers[-5:], invalid)

    def test_extract_literals_without_executing_code(self):
        source = "raise RuntimeError('must not execute')\nclass VQAEval:\n    def __init__(self):\n"
        source += "".join(f"        self.{key} = {self.rules[key]!r}\n" for key in
                          ("contractions", "manualMap", "articles", "punct"))
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "api.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(API_MEMBER, source)
            scorer = extract_rules(archive, Path(tmp) / "rules.json")
            self.assertEqual(scorer.normalize("The two!"), "2")

    def test_parity_with_official_api_accuracy(self):
        archive = Path(__file__).resolve().parents[1] / ".deps/vizwiz_api.zip"
        if not archive.exists():
            self.skipTest("Official API cache absent; test never downloads it")
        with zipfile.ZipFile(archive) as bundle:
            tree = ast.parse(bundle.read(API_MEMBER).decode("utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VQAEval")
        keep = {"__init__", "processPunctuation", "processDigitArticle", "evaluate",
                "setAccuracy", "setEvalQA", "setEvalAnsType", "updateProgress"}
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in keep]
        method = next(n for n in cls.body if n.name == "evaluate")
        # Run the official accuracy block, leaving Java caption metrics out.
        cutoff = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Expr)
                      and isinstance(n.value, ast.Call) and n.value.args
                      and isinstance(n.value.args[0], ast.Constant)
                      and n.value.args[0].value == "computing caption metric")
        method.body = method.body[:cutoff]
        namespace = {"re": re, "sys": sys}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "official_accuracy", "exec"), namespace)
        rows = []
        predictions = []
        for i, (prediction, reference) in enumerate((('The two!', '2'), ('1,000', '1000'),
                ('dont', 'don\ufffdt'), ('unanswerable', 'unanswerable'), ('wrong', 'cat'))):
            item = row(i, "val")
            item["answers"] = [{"answer": reference}] * 3 + [{"answer": "other"}] * 7
            rows.append(item)
            predictions.append(prediction)
        item = row(10, "val")
        item["answers"] = [{"answer": "cat"}] * 3 + [{"answer": ""}] + [{"answer": "other"}] * 6
        rows.append(item)
        predictions.append("cat")
        gt = SimpleNamespace(getImgs=lambda: [r["image"] for r in rows], imgToQA={r["image"]: r for r in rows})
        res = SimpleNamespace(imgToQA={r["image"]: {"image": r["image"], "answer": p} for r, p in zip(rows, predictions)})
        oracle = namespace["VQAEval"](gt, res, n=8)
        with redirect_stdout(io.StringIO()):
            oracle.evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            scorer = extract_rules(archive, Path(tmp) / "rules.json")
            metrics, records = summarize(rows, predictions, scorer)
        self.assertAlmostEqual(metrics["vqa_score"] * 100, oracle.accuracy["overall"])
        for record in records:
            self.assertAlmostEqual(record["score"] * 100, oracle.evalQA[record["image"]])


class OptimizerTests(unittest.TestCase):
    def test_amp_skipped_update_does_not_advance_scheduler(self):
        import torch
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        class OverflowScaler:
            scale_value = 2
            def scale(self, loss): return loss
            def unscale_(self, optimizer): pass
            def get_scale(self): return self.scale_value
            def step(self, optimizer): pass  # Simulate an overflow-skipped step.
            def update(self): self.scale_value = 1
        class LossModel:
            def __call__(self, labels):
                return SimpleNamespace(loss=model(torch.ones(1, 1)).square().mean())
        before = [p.detach().clone() for p in model.parameters()]
        epoch = scheduler.last_epoch
        _, successful = accumulation_update(LossModel(), optimizer, scheduler, OverflowScaler(),
                           [{"labels": torch.tensor([[0, 1]])}], list(model.parameters()), "cpu")
        self.assertFalse(successful)
        self.assertEqual(scheduler.last_epoch, epoch)
        for old, current in zip(before, model.parameters()):
            torch.testing.assert_close(old, current)

    def test_orchestration_exports_best_and_completed_resume_only_exports(self):
        import torch
        import visionbridge_training as training
        from contextlib import ExitStack
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.3))
                self.vision_model = torch.nn.Identity()
                self.text_encoder = torch.nn.Identity()
            def forward(self, labels):
                return SimpleNamespace(loss=(self.weight - .1).square())
            def save_pretrained(self, folder, **kwargs):
                (Path(folder) / 'fixture_weight.txt').write_text(str(self.weight.item()))
        class Processor:
            def save_pretrained(self, folder):
                (Path(folder) / 'fixture_processor.txt').write_text('processor')
        real_scaler = torch.amp.GradScaler
        real_update = accumulation_update
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            original = root / 'original'
            original.mkdir()
            for name in ('preprocessor_config.json', 'tokenizer_config.json', 'vocab.txt'):
                (original / name).write_text('{}')
            write_json(original / 'config.json', {'architectures': ['BlipForQuestionAnswering']})
            (original / 'model.safetensors').write_bytes(b'unchanged fixture input')
            before_hash = sha256(original / 'model.safetensors')
            config = TrainConfig(init_checkpoint=str(original), train_size=4, dev_size=2,
                batch_size=2, accumulation_steps=1, epochs=4, patience=1)
            run = create_run(config, root / 'run')
            write_json(run / 'split_manifest.json', {'train': [row(i) for i in range(4)],
                'dev': [row(i, 'dev') for i in range(2)], 'holdout': [row(0, 'holdout')]})
            scores = iter([.5, .6, .55])
            def evaluate(*args, **kwargs):
                return {'vqa_score': next(scores), 'validation_loss': .1}, []
            stack.enter_context(patch('torch.cuda.is_available', return_value=True))
            stack.enter_context(patch('torch.cuda.manual_seed_all'))
            stack.enter_context(patch('torch.cuda.get_rng_state_all', return_value=[]))
            stack.enter_context(patch('torch.cuda.set_rng_state_all'))
            stack.enter_context(patch('torch.cuda.empty_cache'))
            stack.enter_context(patch('torch.amp.GradScaler', side_effect=lambda *a, **k: real_scaler('cuda', enabled=False)))
            stack.enter_context(patch.object(training, 'load_model', side_effect=lambda *a: (Processor(), Tiny())))
            stack.enter_context(patch.object(training, 'configure_trainable', side_effect=lambda model, _: list(model.parameters())))
            stack.enter_context(patch.object(training, 'make_batch', return_value={'labels': torch.tensor([[0, 1]])}))
            stack.enter_context(patch.object(training, 'evaluate', side_effect=evaluate))
            stack.enter_context(patch.object(training, 'accumulation_update',
                side_effect=lambda model, opt, sched, scaler, encoded, params, device:
                    real_update(model, opt, sched, scaler, encoded, params, 'cpu')))
            scorer = SimpleNamespace(identity={'fixture': True})
            with redirect_stdout(io.StringIO()):
                export = training.train_run(config, run, root, scorer)
            best = run / read_json(run / 'best.json')['checkpoint']
            latest = run / read_json(run / 'latest.json')['checkpoint']
            self.assertNotEqual(best, latest)
            self.assertEqual(read_json(latest / 'state.json')['stopped'], 'early_stopping')
            from safetensors.torch import load_file
            expected = load_file(str(best / 'trainable.safetensors'))['weight'].item()
            self.assertAlmostEqual(float((export / 'fixture_weight.txt').read_text()), expected)
            self.assertEqual(sha256(original / 'model.safetensors'), before_hash)
            summary = read_json(export.parent / 'training_summary.json')
            self.assertGreaterEqual(summary['training_elapsed_seconds'], 0)
            self.assertTrue((run / 'history.csv').is_file())
            self.assertTrue((run / 'plots/training_validation.png').is_file())
            self.assertTrue((run / 'training_progress.json').is_file())
            # A terminal resume must not execute another optimizer update.
            with patch.object(training, 'accumulation_update', side_effect=AssertionError('must not train')):
                with redirect_stdout(io.StringIO()):
                    second_export = training.train_run(config, run, root, scorer, resume=True)
            self.assertNotEqual(export, second_export)
            self.assertTrue(export.is_dir())

    def test_short_accumulation_group_matches_combined_tokens(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.3))
            def forward(self, labels):
                targets = labels[:, 1:]
                targets = targets[targets != -100].float() / 10
                return SimpleNamespace(loss=((self.weight - targets) ** 2).mean())
        one, many = Tiny(), Tiny()
        batches = [{"labels": torch.tensor([[0, 1]])}, {"labels": torch.tensor([[0, 2, 3, 4]])}]
        def update(model, encoded):
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            return accumulation_update(model, optimizer, scheduler, scaler, encoded, list(model.parameters()), "cpu")
        result = update(many, batches)
        update(one, [{"labels": torch.tensor([[0, 1, 2, 3, 4]])}])
        self.assertTrue(result[1])
        torch.testing.assert_close(one.weight, many.weight)

    def test_snapshot_roundtrip_includes_optimizer_and_rng(self):
        import torch
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        model(torch.ones(1, 2)).square().mean().backward()
        optimizer.step()
        scheduler.step()
        expected = {name: p.detach().clone() for name, p in model.named_parameters()}
        with tempfile.TemporaryDirectory() as tmp:
            folder = save_snapshot(Path(tmp), model, optimizer, scheduler, scaler,
                                   {"updates": 1, "best_score": .5}, is_best=True)
            payload = torch.load(folder / "training_state.pt", weights_only=True)
            self.assertTrue(payload["optimizer"]["state"])
            self.assertIn("python_rng", payload)
            self.assertEqual(read_json(folder / "state.json")["best_checkpoint"], str(folder.relative_to(tmp)))
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(10)
            load_snapshot_weights(model, folder)
            for name, p in model.named_parameters():
                torch.testing.assert_close(p, expected[name])
            (folder / "state.json").write_text("corrupted")
            with self.assertRaises(ValueError):
                load_snapshot_weights(model, folder)

    def test_notebook_syntax_and_disabled_training(self):
        root = Path(__file__).resolve().parents[1]
        notebook = read_json(root / "mai_model_train_improved.ipynb")
        self.assertEqual(notebook["nbformat"], 4)
        code = ""
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, "notebook_cell", "exec")
                self.assertEqual(cell["outputs"], [])
                code += source
        self.assertIn("RUN_TRAINING = False", code)
        self.assertIn("RUN_FINAL_EVALUATION = False", code)
        self.assertIn("PREPARE_DATA = False", code)
        self.assertIn("code_hashes.json", code)
        self.assertIn("matplotlib>=3.8,<4", code)


class ComparisonTests(unittest.TestCase):
    def prepare_run(self, root, stopped='max_epochs'):
        original = root / 'original'
        original.mkdir()
        for name in ('preprocessor_config.json', 'tokenizer_config.json', 'vocab.txt'):
            (original / name).write_text('{}')
        write_json(original / 'config.json', {'architectures': ['BlipForQuestionAnswering']})
        (original / 'model.safetensors').write_bytes(b'original fixture')
        config = TrainConfig(init_checkpoint=str(original))
        run = create_run(config, root / 'run')
        rows = [row(i, 'holdout') for i in range(3)]
        write_json(run / 'split_manifest.json', {'holdout': rows})
        scorer = OfficialScorer({'source': 'fixture', 'source_sha256': 'fixture',
                                'contractions': {}, 'manualMap': {}, 'articles': [], 'punct': []})
        write_json(run / 'experiment_identity.json', {'manifest_sha256': sha256(run / 'split_manifest.json'),
                                                     'scorer': scorer.identity})
        export = run / 'exports/export-fixture/blip_vizwiz_v2'
        export.parent.mkdir(parents=True)
        import shutil
        shutil.copytree(original, export)
        write_json(export.parent / 'training_summary.json', {'stop_reason': stopped})
        write_json(run / 'export.json', {'model': str(export.relative_to(run))})
        return config, run, scorer, rows

    def test_paused_run_cannot_consume_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, run, scorer, _ = self.prepare_run(Path(tmp), 'time_budget')
            with patch('visionbridge_training.load_processor', side_effect=AssertionError('must not load')):
                with self.assertRaisesRegex(RuntimeError, 'Resume training'):
                    final_compare(config, run, run, scorer, device='cpu')

    def test_three_models_share_inputs_and_resume_saved_evaluation(self):
        import visionbridge_training as training
        with tempfile.TemporaryDirectory() as tmp:
            config, run, scorer, rows = self.prepare_run(Path(tmp))
            common = SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda: {'cat': 1}))
            evaluated = []
            interrupted = [True]
            def load(source, device):
                return common, SimpleNamespace(source=source, config=SimpleNamespace(_commit_hash='fixture'))
            def evaluate(model, processor, examples, image_root, actual_config, actual_scorer, device, with_loss):
                self.assertIs(processor, common)
                self.assertEqual(examples, rows)
                self.assertIs(actual_config, config)
                self.assertIs(actual_scorer, scorer)
                self.assertTrue(with_loss)
                evaluated.append(model.source)
                if model.source == config.init_checkpoint and interrupted[0]:
                    interrupted[0] = False
                    raise RuntimeError('simulated comparison disconnect')
                metrics, records = summarize(rows, ['cat'] * len(rows), scorer)
                metrics['validation_loss'] = .1
                return metrics, records
            with patch.object(training, 'load_processor', return_value=common), \
                 patch.object(training, 'load_model', side_effect=load), \
                 patch.object(training, 'evaluate', side_effect=evaluate):
                with self.assertRaisesRegex(RuntimeError, 'disconnect'):
                    final_compare(config, run, run, scorer, device='cpu')
                self.assertFalse((run / 'final_comparison.json').exists())
                results = final_compare(config, run, run, scorer, device='cpu')
            self.assertEqual(evaluated.count(config.pretrained_model), 1)
            self.assertEqual(set(results), {'pretrained_blip', 'existing_vizwiz', 'improved_blip'})
            self.assertTrue(all(m['examples'] == 3 for m in results.values()))
            folder = run / read_json(run / 'final_comparison.json')['folder']
            import csv
            with (folder / 'sample_comparison.csv').open(newline='', encoding='utf-8') as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 3)
            self.assertEqual(json.loads(records[0]['answers']), rows[0]['answers'])
            for label in results:self.assertEqual(records[0][label + '_prediction'], 'cat')
            self.assertTrue((folder / 'model_comparison.png').is_file())
            self.assertTrue((folder / 'model_comparison.pdf').is_file())
            self.assertTrue((folder / 'model_comparison.csv').is_file())
            with self.assertRaises(FileExistsError):
                final_compare(config, run, run, scorer, device='cpu')


if __name__ == "__main__":
    unittest.main()
