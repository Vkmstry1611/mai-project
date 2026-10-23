"""Explicit, resumable BLIP fine-tuning for Colab. Imports have no side effects."""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import urllib.request
import uuid
import zipfile
from evaluate_vqa import (OfficialScorer, extract_rules, summarize, API_URL,
                         REFERENCE_POLICY, usable_answers)


@dataclass(frozen=True)
class TrainConfig:
    init_checkpoint: str = "models/blip_vizwiz"
    pretrained_model: str = "Salesforce/blip-vqa-base"
    seed: int = 42
    train_size: int = 5000
    dev_size: int = 1000
    batch_size: int = 4
    accumulation_steps: int = 4
    epochs: int = 5
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    patience: int = 2
    min_delta: float = 0.002
    save_every_updates: int = 500
    train_minutes: float = 120
    question_tokens: int = 128
    answer_tokens: int = 32
    max_new_tokens: int = 20
    decoder_layers: int = 2

    def validate(self):
        names = ("train_size", "dev_size", "batch_size", "accumulation_steps", "epochs",
                 "patience", "save_every_updates", "question_tokens", "answer_tokens",
                 "max_new_tokens", "decoder_layers", "learning_rate", "train_minutes")
        if any(getattr(self, name) <= 0 for name in names):
            raise ValueError("Sizes, epochs, learning rate and time budget must be positive")
        if not 0 <= self.warmup_fraction < 1 or self.min_delta < 0 or self.weight_decay < 0:
            raise ValueError("Invalid warmup, early-stopping delta or weight decay")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Atomic updates to run-owned metadata, never the input model."""
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
    temp.replace(path)


def guard_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output and input checkpoint must be separate directories")


def verify_checkpoint(folder):
    folder = Path(folder)
    required = ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer_config.json", "vocab.txt")
    for name in required:
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {folder / name}")
    with (folder / "model.safetensors").open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Weights are an LFS pointer; run git lfs pull first")
    if read_json(folder / "config.json").get("architectures") != ["BlipForQuestionAnswering"]:
        raise ValueError("Expected BlipForQuestionAnswering")
    return {name: sha256(folder / name) for name in required}


def create_run(config, run_dir):
    config.validate()
    guard_output(config.init_checkpoint, run_dir)
    identity = verify_checkpoint(config.init_checkpoint)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", asdict(config))
    write_json(run_dir / "input_checkpoint_hashes.json", identity)
    return run_dir


def check_run(config, run_dir):
    config.validate()
    guard_output(config.init_checkpoint, run_dir)
    if read_json(Path(run_dir) / "config.json") != asdict(config):
        raise ValueError("Settings changed; create a new experiment instead")
    if verify_checkpoint(config.init_checkpoint) != read_json(Path(run_dir) / "input_checkpoint_hashes.json"):
        raise ValueError("Starting checkpoint changed since run creation")


def stratified_sample(rows, count, seed):
    if not 0 < count <= len(rows):
        raise ValueError(f"Requested {count} rows from {len(rows)}")
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["answerable"]), row["answer_type"])].append(row)
    keys = sorted(groups)
    quotas = {key: count * len(groups[key]) / len(rows) for key in keys}
    allocation = {key: int(quotas[key]) for key in keys}
    remainder = count - sum(allocation.values())
    for key in sorted(keys, key=lambda k: (-(quotas[k] - allocation[k]), k))[:remainder]:
        allocation[key] += 1
    rng, selected = random.Random(seed), []
    for key in keys:
        selected.extend(rng.sample(groups[key], allocation[key]))
    rng.shuffle(selected)
    return selected


def annotation_rows(path, *, return_audit=False):
    """Keep raw ten-slot references; exclude no-target rows before any split.

    Bad individual answers are not repaired or deduplicated. Structural errors
    (including a non-ten-slot list) still fail rather than change evaluation.
    """
    rows, names = read_json(path), set()
    eligible, affected, excluded = [], [], []
    for row in rows:
        name = row["image"]
        if Path(name).name != name or name in names:
            raise ValueError(f"Unsafe or duplicate image name: {name}")
        names.add(name)
        if not isinstance(row["answers"], list) or len(row["answers"]) != 10 or row["answerable"] not in (0, 1):
            raise ValueError(f"Expected ten answers and binary answerability: {name}")
        valid = usable_answers(row["answers"])
        if len(valid) != 10:
            affected.append({"image": name, "answerable": row["answerable"],
                             "answer_type": row["answer_type"],
                             "invalid_answer_slots": 10 - len(valid)})
        if not valid:
            excluded.append({"image": name, "answerable": row["answerable"],
                             "answer_type": row["answer_type"], "reason": "no_usable_reference_answers"})
        else:
            eligible.append(row)
    audit = {"source_rows": len(rows), "eligible_rows": len(eligible),
             "invalid_answer_slots": sum(r["invalid_answer_slots"] for r in affected),
             "affected_examples": affected, "excluded_examples": excluded}
    return (eligible, audit) if return_audit else eligible


def prepare_manifest(config, run_dir, train_json, val_json):
    path = Path(run_dir) / "split_manifest.json"
    hashes = {"train": sha256(train_json), "validation": sha256(val_json)}
    if path.exists():
        manifest = read_json(path)
        if manifest["annotation_sha256"] != hashes:
            raise ValueError("Annotation files changed; refusing to change the splits")
        if manifest.get("reference_policy") != REFERENCE_POLICY:
            raise ValueError("Reference policy changed; preserve this run and create a new run")
        return manifest
    train, train_audit = annotation_rows(train_json, return_audit=True)
    val, val_audit = annotation_rows(val_json, return_audit=True)
    if config.dev_size >= len(val):
        raise ValueError("Keep some validation images for final evaluation")
    if {r["image"] for r in train} & {r["image"] for r in val}:
        raise ValueError("Train/validation overlap")
    dev = stratified_sample(val, config.dev_size, config.seed + 1)
    dev_names = {r["image"] for r in dev}
    manifest = {"dataset": "official VizWiz-VQA", "seed": config.seed,
                "reference_policy": REFERENCE_POLICY,
                "annotation_audit": {"train": train_audit, "validation": val_audit},
                "annotation_sha256": hashes,
                "sampling": "proportional stratification by answerable and answer_type",
                "train": stratified_sample(train, config.train_size, config.seed),
                "dev": dev, "holdout": [r for r in val if r["image"] not in dev_names]}
    write_json(path, manifest)
    return manifest


DATA_BASE = "https://vizwiz.cs.colorado.edu/VizWiz_final/"


def download_once(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.suffix == '.zip' and not zipfile.is_zipfile(path):
            raise ValueError(f'Existing archive is incomplete; preserve it and choose a new data directory: {path}')
        return path
    partial = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as source, partial.open("xb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        expected = source.headers.get('Content-Length')
    if expected is not None and partial.stat().st_size != int(expected):
        raise ValueError(f'Incomplete download retained at {partial}; rerun preparation')
    if path.suffix == '.zip' and not zipfile.is_zipfile(partial):
        raise ValueError(f'Invalid archive retained at {partial}; rerun preparation')
    if path.exists():
        raise FileExistsError(f"Concurrent download appeared; kept {partial}")
    partial.rename(path)
    return path


def extract_named(archive, names, destination):
    import zlib
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        by_name = defaultdict(list)
        for item in bundle.infolist():
            if not item.is_dir():
                by_name[Path(item.filename).name].append(item)
        for name in names:
            if Path(name).name != name or len(by_name[name]) != 1:
                raise ValueError(f"Unsafe, missing or ambiguous member: {name}")
            target = destination / name
            if not target.exists():
                partial = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
                with bundle.open(by_name[name][0]) as source, partial.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if target.exists():
                    raise FileExistsError(f'Concurrent extraction appeared; kept {partial}')
                partial.rename(target)
            else:
                crc = 0
                with target.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        crc = zlib.crc32(chunk, crc)
                if target.stat().st_size != by_name[name][0].file_size or crc != by_name[name][0].CRC:
                    raise ValueError(f"Existing file differs from archive: {target}")


def prepare_official_data(config, run_dir, data_root):
    """Explicit Colab preparation; full archives need several GB of disk/bandwidth."""
    check_run(config, run_dir)
    data_root = Path(data_root)
    guard_output(config.init_checkpoint, data_root)
    archive = download_once(DATA_BASE + "vqa_data/Annotations.zip", data_root / "Annotations.zip")
    extract_named(archive, ["train.json", "val.json"], data_root / "annotations")
    manifest = prepare_manifest(config, run_dir, data_root / "annotations/train.json", data_root / "annotations/val.json")
    for split, rows in (("train", manifest["train"]), ("val", manifest["dev"] + manifest["holdout"])):
        archive = download_once(DATA_BASE + f"images/{split}.zip", data_root / f"{split}.zip")
        extract_named(archive, [row["image"] for row in rows], data_root / "images")
    api = download_once(API_URL, data_root / "API.zip")
    extract_rules(api, Path(run_dir) / "official_normalization.json")
    return manifest


def load_model(source, device):
    from transformers import BlipProcessor, BlipForQuestionAnswering
    local = Path(source).is_dir()
    processor = BlipProcessor.from_pretrained(source, local_files_only=local, use_fast=False)
    model, info = BlipForQuestionAnswering.from_pretrained(
        source, local_files_only=local, use_safetensors=True, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Strict checkpoint loading failed: {info}")
    return processor, model.to(device)


def load_processor(source):
    from transformers import BlipProcessor
    return BlipProcessor.from_pretrained(source, local_files_only=Path(source).is_dir(), use_fast=False)


def write_csv(path, records, fields):
    """Replace only run-owned summaries, retaining all reference answers as JSON."""
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(record[key], ensure_ascii=False) if isinstance(record.get(key), (list, dict))
                             else record.get(key) for key in fields})
    temp.replace(path)


def save_training_plots(run_dir, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(run_dir) / 'plots'
    folder.mkdir(exist_ok=True)
    train = [r for r in history if r['stage'] == 'train']
    dev = [r for r in history if r['stage'] in ('dev', 'starting_checkpoint')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot([r['updates'] for r in train], [r['loss'] for r in train])
    axes[0].set(title='Training token loss', ylabel='Loss')
    axes[1].plot([r['updates'] for r in dev], [r['validation_loss'] for r in dev], marker='o')
    axes[1].set(title='Development token loss', ylabel='Loss')
    axes[2].plot([r['updates'] for r in dev], [100*r['vqa_score'] for r in dev], marker='o')
    axes[2].set(title='Development VQA agreement', ylabel='VQA score (%)', ylim=(0, 100))
    for axis in axes:
        axis.set_xlabel('Successful optimizer updates')
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / 'training_validation.png', dpi=160)
    fig.savefig(folder / 'training_validation.pdf')
    plt.close(fig)
    write_csv(Path(run_dir) / 'history.csv', history,
              ['stage', 'epoch', 'updates', 'loss', 'validation_loss', 'vqa_score', 'learning_rate',
               'successful', 'training_elapsed_seconds'])


def configure_trainable(model, layer_count):
    if not 0 < layer_count <= len(model.text_decoder.bert.encoder.layer):
        raise ValueError("Invalid decoder layer count")
    model.requires_grad_(False)
    for layer in model.text_decoder.bert.encoder.layer[-layer_count:]:
        layer.requires_grad_(True)
    model.text_decoder.cls.requires_grad_(True)
    # BLIP's LM output weight is tied to decoder token embeddings; those also train.
    return [p for p in model.parameters() if p.requires_grad]


def chosen_answer(row, seed, epoch, sample):
    answers = [answer.strip() for answer in usable_answers(row["answers"])]
    if not answers:
        raise ValueError(f"No usable reference target: {row['image']}; rebuild a fresh run using the documented exclusion policy")
    if sample:
        return random.Random(f"{seed}:{epoch}:{row['image']}").choice(answers)
    return Counter(answers).most_common(1)[0][0]


def make_batch(rows, image_root, processor, model, config, device, epoch=0, targets=False, sample=False):
    from PIL import Image, ImageOps
    images = []
    for row in rows:
        with Image.open(Path(image_root) / row["image"]) as im:
            images.append(ImageOps.exif_transpose(im).convert("RGB"))
    batch = processor(images=images, text=[r["question"] for r in rows], padding=True,
                      truncation=True, max_length=config.question_tokens, return_tensors="pt")
    if targets:
        encoded = processor.tokenizer([chosen_answer(r, config.seed, epoch, sample) for r in rows],
                    padding=True, truncation=True, max_length=config.answer_tokens, return_tensors="pt")
        ids = encoded.input_ids.clone()
        ids[:, 0] = model.config.text_config.bos_token_id
        labels = ids.clone()
        labels[encoded.attention_mask == 0] = -100
        batch.update(decoder_input_ids=ids, decoder_attention_mask=encoded.attention_mask, labels=labels)
    return {key: value.to(device) for key, value in batch.items()}


def amp_context(device):
    import torch
    return torch.autocast("cuda", dtype=torch.float16) if str(device).startswith("cuda") else nullcontext()


def evaluate(model, processor, rows, image_root, config, scorer, device, with_loss=True):
    import torch
    predictions, total_loss, tokens = [], 0.0, 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), config.batch_size):
            batch = make_batch(rows[start:start + config.batch_size], image_root, processor,
                               model, config, device, targets=with_loss)
            with amp_context(device):
                if with_loss:
                    count = int((batch["labels"][:, 1:] != -100).sum())
                    loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite validation loss")
                    total_loss += loss.item() * count
                    tokens += count
                inputs = {k: v for k, v in batch.items() if k not in
                          {"labels", "decoder_input_ids", "decoder_attention_mask"}}
                generated = model.generate(**inputs, max_new_tokens=config.max_new_tokens, num_beams=1, do_sample=False)
            predictions.extend(processor.batch_decode(generated, skip_special_tokens=True))
    metrics, records = summarize(rows, predictions, scorer)
    metrics["validation_loss"] = total_loss / tokens if tokens else None
    return metrics, records


def batches_for_epoch(rows, config, epoch):
    import torch
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(config.seed + epoch)).tolist()
    return [[rows[i] for i in order[start:start + config.batch_size]] for start in range(0, len(rows), config.batch_size)]


def save_snapshot(run_dir, model, optimizer, scheduler, scaler, state, is_best=False):
    """Trainable-only delta plus full optimizer/RNG state, in a new directory."""
    import torch
    from safetensors.torch import save_file
    folder = Path(run_dir) / "checkpoints" / f"step-{state['updates']:06d}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    if is_best:
        state["best_checkpoint"] = str(folder.relative_to(run_dir))
    values = {name: p.detach().cpu().contiguous().clone() for name, p in model.named_parameters() if p.requires_grad}
    save_file(values, str(folder / "trainable.safetensors"))
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python_rng": random.getstate()}, folder / "training_state.pt")
    write_json(folder / "state.json", state)
    write_json(folder / "checksums.json", {name: sha256(folder / name) for name in
               ("trainable.safetensors", "training_state.pt", "state.json")})
    if is_best:
        write_json(Path(run_dir) / "best.json", {"checkpoint": state["best_checkpoint"], "vqa_score": state["best_score"]})
    if 'history' in state:
        write_json(Path(run_dir) / 'history.json', state['history'])
        write_json(Path(run_dir) / 'training_progress.json', {key: state.get(key) for key in
                   ('epoch', 'next_batch', 'updates', 'stopped', 'training_elapsed_seconds')})
    write_json(Path(run_dir) / "latest.json", {"checkpoint": str(folder.relative_to(run_dir))})
    return folder


def load_snapshot_weights(model, folder):
    import torch
    from safetensors.torch import load_file
    folder = Path(folder)
    for name, expected in read_json(folder / "checksums.json").items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Incomplete or changed snapshot: {folder / name}")
    values, params = load_file(str(folder / "trainable.safetensors")), dict(model.named_parameters())
    if set(values) != {name for name, p in params.items() if p.requires_grad}:
        raise ValueError("Snapshot trainable architecture changed")
    with torch.no_grad():
        for name, value in values.items():
            params[name].copy_(value.to(params[name].device))


def accumulation_update(model, optimizer, scheduler, scaler, encoded, params, device):
    """Token-weighted accumulation, including short groups and skipped AMP steps."""
    import torch
    counts = [int((b["labels"][:, 1:] != -100).sum()) for b in encoded]
    if sum(counts) == 0:
        raise ValueError("No answer tokens in accumulation group")
    optimizer.zero_grad(set_to_none=True)
    group_loss = 0.0
    for batch, count in zip(encoded, counts):
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp_context(device):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss; latest saved snapshot remains available")
        scaler.scale(loss * count / sum(counts)).backward()
        group_loss += loss.item() * count / sum(counts)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    successful = scaler.get_scale() >= old_scale
    if successful:
        scheduler.step()
    return group_loss, successful


def validation_decision(state, score, config):
    """Select any better model, but reset patience only for meaningful gains."""
    improved = score > state["best_score"]
    meaningful = score > state["patience_score"] + config.min_delta
    state["bad_checks"] = 0 if meaningful else state["bad_checks"] + 1
    if meaningful:
        state["patience_score"] = score
    if improved:
        state["best_score"] = score
    return improved, state["bad_checks"] >= config.patience


def train_run(config, run_dir, image_root, scorer, resume=False, device="cuda"):
    import torch
    from transformers import get_linear_schedule_with_warmup
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Full training requires a Colab GPU")
    check_run(config, run_dir)
    run_dir = Path(run_dir)
    if (run_dir / "latest.json").exists() != resume:
        raise ValueError("Use resume=True for an existing run, or create a new run")
    manifest = read_json(run_dir / "split_manifest.json")
    identity = {"manifest_sha256": sha256(run_dir / "split_manifest.json"), "scorer": scorer.identity}
    if (run_dir / "experiment_identity.json").exists():
        if read_json(run_dir / "experiment_identity.json") != identity:
            raise ValueError("Split manifest or scoring rules changed")
    else:
        write_json(run_dir / "experiment_identity.json", identity)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    processor, model = load_model(config.init_checkpoint, device)
    params = configure_trainable(model, config.decoder_layers)
    print(f"Trainable parameters: {sum(p.numel() for p in params):,}", flush=True)
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    per_epoch = math.ceil(math.ceil(len(manifest["train"]) / config.batch_size) / config.accumulation_steps)
    steps = per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * config.warmup_fraction), steps)
    scaler = torch.amp.GradScaler("cuda")
    state = {"epoch": 0, "next_batch": 0, "updates": 0, "attempts": 0,
             "best_score": None, "patience_score": None, "bad_checks": 0,
             "best_checkpoint": None, "stopped": None, "history": [], 'training_elapsed_seconds': 0.0}
    if resume:
        folder = run_dir / read_json(run_dir / "latest.json")["checkpoint"]
        load_snapshot_weights(model, folder)
        state = read_json(folder / "state.json")
        payload = torch.load(folder / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        torch.set_rng_state(payload["torch_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        # A completed run may still need export after a disconnect. Do not retrain it.
        if state["stopped"] == "time_budget":
            state["stopped"] = None
        write_json(run_dir / "best.json", {"checkpoint": state["best_checkpoint"], "vqa_score": state["best_score"]})
    else:
        metrics, predictions = evaluate(model, processor, manifest["dev"], image_root, config, scorer, device)
        state["best_score"] = state["patience_score"] = metrics["vqa_score"]
        state["history"].append({"stage": "starting_checkpoint", "updates": 0, **metrics})
        write_json(run_dir / "starting_dev_predictions.json", predictions)
        write_json(run_dir / "best.json", {"checkpoint": None, "vqa_score": state["best_score"]})
        save_snapshot(run_dir, model, optimizer, scheduler, scaler, state)
    invocation_start = time.monotonic()
    previous_elapsed = state.get('training_elapsed_seconds', 0.0)
    def stamp_time():
        state['training_elapsed_seconds'] = previous_elapsed + time.monotonic() - invocation_start
        return state['training_elapsed_seconds']
    deadline = invocation_start + config.train_minutes * 60
    while state["epoch"] < config.epochs and state["stopped"] is None:
        epoch = state["epoch"]
        batches = batches_for_epoch(manifest["train"], config, epoch)
        model.train()
        model.vision_model.eval()
        model.text_encoder.eval()
        while state["next_batch"] < len(batches):
            if time.monotonic() >= deadline:
                state["stopped"] = "time_budget"
                break
            start = state["next_batch"]
            group = batches[start:start + config.accumulation_steps]
            encoded = [make_batch(rows, image_root, processor, model, config, "cpu", epoch=epoch,
                                  targets=True, sample=True) for rows in group]
            group_loss, successful = accumulation_update(model, optimizer, scheduler, scaler, encoded, params, device)
            if successful:
                state["updates"] += 1
            state["attempts"] += 1
            state["next_batch"] += len(group)
            state["history"].append({"stage": "train", "epoch": epoch + 1, "updates": state["updates"],
                                     "loss": group_loss, "learning_rate": scheduler.get_last_lr()[0], "successful": successful,
                                     'training_elapsed_seconds': stamp_time()})
            if successful and state["updates"] % config.save_every_updates == 0:
                save_snapshot(run_dir, model, optimizer, scheduler, scaler, state)
            if state["attempts"] % 25 == 0:
                print(f"Epoch {epoch + 1}: update {state['updates']}, loss {group_loss:.3f}", flush=True)
        if state["stopped"]:
            stamp_time()
            save_snapshot(run_dir, model, optimizer, scheduler, scaler, state)
            break
        metrics, predictions = evaluate(model, processor, manifest["dev"], image_root, config, scorer, device)
        score = metrics["vqa_score"]
        state["history"].append({"stage": "dev", "epoch": epoch + 1, "updates": state["updates"],
                                 'training_elapsed_seconds': stamp_time(), **metrics})
        improved, stop_early = validation_decision(state, score, config)
        state["epoch"] += 1
        state["next_batch"] = 0
        if stop_early:
            state["stopped"] = "early_stopping"
        elif state["epoch"] >= config.epochs:
            state["stopped"] = "max_epochs"
        folder = save_snapshot(run_dir, model, optimizer, scheduler, scaler, state, is_best=improved)
        write_json(folder / 'dev_predictions.json', predictions)
        write_json(run_dir / "history.json", state["history"])
        save_training_plots(run_dir, state['history'])
        print(f"Epoch {epoch + 1}: dev VQA {score:.3%}, loss {metrics['validation_loss']:.3f}", flush=True)
        if state["stopped"]:
            break
    del model, params, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    processor, model = load_model(config.init_checkpoint, device)
    configure_trainable(model, config.decoder_layers)
    best = read_json(run_dir / "best.json")
    if best["checkpoint"]:
        load_snapshot_weights(model, run_dir / best["checkpoint"])
    export = run_dir / "exports" / f"export-{state['updates']:06d}-{uuid.uuid4().hex[:8]}" / "blip_vizwiz_v2"
    export.mkdir(parents=True, exist_ok=False)
    model.eval().save_pretrained(export, safe_serialization=True)
    processor.save_pretrained(export)
    write_json(export.parent / "training_summary.json", {"updates": state["updates"], "completed_epochs": state["epoch"],
               "stop_reason": state["stopped"], "best": best, "selected_starting_checkpoint": best["checkpoint"] is None,
               'training_elapsed_seconds': state['training_elapsed_seconds'],
               'time_definition': 'Saved cumulative training-stage wall time; includes validation/checkpoint IO, excludes setup/initial evaluation/export'})
    write_json(run_dir / "export.json", {"model": str(export.relative_to(run_dir))})
    write_json(run_dir / "history.json", state["history"])
    save_training_plots(run_dir, state['history'])
    print(f"Best model exported to {export}; input checkpoint preserved.", flush=True)
    return export


def final_compare(config, run_dir, image_root, scorer, device="cuda"):
    import torch
    check_run(config, run_dir)
    run_dir = Path(run_dir)
    if (run_dir / "final_comparison.json").exists():
        raise FileExistsError("Holdout comparison already exists; do not repeatedly tune on it")
    identity = read_json(run_dir / "experiment_identity.json")
    if identity != {"manifest_sha256": sha256(run_dir / "split_manifest.json"), "scorer": scorer.identity}:
        raise ValueError("Evaluation inputs differ from training")
    rows = read_json(run_dir / "split_manifest.json")["holdout"]
    export = run_dir / read_json(run_dir / "export.json")["model"]
    if read_json(export.parent / 'training_summary.json')['stop_reason'] not in ('early_stopping', 'max_epochs'):
        raise RuntimeError('Model selection is paused. Resume training before final holdout evaluation.')
    common_processor = load_processor(config.init_checkpoint)
    protocol = {**identity, 'export_sha256': verify_checkpoint(export),
                'processor_source': config.init_checkpoint, 'batch_size': config.batch_size,
                'question_tokens': config.question_tokens, 'max_new_tokens': config.max_new_tokens,
                'num_beams': 1, 'do_sample': False, 'with_loss': True}
    pending = run_dir / 'comparison_pending.json'
    if pending.exists():
        plan = read_json(pending)
        if plan['protocol'] != protocol:
            raise ValueError('Cannot resume comparison with changed weights or evaluation settings')
        folder = run_dir / plan['folder']
    else:
        folder = run_dir / f"comparison-{uuid.uuid4().hex[:8]}"
        folder.mkdir(exist_ok=False)
        write_json(pending, {'folder': str(folder.relative_to(run_dir)), 'protocol': protocol})
    results = {}
    for label, source in (("pretrained_blip", config.pretrained_model),
                          ("existing_vizwiz", config.init_checkpoint), ("improved_blip", str(export))):
        prediction_path, metric_path = folder / f'{label}_predictions.json', folder / f'{label}_metrics.json'
        if metric_path.exists():
            cached = read_json(metric_path)
            if cached['predictions_sha256'] != sha256(prediction_path):
                raise ValueError('Saved comparison predictions changed')
            results[label] = cached['metrics']
            continue
        processor, model = load_model(source, device)
        if processor.tokenizer.get_vocab() != common_processor.tokenizer.get_vocab():
            raise ValueError('Models must share the tokenizer vocabulary for identical preprocessing')
        metrics, predictions = evaluate(model, common_processor, rows, image_root, config, scorer, device, with_loss=True)
        results[label] = {"source": source, "revision": getattr(model.config, "_commit_hash", None), **metrics}
        write_json(prediction_path, predictions)
        write_json(metric_path, {'metrics': results[label], 'predictions_sha256': sha256(prediction_path)})
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(folder / "summary.json", {"models": results, "protocol": protocol})
    fields = ['model', 'examples', 'vqa_score', 'validation_loss', 'unanswerable_output_rate', 'answered_coverage']
    write_csv(folder / 'model_comparison.csv', [{'model': label, **metrics} for label, metrics in results.items()], fields)
    labels = list(results)
    predictions = {label: read_json(folder / f'{label}_predictions.json') for label in labels}
    combined = []
    for i, row in enumerate(rows):
        record = {key: row[key] for key in ('image', 'question', 'answers', 'answerable', 'answer_type')}
        for label in labels:
            if predictions[label][i]['image'] != row['image']:
                raise ValueError('Model predictions must have exactly the same holdout ordering')
            record[label + '_prediction'] = predictions[label][i]['prediction']
            record[label + '_score'] = predictions[label][i]['score']
        combined.append(record)
    write_csv(folder / 'sample_comparison.csv', combined, list(combined[0]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(['Pretrained BLIP', 'Existing VizWiz', 'Improved export'], [100*results[k]['vqa_score'] for k in labels])
    axis.bar_label(bars, fmt='%.2f%%')
    axis.set(ylabel='Official VQA agreement (%)', ylim=(0, 105), title=f'Identical holdout examples (n={len(rows)})')
    fig.tight_layout()
    fig.savefig(folder / 'model_comparison.png', dpi=160)
    fig.savefig(folder / 'model_comparison.pdf')
    plt.close(fig)
    check_run(config, run_dir)
    write_json(run_dir / "final_comparison.json", {"folder": str(folder.relative_to(run_dir)), "models": results})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = TrainConfig(**read_json(Path(args.run_dir) / "config.json"))
    if args.action == "prepare":
        prepare_official_data(config, args.run_dir, args.data_root)
    else:
        scorer = OfficialScorer.from_file(Path(args.run_dir) / "official_normalization.json")
        if args.action == "train":
            train_run(config, args.run_dir, Path(args.data_root) / "images", scorer, resume=args.resume)
        else:
            print(json.dumps(final_compare(config, args.run_dir, Path(args.data_root) / "images", scorer), indent=2))
