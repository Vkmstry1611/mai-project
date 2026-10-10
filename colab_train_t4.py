# Paste this entire file into ONE cell in a fresh Google Colab T4 runtime.
import os, sys, subprocess, time
START = time.monotonic()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade',
    '--only-binary=:all:', 'transformers==4.57.6', 'tokenizers==0.22.2',
    'datasets==4.5.0', 'huggingface-hub>=0.34,<1.0',
    'Pillow>=11.1,<12', 'pandas', 'matplotlib', 'tqdm'])

import json, random, re, io, shutil
from pathlib import Path
from collections import Counter
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset, Image as DatasetImage
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForQuestionAnswering
from tqdm.auto import tqdm

# Conservative defaults; the timer may stop before all epochs finish.
SEED, TRAIN_SIZE, EVAL_SIZE = 42, 800, 100
BATCH_SIZE, EPOCHS, LR = 4, 2, 2e-5
TRAIN_MINUTES = 25
MODEL_ID = 'Salesforce/blip-vqa-base'
DATASET_ID = 'Raluck/vizwiz-vqa-en'
ROOT = Path('/content/visionbridge_t4')
OUT = ROOT / 'export'
MODEL_OUT = OUT / 'blip_vizwiz'
OUT.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU.'
torch.cuda.manual_seed_all(SEED)
print('GPU:', torch.cuda.get_device_name(0))

def elapsed():
    return (time.monotonic() - START) / 60

def normalize(text):
    # Lightweight diagnostic normalization, NOT the official evaluator.
    text = re.sub(r'[^a-z0-9 ]', ' ', str(text).lower())
    return ' '.join(text.split())

def references(answers):
    if isinstance(answers, dict):
        return [str(a) for a in answers['answer']]
    return [str(a['answer'] if isinstance(a, dict) else a) for a in answers]

def download_subset(split, count):
    # Stream parquet: avoid downloading/extracting the full image archives.
    # Bounded shuffle is a speed compromise, not a uniform full-split sample.
    stream = load_dataset(DATASET_ID, split=split, streaming=True)
    stream = stream.cast_column('image', DatasetImage(decode=False))
    stream = stream.shuffle(seed=SEED, buffer_size=64)
    folder = ROOT / split
    folder.mkdir(parents=True, exist_ok=True)
    rows, failed = [], 0
    for record in tqdm(stream, total=count, desc='Download ' + split):
        if elapsed() > 20:
            raise RuntimeError('Data preparation exceeded 20 min. Retry with a smaller subset.')
        try:
            raw = record['image']
            source = io.BytesIO(raw['bytes']) if raw.get('bytes') else raw['path']
            name = record.get('image_name', f'{split}_{len(rows):06d}.jpg')
            path = folder / Path(name).name
            with Image.open(source) as im:
                im = im.convert('RGB')
                im.thumbnail((640, 640))
                im.save(path, quality=90)
            refs = references(record['answers'])
            target = Counter(normalize(a) for a in refs if normalize(a)).most_common(1)[0][0]
            rows.append(dict(image=str(path), image_name=name,
                question=record['question'], answers=refs,
                answerable=int(record['answerable']), target=target))
        except (OSError, ValueError, KeyError, IndexError) as error:
            failed += 1
            if failed <= 3:
                print('Skipped record:', error)
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise RuntimeError(f'Only {len(rows)} usable {split} records; requested {count}.')
    print(split, len(rows), 'records; skipped:', failed)
    return rows

train_rows = download_subset('train', TRAIN_SIZE)
eval_rows = download_subset('validation', EVAL_SIZE)
assert not ({r['image_name'] for r in train_rows} & {r['image_name'] for r in eval_rows})
print(f'Data ready: {elapsed():.1f} min')

processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForQuestionAnswering.from_pretrained(MODEL_ID).to('cuda')
# Keep the pretrained image and question encoders; train the last two decoder
# transformer blocks and language-model head. This is partial fine-tuning.
model.requires_grad_(False)
for layer in model.text_decoder.bert.encoder.layer[-2:]:
    layer.requires_grad_(True)
model.text_decoder.cls.requires_grad_(True)
params = [p for p in model.parameters() if p.requires_grad]
print('Trainable parameters:', f'{sum(p.numel() for p in params):,}')

def collate(rows, training=True):
    images = []
    for row in rows:
        with Image.open(row['image']) as im:
            images.append(im.convert('RGB'))
    inputs = processor(images=images, text=[r['question'] for r in rows],
        padding=True, truncation=True, max_length=48, return_tensors='pt')
    if training:
        answers = processor.tokenizer([r['target'] for r in rows],
            padding=True, truncation=True, max_length=24, return_tensors='pt')
        decoder_ids = answers.input_ids.clone()
        decoder_ids[:, 0] = model.config.text_config.bos_token_id
        labels = decoder_ids.clone()
        labels[answers.attention_mask == 0] = -100
        # Separate decoder IDs are essential: -100 cannot be embedded.
        inputs['decoder_input_ids'] = decoder_ids
        inputs['decoder_attention_mask'] = answers.attention_mask
        inputs['labels'] = labels
    return {k: v.to('cuda') for k, v in inputs.items()}

@torch.inference_mode()
def predict_rows(rows):
    model.eval()
    predictions = []
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc='Evaluate'):
        with torch.autocast('cuda', dtype=torch.float16):
            tokens = model.generate(**collate(rows[i:i+BATCH_SIZE], False),
                max_new_tokens=16, num_beams=1, do_sample=False)
        predictions.extend(processor.batch_decode(tokens, skip_special_tokens=True))
    return predictions

def score(pred, refs):
    # Leave-one-annotator-out agreement; normalization remains approximate.
    refs = [normalize(a) for a in refs]
    pred = normalize(pred)
    return sum(min(1.0, sum(pred == a for j, a in enumerate(refs) if j != i)/3)
               for i in range(len(refs))) / max(1, len(refs))

baseline_predictions = predict_rows(eval_rows)
baseline_score = sum(score(p, r['answers']) for p, r in
    zip(baseline_predictions, eval_rows)) / len(eval_rows)
print(f'Baseline diagnostic agreement: {baseline_score:.3f}')

loader = DataLoader(train_rows, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, collate_fn=collate)
optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda')
TRAIN_START = time.monotonic()
# Stop training by 45 min from cell start, reserving time for evaluation/export.
DEADLINE = min(TRAIN_START + TRAIN_MINUTES*60, START + 45*60)
history, updates, stopped = [], 0, False
for epoch in range(EPOCHS):
    model.train()
    model.vision_model.eval()
    model.text_encoder.eval()
    progress = tqdm(loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for batch in progress:
        if time.monotonic() >= DEADLINE:
            stopped = True
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss; stop and lower the learning rate.')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            updates += 1
        value = loss.item()
        history.append(dict(epoch=epoch+1, loss=value, minutes=elapsed()))
        progress.set_postfix(loss=f'{value:.3f}', minutes=f'{elapsed():.1f}')
    if stopped:
        print('Time budget reached; exporting completed training.')
        break
assert updates > 0, 'No optimizer updates completed; this is not a trained checkpoint.'

# Save immediately so the trained checkpoint survives an evaluation error.
model.eval()
model.save_pretrained(MODEL_OUT, safe_serialization=True)
processor.save_pretrained(MODEL_OUT)
pd.DataFrame(history).to_csv(OUT / 'training_loss.csv', index=False)
json.dump({'dataset': DATASET_ID, 'seed': SEED,
    'train_images': [r['image_name'] for r in train_rows],
    'eval_images': [r['image_name'] for r in eval_rows]},
    open(OUT / 'subset_manifest.json', 'w'), indent=2)

trained_predictions = predict_rows(eval_rows)
results = pd.DataFrame([dict(image=r['image_name'], question=r['question'],
    answerable=r['answerable'], references=' | '.join(r['answers']),
    baseline=b, trained=p, baseline_score=score(b, r['answers']),
    trained_score=score(p, r['answers']))
    for r, b, p in zip(eval_rows, baseline_predictions, trained_predictions)])
results.to_csv(OUT / 'predictions.csv', index=False)
summary = dict(model=MODEL_ID, dataset=DATASET_ID, train_size=len(train_rows),
    eval_size=len(eval_rows), epochs_requested=EPOCHS, optimizer_updates=updates,
    stopped_by_timer=stopped, baseline_agreement=float(results.baseline_score.mean()),
    trained_agreement=float(results.trained_score.mean()),
    fine_tuning='last two decoder layers and LM head',
    metric='diagnostic VQA agreement with approximate normalization',
    minutes_before_export=elapsed())
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=2)
print(json.dumps(summary, indent=2))
display(results.head(10))
plt.figure(figsize=(8, 3))
plt.plot([r['loss'] for r in history])
plt.xlabel('Training batch')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(OUT / 'loss.png', dpi=150)
plt.show()

# ZIP_STORED avoids spending minutes compressing already dense model weights.
import zipfile
archive = ROOT / 'visionbridge_t4.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
    for path in OUT.rglob('*'):
        if path.is_file():
            z.write(path, path.relative_to(OUT))
print(f'Finished in {elapsed():.1f} min. ZIP: {archive.stat().st_size/1e9:.2f} GB')
from google.colab import files
files.download(str(archive))

# Optional inference after training: ask('/content/photo.jpg', 'What is this?')
@torch.inference_mode()
def ask(image_path, question):
    model.eval()
    with Image.open(image_path) as im:
        inputs = processor(images=im.convert('RGB'), text=question,
            return_tensors='pt').to('cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        tokens = model.generate(**inputs, max_new_tokens=20, num_beams=1)
    return processor.decode(tokens[0], skip_special_tokens=True).strip()
