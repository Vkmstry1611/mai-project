"""Experimental cardinal-orientation OCR. No frozen policy imports are changed.

CPU-only, English, no spelling correction, contrast enhancement or upscaling.
Selection scores describe consistency of OCR evidence, NOT correctness probabilities.
"""
import math
from pathlib import Path
import re
import statistics
import time
import uuid
import json

from modules.evidence_store import digest, image_identity, standardize
from visionbridge.experimental_reading import reconstruct

VERSION = 'experimental-cardinal-ocr-v1'
ANGLES = (0, 90, 180, 270)
THRESHOLD = .90
MARGIN = .12
MAX_SIDE = 1600
BUDGET_SECONDS = 90
NOTICE = ('I cannot determine a reliable reading orientation. Please use Advanced / Developer '
          'to select Experimental OCR Reading and rotate the image, or upload a clearer upright photo.')


def transform(point, width, height, angle, inverse=False):
    """Continuous image-edge coordinates, CCW; width/height are ORIGINAL extents."""
    if angle not in ANGLES:
        raise ValueError('Use a cardinal angle')
    x, y = point
    if inverse:
        return {0: (x, y), 90: (width-y, x), 180: (width-x, height-y), 270: (y, height-x)}[angle]
    return {0: (x, y), 90: (y, width-x), 180: (width-x, height-y), 270: (height-y, x)}[angle]


def bounds(polygon):
    return [min(p[0] for p in polygon), min(p[1] for p in polygon),
            max(p[0] for p in polygon), max(p[1] for p in polygon)]


def matrices(original_size, working_size, angle):
    ow, oh = original_size
    w, h = working_size
    def matrix(fn):
        a, b, c = fn((0, 0)), fn((1, 0)), fn((0, 1))
        return [[b[0]-a[0], c[0]-a[0], a[0]], [b[1]-a[1], c[1]-a[1], a[1]], [0, 0, 1]]
    def forward(p):
        return transform((p[0]*w/ow, p[1]*h/oh), w, h, angle)
    def inverse(p):
        x, y = transform(p, w, h, angle, inverse=True)
        return x*ow/w, y*oh/h
    return dict(forward_3x3=matrix(forward), inverse_3x3=matrix(inverse))


def process_memory_mib():
    """Windows process working set, not GPU memory; peak is cumulative."""
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    c=Counters();c.cb=ctypes.sizeof(c)
    process=ctypes.windll.kernel32.GetCurrentProcess;process.restype=wintypes.HANDLE
    query=ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not query(process(),ctypes.byref(c),c.cb):raise ctypes.WinError()
    return dict(host_working_set_mib=c.WorkingSetSize/2**20,host_peak_working_set_mib=c.PeakWorkingSetSize/2**20)


def credible(item):
    text = ''.join(c for c in item['text'] if c.isalnum())
    x1, y1, x2, y2 = item['box']
    # Isolated single glyphs cannot establish an automatic orientation. This
    # includes letters AND digits; multi-digit text remains eligible.
    return len(text) >= 2 and item['confidence'] >= THRESHOLD and (x2-x1) >= 1.1*(y2-y1)


def assess(items):
    """Normalized evidence quality, not character/fragment count maximization."""
    reliable = [i for i in items if credible(i)]
    median = statistics.median([i['confidence'] for i in items]) if items else 0
    neighbors = set()
    for a in reliable:
        for b in reliable:
            if a['id'] == b['id']:
                continue
            ab, bb = a['box'], b['box']
            h, k = ab[3]-ab[1], bb[3]-bb[1]
            if (min(h, k) >= .6*max(h, k) and abs((ab[1]+ab[3]-bb[1]-bb[3])/2) <= .6*min(h, k)
                    and -min(h, k) <= bb[0]-ab[2] <= 3*max(h, k)):
                neighbors.add(a['id'])
    coherence = len(neighbors)/len(reliable) if len(reliable)>1 else (.5 if reliable else 0)
    fraction = len(reliable)/len(items) if items else 0
    score = .45*median + .35*fraction + .20*coherence if reliable else 0
    return dict(score=score, median_confidence=median, reliable_fraction=fraction,
                spatial_coherence=coherence, reliable_ids=[i['id'] for i in reliable], eligible=bool(reliable))


def rank(candidates):
    eligible = sorted([c for c in candidates if c['assessment']['eligible']],
                      key=lambda c: (-c['assessment']['score'], c['angle']))
    if not eligible:
        return None, 'insufficient_orientation_evidence'
    if len(eligible)>1 and eligible[0]['assessment']['score']-eligible[1]['assessment']['score'] < MARGIN:
        return None, 'ambiguous_or_mixed_orientation'
    return eligible[0], 'dominant_orientation_pending_verification'


def normalize(text):
    return re.sub(r'\s+', ' ', text.strip()).casefold()


def verify(reader, image, candidate):
    """Recognition-only crop rechecks; no new text is added to the answer."""
    import numpy as np
    from PIL import ImageOps
    reliable = sorted([i for i in candidate['items'] if credible(i)],
                      key=lambda i: (-i['confidence'], i['id']))[:3]
    grey = np.asarray(ImageOps.grayscale(image))
    checks = []
    for item in reliable:
        x1, y1, x2, y2 = item['box']
        rectangle = [max(0, int(x1)), min(image.width, math.ceil(x2)),
                     max(0, int(y1)), min(image.height, math.ceil(y2))]
        readings = reader.recognize(grey, horizontal_list=[rectangle], free_list=[], reformat=False)
        matched = any(normalize(str(r[1])) == normalize(item['text']) and float(r[2]) >= THRESHOLD for r in readings)
        checks.append(dict(evidence_id=item['id'], original_text=item['text'], matched=matched,
                           readings=[dict(text=str(r[1]), confidence=float(r[2])) for r in readings]))
    return checks, bool(checks) and sum(c['matched'] for c in checks) >= math.ceil(2*len(checks)/3)


def mapped_items(candidate, original_size, working_size, identity):
    ow, oh = original_size
    w, h = working_size
    result = []
    for row in candidate['items']:
        polygon = row['polygon'] or [[row['box'][0], row['box'][1]], [row['box'][2], row['box'][1]],
                                    [row['box'][2], row['box'][3]], [row['box'][0], row['box'][3]]]
        original = []
        for point in polygon:
            x, y = transform(point, w, h, candidate['angle'], inverse=True)
            original.append([min(ow, max(0, x*ow/w)), min(oh, max(0, y*oh/h))])
        result.append(dict(row, original_polygon=original, original_box=bounds(original),
            corrected_polygon=polygon, corrected_box=row['box'], polygon=original, box=bounds(original),
            image_id=identity, coordinate_space='original-exif-corrected-rgb-pixels',
            corrected_coordinate_space='cardinal-corrected-working-image-pixels'))
    return result


def _raw(reader, image):
    import numpy as np
    rows = []
    for poly, text, confidence in reader.readtext(np.asarray(image)):
        polygon = [[float(x), float(y)] for x, y in poly]
        rows.append(dict(text=str(text), confidence=float(confidence), polygon=polygon, box=bounds(polygon)))
    return rows


def read(image, cache_dir, source, get_reader, budget=BUDGET_SECONDS, clock=time.perf_counter):
    """All four passes required. Budget is soft: checked BETWEEN native calls.

    No unvalidated estimator prunes angles. Cache contains every candidate and
    the selection decision; previous OCR caches are neither read nor written.
    """
    from PIL import ImageOps
    start = clock()
    original = ImageOps.exif_transpose(image).convert('RGB')
    if original.width*original.height > 40_000_000:
        raise ValueError('Resize this photo below 40 megapixels before experimental OCR.')
    work = original.copy()
    work.thumbnail((MAX_SIDE, MAX_SIDE))  # Bounded downscaling ONLY; never upscale.
    identity = dict(schema=VERSION, image=image_identity(original), source=source, threshold=THRESHOLD,
        angles=ANGLES, margin=MARGIN, max_side=MAX_SIDE, preprocessing='none', budget_seconds=budget,
        selector='median-fraction-spatial-v1-crop-recheck', grouping='unchanged-literal-reader-v1')
    key = digest(identity)
    path = Path(cache_dir)/VERSION/(key+'.json')
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if digest(saved['identity']) != digest(identity) or digest(saved['result']) != saved['result_sha256']:
            raise ValueError('Experimental orientation cache integrity mismatch; preserve it and use another directory.')
        result = saved['result']
        result['orientation']['cache_hit'] = True
        result['extraction_status']['ocr']['cache_hit'] = True
        result['orientation']['request_seconds'] = clock()-start
        return result
    loader = clock()
    reader = get_reader()
    loader_seconds = clock()-loader
    candidates = []
    timed_out = False
    for angle in ANGLES:
        if clock()-start >= budget:
            timed_out = True
            break
        rotated = work.rotate(angle, expand=True)
        tick = clock()
        raw = _raw(reader, rotated)
        seconds = clock()-tick
        record = standardize(rotated, raw, 'ocr', dict(source, experimental=VERSION, angle_ccw=angle,
            original_image_id=identity['image'], working_size=list(work.size)))
        candidates.append(dict(angle=angle, raw_rows=raw, items=record['items'], status=record['status'],
            rejections=record['rejections'], assessment=assess(record['items']), ocr_seconds=seconds))
    timed_out = timed_out or clock()-start >= budget
    winner, reason = rank(candidates) if not timed_out and len(candidates)==4 else (None, 'orientation_time_budget_exceeded')
    checks = []
    if winner is not None:
        tick = clock()
        checks, stable = verify(reader, work.rotate(winner['angle'], expand=True), winner)
        verification_seconds = clock()-tick
        if not stable:
            winner, reason = None, 'orientation_verification_failed'
        elif clock()-start >= budget:
            winner, reason, timed_out = None, 'orientation_time_budget_exceeded', True
        else:
            reason = 'confidence_geometry_and_crop_consistency'
    else:
        verification_seconds = 0
    if winner is not None:
        decision = reconstruct(winner['items'], THRESHOLD)
        evidence = mapped_items(winner, original.size, work.size, identity['image'])
    else:
        decision = dict(answer=NOTICE, lines=[], decision='experimental_orientation_abstained',
            supporting_evidence_ids=[], conflicting_ids=[], partial=True, threshold=THRESHOLD,
            answer_source='abstention', reason=reason)
        evidence = []
    orientation = dict(version=VERSION, selected_ccw_degrees=winner['angle'] if winner else None,
        decision_reason=reason, cache_hit=False, source_image_id=identity['image'], original_size=list(original.size),
        working_size=list(work.size), corrected_size=list(work.rotate(winner['angle'], expand=True).size) if winner else None,
        transform=dict(convention='continuous image-edge coordinates, CCW rotation after scale',
            scale_x=work.width/original.width, scale_y=work.height/original.height,
            inverse='inverse cardinal rotation then divide by actual scale_x/scale_y',
            **(matrices(original.size, work.size, winner['angle']) if winner else {})),
        confidence_threshold=THRESHOLD, selection_margin=MARGIN, verification=checks,
        all_candidates=candidates, preprocessing='none; bounded downscaling only if >1600 pixels',
        budget_seconds=budget, budget_is_soft=True, timed_out=timed_out, reader_loading_seconds=loader_seconds,
        verification_seconds=verification_seconds, request_seconds=clock()-start,
        estimator_used_to_skip_angles=False, warning='Orientation consistency is not proof of a correct or complete reading.')
    details = dict(decision, mode='EXPERIMENTAL automatic-orientation OCR — not frozen Phase 4', baseline_answer=None,
        final_answer=decision['answer'], evidence=evidence,
        selected_evidence=[e for e in evidence if e['id'] in decision['supporting_evidence_ids']],
        orientation=orientation, coordinate_space='original-exif-corrected-rgb-pixels',
        extraction_status={'ocr': dict(status=winner['status'] if winner else 'orientation_unresolved', cache_hit=False)},
        blip_status='not run; no BLIP fallback', yolo_status='not required for literal reading',
        warnings=['Experimental fragments only; unreadable or clipped words are not reconstructed.',
                  'A dominant orientation may omit text at other angles. Reading may be incomplete.'])
    if not timed_out:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(identity=identity, result=details, result_sha256=digest(details))
        temporary = path.with_name(key+'.'+uuid.uuid4().hex+'.partial')
        with temporary.open('x', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, allow_nan=False)
        if path.exists():
            raise FileExistsError('Concurrent experimental cache writer; existing entry preserved.')
        temporary.rename(path)
    return details
