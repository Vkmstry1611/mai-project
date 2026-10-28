"""Real synthetic-photo UI smoke checks; never reads benchmark images."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    from PIL import Image, ImageDraw, ImageFont
    from launch_phase4 import install_worker
    out = ROOT / 'reports/application_validation' / ('run-local-' + time.strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'inputs').mkdir()
    app = install_worker()
    frozen = ROOT / 'reports/phase4_policy/run-local-20261013-v1'
    font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 80)
    cases = [
        ('welcome', [(80, 140, 'WELCOME')], 'What does this sign say?'),
        ('ambiguous', [(50, 60, 'STOP'), (430, 300, 'NOTICE')], 'What does this sign say?'),
        ('bottle_route', [(80, 140, 'WELCOME')], 'What does the bottle say?'),
    ]
    records = []
    for name, lines, question in cases:
        image = Image.new('RGB', (900, 480), 'white')
        draw = ImageDraw.Draw(image)
        for x, y, text in lines:
            draw.text((x, y), text, font=font, fill='black')
        image.save(out / 'inputs' / (name + '.png'))
        start = time.perf_counter()
        print('Starting synthetic callback:', name, flush=True)
        try:
            shown, audio, status, details, annotated = app.ask(image, question, False, frozen)
            record = dict(name=name, question=question, answer=shown, status=status,
                          seconds=time.perf_counter()-start, details=details, callback_completed=True)
            annotated.save(out / (name + '-evidence.png'))
        except Exception as error:
            record = dict(name=name, seconds=time.perf_counter()-start, callback_completed=False,
                          error=repr(error))
        records.append(record)
        (out / (name + '.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(name, record.get('answer', record.get('error')), round(record['seconds'], 2), flush=True)
    errors = []
    for image, question in [(None, 'Read this sign'), (Image.new('RGB', (10, 10)), '   ')]:
        try:
            app.ask(image, question, False, frozen)
            errors.append({'rejected': False})
        except ValueError as error:
            errors.append({'rejected': True, 'message': str(error)})
    ui = app.build_app(frozen)
    summary = {'records': records, 'input_errors': errors, 'ui_blocks': len(ui.blocks),
               'environment': sys.prefix, 'synthetic_only': True, 'policy_changed': False,
               'accuracy_evaluation': False, 'browser_hardware_tested': False}
    (out / 'SMOKE_RESULTS.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Results:', out, flush=True)


if __name__ == '__main__':
    main()
