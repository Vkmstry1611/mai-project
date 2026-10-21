"""Separate opt-in workspace with experimental automatic cardinal OCR."""
import argparse
from pathlib import Path


if __name__ == '__main__':
    import sys
    if Path(sys.prefix).name != '.venv':
        raise RuntimeError('Use .venv for Gradio; the worker uses .venv-gpu')
    from launch_workspace import build_app, launch_options
    from orientation_workspace_session import OrientationSession
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='reports/phase4_policy/run-local-20261013-v1')
    parser.add_argument('--port', type=int, default=7866)
    args = parser.parse_args()
    session = OrientationSession(args.run_dir)
    demo, _ = build_app(Path(args.run_dir).resolve(), session=session)
    print('EXPERIMENTAL automatic-orientation OCR. First uncached reading searches four angles; this may take a minute. '
          'Advanced manual OCR bypasses this feature. Ctrl+C stops this server and its owned worker.', flush=True)
    try:
        demo.queue(default_concurrency_limit=1).launch(server_port=args.port, **launch_options())
    finally:
        session.close()
