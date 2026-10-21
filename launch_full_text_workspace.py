"""Explicit entry point to the experimental full-text unified workspace."""
from pathlib import Path
import argparse

if __name__=='__main__':
    from launch_workspace import build_app,launch_options
    from full_text_session import FullTextSession
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=7867)
    p.add_argument('--run-dir',default='reports/phase4_policy/run-local-20261013-v1');a=p.parse_args()
    session=FullTextSession(a.run_dir);demo,_=build_app(Path(a.run_dir).resolve(),session)
    try:demo.queue(default_concurrency_limit=1).launch(server_port=a.port,**launch_options())
    finally:session.close()
