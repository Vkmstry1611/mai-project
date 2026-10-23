"""Reproduce the browser upload -> analyze event sequence; no dataset evaluation."""
import json
import socket
import traceback
from pathlib import Path


def main():
    from full_text_session import FullTextSession
    from launch_workspace import build_app, launch_options
    from gradio_client import Client, handle_file
    from workspace_ui import ASSISTANT
    session = FullTextSession('reports/phase4_policy/run-local-20261013-v1')
    demo, _ = build_app(session.run_dir, session)
    # Expose only on this temporary test server so the client can reproduce the
    # normally browser-only upload event with the identical callback/bindings.
    for fn in demo.fns.values():
        if fn.fn and fn.fn.__name__ == 'upload_reset':
            fn.api_visibility = 'undocumented'
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        _, url, _ = demo.queue().launch(server_port=port, prevent_thread_lock=True, quiet=True, **launch_options())
        client = Client(url, verbose=False)
        photo = Path('reports/phase4_policy/run-local-20261013-v1/application-requests/d57ffbb579ea495d86fb10c36cc87b93/input.png')
        print('UPLOAD EVENT', flush=True)
        client.predict(handle_file(str(photo)), api_name='/upload_reset')
        print('ANALYZE AFTER UPLOAD', flush=True)
        result = client.predict(handle_file(str(photo)), 'what is written in the image?', True, ASSISTANT, '0', api_name='/analyze')
        print(json.dumps({'answer': result[0], 'outputs': len(result)}, ensure_ascii=False), flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        demo.close()
        session.close()


if __name__ == '__main__':
    main()
