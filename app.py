"""Entry point: `python app.py` locally and in the Docker image.

Local:            http://127.0.0.1:7860
Docker/Cloud Run: Cloud Run sets PORT (8080) and the app listens on 0.0.0.0:$PORT
"""

from __future__ import annotations

import logging
import os
import threading

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma telemetry off

from notebooklm.services import build_services, warm_up  # noqa: E402
from notebooklm.ui.gradio_app import CSS, HEAD, THEME, build_app  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

services = build_services()
demo = build_app(services)


def main() -> None:
    # Load the embedding + reranker models in the background so the UI comes up immediately.
    threading.Thread(target=warm_up, args=(services,), daemon=True).start()

    port = os.getenv("PORT")
    auth = None
    if os.getenv("APP_USERNAME") and os.getenv("APP_PASSWORD"):
        auth = (os.environ["APP_USERNAME"], os.environ["APP_PASSWORD"])

    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0" if port else None,
        server_port=int(port) if port else None,
        allowed_paths=[str(services.settings.data_dir)],
        auth=auth,
        theme=THEME,
        css=CSS,
        head=HEAD,
        footer_links=[],
        show_error=True,
    )


if __name__ == "__main__":
    main()
