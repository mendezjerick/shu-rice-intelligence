from __future__ import annotations

import sys
from pathlib import Path

import webview

from backend import RiceAppBackend


def main() -> None:
    """Launch the Shu Rice Intelligence desktop window backed by PyWebView."""
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    backend = RiceAppBackend(base_path=base_path)
    html_path = base_path / "web" / "index.html"
    window = webview.create_window(
        "Shu Rice Intelligence",
        html_path.as_uri(),
        js_api=backend,
        width=1280,
        height=800,
        resizable=True,
    )
    webview.start()


if __name__ == "__main__":
    main()
