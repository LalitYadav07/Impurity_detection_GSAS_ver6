"""Module entry point for the RADAR-PD NOVA application."""

import argparse
from pathlib import Path

from .app import RadarPdNovaApp


def main() -> None:
    parser = argparse.ArgumentParser(description="RADAR-PD NOVA interactive client")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--timeout", default=0, type=int)
    parser.add_argument("--galaxy-history-id", default="")
    args, _unknown = parser.parse_known_args()
    app = RadarPdNovaApp()
    # Nginx serves this in NDIP; direct local launches need the same guide.
    guide=next((path for path in (Path('/app/www-content/user-guide'),
               Path(__file__).resolve().parents[2]/'help') if (path/'index.html').is_file()),None)
    if guide is not None:
        app.server.enable_module({'serve':{'user-guide':str(guide)}})
    app.server.start(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        open_browser=False,
    )


if __name__ == "__main__":
    main()
