"""Module entry point for the RADAR-PD NOVA application."""

import argparse

from .app import RadarPdNovaApp


def main() -> None:
    parser = argparse.ArgumentParser(description="RADAR-PD NOVA interactive client")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--timeout", default=0, type=int)
    parser.add_argument("--galaxy-history-id", default="")
    args, _unknown = parser.parse_known_args()
    app = RadarPdNovaApp()
    app.server.start(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        open_browser=False,
    )


if __name__ == "__main__":
    main()
