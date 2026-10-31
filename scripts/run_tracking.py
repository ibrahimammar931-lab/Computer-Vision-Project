"""Convenience launcher for the ByteTrack stage."""
import sys

from src.main import main


if __name__ == "__main__":
    raise SystemExit(main(["track", *sys.argv[1:]]))
