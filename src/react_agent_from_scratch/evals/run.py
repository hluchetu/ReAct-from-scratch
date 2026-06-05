from __future__ import annotations

import sys

import pytest


def main() -> None:
    raise SystemExit(pytest.main(["src/react_agent_from_scratch/evals", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
