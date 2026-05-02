from __future__ import annotations

import sys

from shogi_arena_agent.usi import UsiEngine


def main() -> None:
    engine = UsiEngine()
    for line in sys.stdin:
        if line.strip() == "quit":
            break
        for response in engine.handle_line(line):
            print(response, flush=True)


if __name__ == "__main__":
    main()
