import unittest
import sys

from shogi_arena_agent.shogi_game import play_shogi_game
from shogi_arena_agent.usi_process import UsiProcess


class UsiProcessTest(unittest.TestCase):
    def test_process_returns_bestmove(self) -> None:
        with UsiProcess() as process:
            process.position("position startpos moves 7g7f")
            result = process.go()

        self.assertNotEqual(result.bestmove, "")

    def test_explicit_command_can_be_used_as_external_engine(self) -> None:
        command = [sys.executable, "-m", "shogi_arena_agent"]
        with UsiProcess(command=command) as process:
            process.position("position startpos")
            result = process.go()

        self.assertNotEqual(result.bestmove, "")

    def test_go_strips_ponder_move_from_bestmove(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "for line in sys.stdin:\n"
                "    line = line.strip()\n"
                "    if line == 'usi': print('usiok', flush=True)\n"
                "    elif line == 'isready': print('readyok', flush=True)\n"
                "    elif line.startswith('go'): print('bestmove 7g7f ponder 3c3d', flush=True)\n"
                "    elif line == 'quit': break\n"
            ),
        ]
        with UsiProcess(command=command) as process:
            result = process.go()

        self.assertEqual(result.bestmove, "7g7f")
        self.assertEqual(result.ponder, "3c3d")

    def test_go_command_can_be_overridden(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "for line in sys.stdin:\n"
                "    line = line.strip()\n"
                "    if line == 'usi': print('usiok', flush=True)\n"
                "    elif line == 'isready': print('readyok', flush=True)\n"
                "    elif line == 'go nodes 1': print('bestmove 2g2f', flush=True)\n"
                "    elif line.startswith('go'): print('bestmove 7g7f', flush=True)\n"
                "    elif line == 'quit': break\n"
            ),
        ]
        with UsiProcess(command=command, go_command="go nodes 1") as process:
            result = process.go()

        self.assertEqual(result.bestmove, "2g2f")

    def test_go_preserves_info_lines(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "for line in sys.stdin:\n"
                "    line = line.strip()\n"
                "    if line == 'usi': print('usiok', flush=True)\n"
                "    elif line == 'isready': print('readyok', flush=True)\n"
                "    elif line.startswith('go'):\n"
                "        print('info depth 4 nodes 100 score cp 23 pv 7g7f', flush=True)\n"
                "        print('info multipv 2 depth 3 nodes 80 score cp 10 pv 2g2f', flush=True)\n"
                "        print('bestmove 7g7f', flush=True)\n"
                "    elif line == 'quit': break\n"
            ),
        ]
        with UsiProcess(command=command) as process:
            result = process.go()

        self.assertEqual(result.bestmove, "7g7f")
        self.assertEqual(
            result.info_lines,
            (
                "info depth 4 nodes 100 score cp 23 pv 7g7f",
                "info multipv 2 depth 3 nodes 80 score cp 10 pv 2g2f",
            ),
        )

    def test_stderr_output_does_not_block_bestmove(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "for line in sys.stdin:\n"
                "    line = line.strip()\n"
                "    if line == 'usi': print('usiok', flush=True)\n"
                "    elif line == 'isready': print('readyok', flush=True)\n"
                "    elif line.startswith('go'):\n"
                "        print('diagnostic stderr output', file=sys.stderr, flush=True)\n"
                "        print('bestmove 7g7f', flush=True)\n"
                "    elif line == 'quit': break\n"
            ),
        ]
        with UsiProcess(command=command) as process:
            result = process.go()

        self.assertEqual(result.bestmove, "7g7f")

    def test_rejects_non_go_command(self) -> None:
        with self.assertRaises(ValueError):
            UsiProcess(go_command="position startpos")

    def test_process_can_play_shogi_game(self) -> None:
        with UsiProcess() as black, UsiProcess() as white:
            result = play_shogi_game(black=black, white=white, max_plies=4)

        self.assertEqual(result.end_reason, "max_plies")
        self.assertEqual(len(result.transitions), 4)

    def test_start_times_out_when_process_does_not_answer(self) -> None:
        command = [sys.executable, "-c", "import time; time.sleep(10)"]
        process = UsiProcess(command=command, read_timeout_seconds=0.1)

        with self.assertRaises(TimeoutError):
            process.start()
        process.close()


if __name__ == "__main__":
    unittest.main()
