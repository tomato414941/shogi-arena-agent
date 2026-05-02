import unittest

from shogi_arena_agent.local_match import play_local_match
from shogi_arena_agent.usi_process import UsiProcess


class UsiProcessTest(unittest.TestCase):
    def test_process_returns_bestmove(self) -> None:
        with UsiProcess() as process:
            process.position("position startpos moves 7g7f")
            bestmove = process.go()

        self.assertNotEqual(bestmove, "")

    def test_process_can_play_local_match(self) -> None:
        with UsiProcess() as black, UsiProcess() as white:
            result = play_local_match(black=black, white=white, max_plies=4)

        self.assertEqual(result.end_reason, "max_plies")
        self.assertEqual(len(result.moves), 4)


if __name__ == "__main__":
    unittest.main()
