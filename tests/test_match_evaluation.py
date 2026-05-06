import unittest
import sys

from shogi_arena_agent.match_evaluation import evaluate_player_against_deterministic_legal, evaluate_player_against_usi_engine
from shogi_arena_agent.usi import UsiEngine


class MatchEvaluationTest(unittest.TestCase):
    def test_evaluates_player_on_both_sides(self) -> None:
        evaluation = evaluate_player_against_deterministic_legal(UsiEngine(), game_count=2, max_plies=4)

        self.assertEqual(evaluation.game_count, 2)
        self.assertEqual(evaluation.black_game_count, 1)
        self.assertEqual(evaluation.white_game_count, 1)
        self.assertEqual(evaluation.end_reasons, {"max_plies": 2})
        self.assertEqual(evaluation.average_plies, 4.0)
        self.assertEqual(evaluation.illegal_move_count, 0)
        self.assertEqual(evaluation.player_wins, 0)
        self.assertEqual(evaluation.player_losses, 0)
        self.assertEqual(evaluation.draws, 2)
        self.assertEqual(len(evaluation.results), 2)
        self.assertEqual(evaluation.results[0].black_actor.name, "shogi-arena-agent")
        self.assertEqual(evaluation.results[0].white_actor.kind, "deterministic_legal")

    def test_evaluates_player_against_external_usi_engine(self) -> None:
        evaluation = evaluate_player_against_usi_engine(
            UsiEngine(),
            [sys.executable, "-m", "shogi_arena_agent"],
            game_count=2,
            max_plies=4,
        )

        self.assertEqual(evaluation.game_count, 2)
        self.assertEqual(evaluation.end_reasons, {"max_plies": 2})
        self.assertEqual(evaluation.illegal_move_count, 0)
        self.assertEqual(evaluation.draws, 2)

    def test_evaluates_external_engine_with_custom_go_command(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "for line in sys.stdin:\n"
                "    line = line.strip()\n"
                "    if line == 'usi': print('usiok', flush=True)\n"
                "    elif line == 'isready': print('readyok', flush=True)\n"
                "    elif line == 'go nodes 1': print('bestmove 7g7f', flush=True)\n"
                "    elif line.startswith('go'): print('bestmove resign', flush=True)\n"
                "    elif line == 'quit': break\n"
            ),
        ]

        evaluation = evaluate_player_against_usi_engine(
            UsiEngine(),
            command,
            game_count=1,
            max_plies=1,
            engine_go_command="go nodes 1",
        )

        self.assertEqual(evaluation.game_count, 1)
        self.assertEqual(evaluation.illegal_move_count, 0)
        self.assertEqual(evaluation.results[0].white_actor.settings["go_command"], "go nodes 1")

    def test_counts_player_wins_and_losses(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "for line in sys.stdin:\n"
                "    line = line.strip()\n"
                "    if line == 'usi': print('usiok', flush=True)\n"
                "    elif line == 'isready': print('readyok', flush=True)\n"
                "    elif line.startswith('go'): print('bestmove resign', flush=True)\n"
                "    elif line == 'quit': break\n"
            ),
        ]

        evaluation = evaluate_player_against_usi_engine(
            UsiEngine(),
            command,
            game_count=2,
            max_plies=4,
        )

        self.assertEqual(evaluation.player_wins, 2)
        self.assertEqual(evaluation.player_losses, 0)
        self.assertEqual(evaluation.draws, 0)
        self.assertEqual(evaluation.player_black_wins, 1)
        self.assertEqual(evaluation.player_white_wins, 1)


if __name__ == "__main__":
    unittest.main()
