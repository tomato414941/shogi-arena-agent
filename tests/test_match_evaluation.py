import unittest

from shogi_arena_agent.match_evaluation import evaluate_player_against_baseline
from shogi_arena_agent.usi import UsiEngine


class MatchEvaluationTest(unittest.TestCase):
    def test_evaluates_player_on_both_sides(self) -> None:
        evaluation = evaluate_player_against_baseline(UsiEngine(), game_count=2, max_plies=4)

        self.assertEqual(evaluation.game_count, 2)
        self.assertEqual(evaluation.black_game_count, 1)
        self.assertEqual(evaluation.white_game_count, 1)
        self.assertEqual(evaluation.end_reasons, {"max_plies": 2})
        self.assertEqual(evaluation.average_plies, 4.0)
        self.assertEqual(evaluation.illegal_move_count, 0)
        self.assertEqual(len(evaluation.results), 2)


if __name__ == "__main__":
    unittest.main()
