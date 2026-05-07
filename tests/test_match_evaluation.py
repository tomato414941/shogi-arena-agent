import unittest

from shogi_arena_agent.match_evaluation import summarize_match_results
from shogi_arena_agent.shogi_game import play_shogi_game
from shogi_arena_agent.usi import UsiEngine


class MatchEvaluationTest(unittest.TestCase):
    def test_summarizes_player_results_by_side(self) -> None:
        results = [
            play_shogi_game(black=UsiEngine(), white=UsiEngine(), max_plies=2),
            play_shogi_game(black=UsiEngine(), white=UsiEngine(), max_plies=2),
        ]

        evaluation = summarize_match_results(results, ["black", "white"])

        self.assertEqual(evaluation.game_count, 2)
        self.assertEqual(evaluation.black_game_count, 1)
        self.assertEqual(evaluation.white_game_count, 1)
        self.assertEqual(evaluation.end_reasons, {"max_plies": 2})
        self.assertEqual(evaluation.average_plies, 2.0)
        self.assertEqual(evaluation.illegal_move_count, 0)
        self.assertEqual(evaluation.player_wins, 0)
        self.assertEqual(evaluation.player_losses, 0)
        self.assertEqual(evaluation.draws, 2)
        self.assertEqual(len(evaluation.results), 2)


if __name__ == "__main__":
    unittest.main()
