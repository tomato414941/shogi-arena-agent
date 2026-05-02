import unittest

from shogi_arena_agent.local_match import play_local_match
from shogi_arena_agent.model_policy import RankedMovePolicy
from shogi_arena_agent.usi import UsiEngine, UsiPosition, board_from_position


class RankedMovePolicyTest(unittest.TestCase):
    def test_returns_highest_scored_legal_move(self) -> None:
        position = UsiPosition(command="position startpos")
        legal_moves = tuple(sorted(move.usi() for move in board_from_position(position).legal_moves))
        preferred_move = legal_moves[-1]

        def rank_moves(position_sfen: str, candidate_moves: tuple[str, ...]) -> list[float]:
            self.assertTrue(position_sfen)
            return [1.0 if move == preferred_move else 0.0 for move in candidate_moves]

        move = RankedMovePolicy(rank_moves).select_move(position)

        self.assertEqual(move, preferred_move)

    def test_ranked_policy_can_play_local_match(self) -> None:
        def rank_first(_position_sfen: str, candidate_moves: tuple[str, ...]) -> list[float]:
            return [float(len(candidate_moves) - index) for index, _move in enumerate(candidate_moves)]

        result = play_local_match(
            black=UsiEngine(policy=RankedMovePolicy(rank_first)),
            white=UsiEngine(),
            max_plies=4,
        )

        self.assertEqual(result.end_reason, "max_plies")
        self.assertEqual(len(result.moves), 4)


if __name__ == "__main__":
    unittest.main()
