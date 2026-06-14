import unittest

from shogi_arena_agent.mcts_evaluator import aligned_prior_values, normalized_prior_dict, normalized_prior_values


class MovePriorsTest(unittest.TestCase):
    def test_normalizes_mapping_priors_by_legal_move(self) -> None:
        priors = normalized_prior_dict(("7g7f", "2g2f", "5i6h"), {"7g7f": 2.0, "2g2f": -1.0})

        self.assertEqual(priors, {"7g7f": 1.0, "2g2f": 0.0, "5i6h": 0.0})

    def test_normalizes_aligned_sequence_priors(self) -> None:
        priors = normalized_prior_values(("7g7f", "2g2f"), (2.0, 1.0))

        self.assertEqual(priors, (2.0 / 3.0, 1.0 / 3.0))

    def test_uses_uniform_priors_when_total_is_zero(self) -> None:
        priors = normalized_prior_values(("7g7f", "2g2f"), (0.0, -1.0))

        self.assertEqual(priors, (0.5, 0.5))

    def test_aligned_sequence_can_validate_without_usi_moves(self) -> None:
        priors = normalized_prior_values((), (2.0, 1.0), expected_count=2)

        self.assertEqual(priors, (2.0 / 3.0, 1.0 / 3.0))

    def test_mapping_priors_require_usi_moves(self) -> None:
        with self.assertRaisesRegex(ValueError, "mapping priors require USI legal moves"):
            normalized_prior_values((), {"7g7f": 1.0}, expected_count=1)

    def test_aligned_values_can_preserve_negative_values_for_argmax(self) -> None:
        priors = aligned_prior_values(("7g7f", "2g2f"), {"7g7f": -1.0}, clamp_negative=False)

        self.assertEqual(priors, (-1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
