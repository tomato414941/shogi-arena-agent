import unittest

from shogi_arena_agent.multipv_policy import (
    MultiPVPolicyTargetConfig,
    MultiPVPolicyTargetPlayer,
    parse_multipv_info_line,
    policy_targets_from_multipv_info,
)
from shogi_arena_agent.usi_process import UsiGoResult


class StubPlayer:
    def position(self, command: str) -> None:
        pass

    def go(self) -> UsiGoResult:
        return UsiGoResult(
            bestmove="7g7f",
            info_lines=(
                "info depth 4 multipv 1 score cp 100 pv 7g7f",
                "info depth 4 multipv 2 score cp 0 pv 2g2f",
            ),
        )


class MultiPVPolicyTest(unittest.TestCase):
    def test_parses_multipv_info_line(self) -> None:
        info = parse_multipv_info_line("info depth 4 multipv 2 score cp 10 pv 2g2f")

        self.assertIsNotNone(info)
        self.assertEqual(info.multipv, 2)
        self.assertEqual(info.score_cp, 10.0)
        self.assertEqual(info.move, "2g2f")

    def test_policy_targets_from_multipv_info(self) -> None:
        targets = policy_targets_from_multipv_info(
            (
                "info depth 4 multipv 1 score cp 100 pv 7g7f",
                "info depth 4 multipv 2 score cp 0 pv 2g2f",
            ),
            MultiPVPolicyTargetConfig(multipv=2, temperature_cp=100.0),
        )

        self.assertIsNotNone(targets)
        self.assertGreater(targets["7g7f"], targets["2g2f"])
        self.assertAlmostEqual(sum(targets.values()), 1.0)

    def test_policy_targets_convert_mate_score(self) -> None:
        targets = policy_targets_from_multipv_info(
            (
                "info multipv 1 score mate 3 pv 7g7f",
                "info multipv 2 score cp 900 pv 2g2f",
            ),
            MultiPVPolicyTargetConfig(multipv=2),
        )

        self.assertIsNotNone(targets)
        self.assertGreater(targets["7g7f"], 0.99)

    def test_rejects_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            MultiPVPolicyTargetConfig(multipv=0)
        with self.assertRaises(ValueError):
            MultiPVPolicyTargetConfig(multipv=1, temperature_cp=0.0)

    def test_wrapper_adds_policy_targets_to_go_result(self) -> None:
        player = MultiPVPolicyTargetPlayer(
            StubPlayer(),
            MultiPVPolicyTargetConfig(multipv=2, temperature_cp=100.0),
        )

        result = player.go()

        self.assertIsNotNone(result.policy_targets)
        self.assertGreater(result.policy_targets["7g7f"], result.policy_targets["2g2f"])


if __name__ == "__main__":
    unittest.main()
