import unittest

from shogi_arena_agent.usi import UsiEngine, UsiPosition, run_usi_loop


class FixedPolicy:
    def __init__(self) -> None:
        self.positions: list[UsiPosition] = []

    def select_move(self, position: UsiPosition) -> str:
        self.positions.append(position)
        return "2g2f"


class UsiEngineTest(unittest.TestCase):
    def test_usi_handshake(self) -> None:
        engine = UsiEngine(name="test-engine", author="tester")

        self.assertEqual(
            engine.handle_line("usi"),
            [
                "id name test-engine",
                "id author tester",
                "usiok",
            ],
        )
        self.assertEqual(engine.handle_line("isready"), ["readyok"])

    def test_go_returns_policy_move_for_current_position(self) -> None:
        policy = FixedPolicy()
        engine = UsiEngine(policy=policy)

        engine.handle_line("position startpos moves 7g7f")
        response = engine.handle_line("go btime 0 wtime 0")

        self.assertEqual(response, ["bestmove 2g2f"])
        self.assertEqual(policy.positions, [UsiPosition(command="position startpos moves 7g7f")])

    def test_run_loop_stops_on_quit(self) -> None:
        output = run_usi_loop(["usi\n", "isready\n", "quit\n", "usi\n"])

        self.assertEqual(
            output,
            [
                "id name shogi-arena-agent",
                "id author intrep",
                "usiok",
                "readyok",
            ],
        )


if __name__ == "__main__":
    unittest.main()
