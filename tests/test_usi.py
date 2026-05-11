import unittest

from shogi_arena_agent.board_backend import legal_move_usis
from shogi_arena_agent.usi import UsiEngine, UsiPosition, board_from_position, run_usi_loop


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

    def test_default_policy_returns_legal_move_after_moves(self) -> None:
        position = UsiPosition(command="position startpos moves 7g7f")
        engine = UsiEngine()

        response = engine.handle_line(position.command)
        bestmove_response = engine.handle_line("go btime 0 wtime 0")

        self.assertEqual(response, [])
        self.assertEqual(len(bestmove_response), 1)
        bestmove = bestmove_response[0].removeprefix("bestmove ")
        legal_moves = {move.usi() for move in board_from_position(position).legal_moves}
        self.assertIn(bestmove, legal_moves)

    def test_board_from_sfen_position_with_moves(self) -> None:
        position = UsiPosition(
            command=(
                "position sfen "
                "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 "
                "moves 7g7f"
            )
        )

        board = board_from_position(position)

        self.assertIn("3c3d", {move.usi() for move in board.legal_moves})

    def test_cshogi_board_from_position_matches_python_shogi_legal_moves(self) -> None:
        positions = [
            UsiPosition(),
            UsiPosition(command="position startpos moves 7g7f 3c3d 2g2f"),
            UsiPosition(command="position startpos moves 2g2f 8c8d 2f2e 8d8e 2e2d 8e8f 2d2c+"),
            UsiPosition(command="position startpos moves 7g7f 3c3d 8h2b+ 3a2b B*4e"),
            UsiPosition(command="position startpos moves 7g7f 3c3d 8h2b+ 3a2b B*5e"),
            UsiPosition(
                command=(
                    "position sfen "
                    "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2 "
                    "moves 3c3d 2g2f"
                )
            ),
        ]

        for position in positions:
            with self.subTest(position=position.command):
                python_board = board_from_position(position, backend="python-shogi")
                cshogi_board = board_from_position(position, backend="cshogi")

                self.assertEqual(python_board.sfen(), cshogi_board.sfen())
                self.assertEqual(legal_move_usis(python_board), legal_move_usis(cshogi_board))

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
