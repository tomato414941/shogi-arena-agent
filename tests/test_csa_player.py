from __future__ import annotations

import unittest

import shogi

from shogi_arena_agent.csa_player import run_csa_player
from shogi_arena_agent.usi import UsiEngine, UsiPosition


class FixedPolicy:
    def __init__(self, moves: tuple[str, ...]) -> None:
        self.moves = list(moves)
        self.positions: list[UsiPosition] = []

    def select_move(self, position: UsiPosition) -> str:
        self.positions.append(position)
        return self.moves.pop(0)


CsaServerMessage = tuple[int | None, str | None, int | None, int | None]


class FakeProtocol:
    def __init__(
        self,
        *,
        initial_sfen: str | None = None,
        server_messages: tuple[CsaServerMessage, ...] = (),
    ) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.initial_sfen = initial_sfen or shogi.Board().sfen()
        self.server_messages = list(server_messages)

    def open(self, host: str, port: int = 0) -> None:
        self.calls.append(("open", host, port))

    def login_ex(self, username: str, password: str) -> None:
        self.calls.append(("login_ex", username, password))

    def wait_match(self, block: bool = True) -> dict[str, object]:
        self.calls.append(("wait_match", block))
        return {"summary": {"sfen": self.initial_sfen}, "my_color": shogi.BLACK}

    def agree(self) -> None:
        self.calls.append(("agree",))

    def wait_server_message(
        self,
        board: shogi.Board,
        block: bool = True,
    ) -> CsaServerMessage:
        self.calls.append(("wait_server_message", block))
        return self.server_messages.pop(0)

    def move(self, piece_type: int, color: int, move: shogi.Move) -> None:
        self.calls.append(("move", piece_type, color, move.usi()))

    def resign(self) -> None:
        self.calls.append(("resign",))

    def logout(self) -> None:
        self.calls.append(("logout",))


class CsaPlayerTest(unittest.TestCase):
    def test_runs_one_csa_game_until_resign(self) -> None:
        protocol = FakeProtocol(server_messages=((shogi.WHITE, "3c3d", 1, None),))
        policy = FixedPolicy(("7g7f", "resign"))

        results = run_csa_player(
            protocol=protocol,
            player=UsiEngine(policy=policy),
            host="example.test",
            port=4081,
            username="user",
            password="pass",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].moves_played, 2)
        self.assertIn(("open", "example.test", 4081), protocol.calls)
        self.assertIn(("login_ex", "user", "pass"), protocol.calls)
        self.assertIn(("agree",), protocol.calls)
        self.assertIn(("move", shogi.PAWN, shogi.BLACK, "7g7f"), protocol.calls)
        self.assertIn(("resign",), protocol.calls)
        self.assertEqual(protocol.calls[-1], ("logout",))

    def test_promoting_move_sends_promoted_csa_piece_type(self) -> None:
        board = shogi.Board()
        for move_usi in ("2g2f", "8c8d", "2f2e", "8d8e", "2e2d", "8e8f"):
            board.push_usi(move_usi)
        protocol = FakeProtocol(
            initial_sfen=board.sfen(),
            server_messages=((None, None, None, 1),),
        )
        policy = FixedPolicy(("2d2c+",))

        results = run_csa_player(
            protocol=protocol,
            player=UsiEngine(policy=policy),
            host="example.test",
            port=4081,
            username="user",
            password="pass",
        )

        self.assertEqual(len(results), 1)
        self.assertIn(
            ("move", shogi.PROM_PAWN, shogi.BLACK, "2d2c+"),
            protocol.calls,
        )


if __name__ == "__main__":
    unittest.main()
