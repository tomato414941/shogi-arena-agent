from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import shogi
import shogi.CSA

from shogi_arena_agent.shogi_game import InProcessShogiPlayer, ShogiPlayer
from shogi_arena_agent.usi import RESIGN_MOVE, UsiEngine
from shogi_arena_agent.usi_process import UsiGoResult


class CsaProtocol(Protocol):
    def open(self, host: str, port: int = 0) -> object:
        ...

    def login_ex(self, username: str, password: str) -> object:
        ...

    def wait_match(self, block: bool = True) -> dict[str, object] | None:
        ...

    def agree(self) -> object:
        ...

    def wait_server_message(
        self, board: shogi.Board, block: bool = True
    ) -> tuple[int | None, str | None, int | None, int | None] | None:
        ...

    def move(self, piece_type: int, color: int, move: shogi.Move) -> object:
        ...

    def resign(self) -> object:
        ...

    def logout(self) -> object:
        ...


@dataclass(frozen=True)
class CsaGameResult:
    game_count: int
    end_message: int | None
    moves_played: int


def run_csa_player(
    *,
    protocol: CsaProtocol,
    player: ShogiPlayer | UsiEngine,
    host: str,
    port: int,
    username: str,
    password: str,
    games: int = 1,
) -> tuple[CsaGameResult, ...]:
    if games < 1:
        raise ValueError("games must be positive")

    protocol.open(host, port)
    protocol.login_ex(username, password)
    results: list[CsaGameResult] = []
    try:
        for game_index in range(games):
            match = protocol.wait_match(block=True)
            if match is None:
                break
            results.append(
                _play_csa_match(
                    protocol=protocol,
                    player=player,
                    match=match,
                    game_count=game_index + 1,
                )
            )
    finally:
        protocol.logout()
    return tuple(results)


def _play_csa_match(
    *,
    protocol: CsaProtocol,
    player: ShogiPlayer | UsiEngine,
    match: dict[str, object],
    game_count: int,
) -> CsaGameResult:
    summary = _summary(match)
    board = shogi.Board(str(summary["sfen"]))
    my_color = int(match["my_color"])
    active_player = _as_player(player)
    _new_game(player)
    protocol.agree()

    moves_played = 0
    while not board.is_game_over():
        if board.turn == my_color:
            move_usi = _bestmove(active_player, board)
            if move_usi == RESIGN_MOVE:
                protocol.resign()
                return CsaGameResult(
                    game_count=game_count,
                    end_message=None,
                    moves_played=moves_played,
                )
            move = shogi.Move.from_usi(move_usi)
            if move not in board.legal_moves:
                protocol.resign()
                return CsaGameResult(
                    game_count=game_count,
                    end_message=None,
                    moves_played=moves_played,
                )
            piece_type = _moving_piece_type(board, move)
            protocol.move(piece_type, board.turn, move)
            board.push(move)
            moves_played += 1
            continue

        message = protocol.wait_server_message(board, block=True)
        if message is None:
            return CsaGameResult(
                game_count=game_count,
                end_message=None,
                moves_played=moves_played,
            )
        _color, opponent_move_usi, _time, end_message = message
        if end_message is not None:
            return CsaGameResult(
                game_count=game_count,
                end_message=end_message,
                moves_played=moves_played,
            )
        if opponent_move_usi is not None:
            board.push_usi(opponent_move_usi)
            moves_played += 1

    return CsaGameResult(
        game_count=game_count,
        end_message=None,
        moves_played=moves_played,
    )


class PythonShogiCsaProtocol(shogi.CSA.TCPProtocol):
    def __init__(self, *, game_command: str | None = None, extended_login: bool = False) -> None:
        super().__init__()
        self.game_command = game_command
        self.extended_login = extended_login

    def login_ex(self, username: str, password: str) -> object:
        if self.extended_login:
            result = super().login_ex(username, password)
        else:
            result = self.login(username, password)
        if self.game_command is not None:
            self.write(self.game_command + "\n")
        return result

    def read_game_summary(self, block: bool = True) -> str | None:
        lines: list[str] = []
        while True:
            line = self.read_line(block)
            if line is None:
                return None
            if line == "BEGIN Game_Summary":
                lines.append(line)
                break

        while True:
            line = self.read_line(True)
            if line is None:
                return None
            lines.append(line)
            if line == "END Game_Summary":
                return "\n".join(lines) + "\n"


def new_python_shogi_csa_protocol(
    *,
    game_command: str | None = None,
    extended_login: bool = False,
) -> CsaProtocol:
    return PythonShogiCsaProtocol(game_command=game_command, extended_login=extended_login)


def _as_player(player: ShogiPlayer | UsiEngine) -> ShogiPlayer:
    if isinstance(player, UsiEngine):
        return InProcessShogiPlayer(player)
    return player


def _new_game(player: ShogiPlayer | UsiEngine) -> None:
    if isinstance(player, UsiEngine):
        player.new_game()


def _summary(match: dict[str, object]) -> dict[str, object]:
    summary = match.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("CSA match must include a summary")
    if "sfen" not in summary:
        raise ValueError("CSA match summary must include sfen")
    return summary


def _bestmove(player: ShogiPlayer, board: shogi.Board) -> str:
    player.position(f"position sfen {board.sfen()}")
    result = player.go()
    if isinstance(result, UsiGoResult):
        return result.bestmove
    return result


def _moving_piece_type(board: shogi.Board, move: shogi.Move) -> int:
    if move.drop_piece_type is not None:
        return move.drop_piece_type
    piece_type = board.piece_type_at(move.from_square)
    if piece_type is None:
        raise ValueError(f"move has no moving piece: {move.usi()}")
    if move.promotion:
        promoted_piece_type = shogi.PIECE_PROMOTED[piece_type]
        if promoted_piece_type is None:
            raise ValueError(f"move cannot promote moving piece: {move.usi()}")
        return promoted_piece_type
    return piece_type
