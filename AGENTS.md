# Project Agent Policy

このプロジェクトでは日本語で応答し、コードコメントは英語で書く。

## Purpose

`shogi-arena-agent` は将棋アリーナ接続用の runtime repository である。
モデル研究や学習本体は `intelligence-representation` 側に置く。

## Boundaries

- このリポジトリは USI / Lishogi / Floodgate / engine process / deployment を扱ってよい。
- `intelligence-representation` はこのリポジトリを知らない前提を守る。
- 依存方向は `shogi-arena-agent -> intelligence-representation` のみ許可する。
- 学習済みモデルは checkpoint または軽量 inference API として受け取る。
- アリーナ固有の token / secret / account 情報はコミットしない。

## Development Rules

- 小さく頻繁にコミットする。
- 実装後は関連テストを実行する。
- まだ外部サービス接続を自動化しない。まずローカル USI とローカル対局を優先する。
- Computer-shogi CLI defaults should use `max-plies=320`; shorter overrides are allowed only with a warning.
- ShogiGameRecord schema is mirrored with `../intelligence-representation`; update both repositories' read/write/tests together when changing it.
