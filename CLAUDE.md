# illust-asset-forge 作業規約（Claude 向け）

人間向けの手引きは [`README.md`](README.md)。ここには Claude が守ることだけを書く。

## 役割の分担

- GPU を使う処理（`forge.load()`・生成・inpaint）は**依頼者が Colab で動かす**。このサーバーには GPU が無い。Claude は偽のパイプラインを使ったテストで確かめる
- Colab は GitHub からリポジトリを取るので、設定の変更は push するまで Colab に届かない。**push は依頼者が頼んだときだけ**行う

## 受け取った画像の扱い

- 候補（生成された画像）はリポジトリに入れない。入れるのは `characters/<id>/source/` の採用した画像と、書き出した画像・`params.json` だけ
- 受け取った PNG は `source/` に `face.png`・`face_<表情>.png`・`fullbody.png` の名前で置き、**必ず `python -m artgen.pnginfo` で条件を確かめる**。条件が無ければ添付で消えた可能性があるので、依頼者に知らせて止める
- 条件の `overrides` は、`character.yaml` などの設定に戻してからコミットする（以後は上書きなしで同じプロンプトになるように）
- `inpaint_box` は、採用した顔の画像を見て決める。顔の自動検出は入れない
- 書き出しは `python -m artgen.export <id>`

## 守ること

- リポジトリにゲーム側の事情（ファイル名・画面の大きさの定数・JSON の形）や、ゲームのリポジトリの名前を書かない。キャラクターの ID はただの名前
- diffusers・torch は `forge.load()` の中でだけ import する。ほかのモジュールとテストはそれらが無くても動くままにする
- 設定の誤りは `ConfigError` で、`<相対パス>: <キーの位置>: <何がおかしいか>` の形で止める
- コミットは Conventional Commits + 日本語の要約

## テスト

```sh
uv venv .venv && uv pip install -e '.[dev]' && .venv/bin/pytest
```

## ドキュメントの扱い

- `docs/superpowers/` 配下（設計書・計画）は作成時点のログ。後から実装が変わっても遡って更新しない
- 最新の状態の正典は `README.md` と `CLAUDE.md`。両者が食い違うときは正典が正しい
