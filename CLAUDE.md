# illust-asset-forge 作業規約（Claude 向け）

人間向けの手引きは [`README.md`](README.md)。ここには Claude が守ることだけを書く。

## 役割の分担

- GPU を使う処理（`F.load_pipeline()`・生成・inpaint）は**依頼者が Colab で動かす**。このサーバーには GPU が無い。Claude は偽のパイプラインを使ったテストで確かめる。ノートブックの画面（`picker`）も、偽のウィジェットでしか確かめていない
- Colab は GitHub からリポジトリを取るので、設定の変更は push するまで Colab に届かない。**push は依頼者が頼んだときだけ**行う

## 受け取った画像の扱い

- 候補（生成された画像）はリポジトリに入れない。入れるのは `characters/<id>/source/` の採用した画像と、書き出した画像・`params.json` だけ
- 採用した絵は、依頼者がノートブックのまとめのセルで作った zip で届く。**`python -m artgen.intake <zip>` で `source/` に取り込む**（`face.png`・`face_<表情>.png`・`fullbody.png` の名前、PNG に条件があること、キャラクターの一致を検査し、通らなければ何も書かずに止まる）。条件が無いときは添付で消えた可能性があるので、依頼者に知らせて止める
- `intake` は `source/` を丸ごと置き換える。出力の「消した」に意図しないもの（特に `face.png`）があるときはコミットせず、依頼者に確かめる
- `intake` の出力にある書き戻しの一覧を、`character.yaml` などの設定に戻してからコミットする（以後は上書きなしで同じプロンプトになるように）。条件の `overrides` の `style` は `character.yaml` の `style` に戻す
- 表情の範囲（`inpaint_box`）は、依頼者がノートブックで決め、`selection.json` で届く。Claude は `intake` の一覧を見て `character.yaml` に書き戻す。顔の自動検出は入れず、Claude が顔を見て範囲を決めることもしない
- 書き出しは `python -m artgen.export <id>`

## 守ること

- リポジトリにゲーム側の事情（ファイル名・画面の大きさの定数・JSON の形）や、ゲームのリポジトリの名前を書かない。キャラクターの ID はただの名前
- diffusers・torch は `forge.load_pipeline()`・`forge.attach()` の中でだけ import する。`ipywidgets`・`IPython` も関数の中でだけ import する。ほかのモジュールとテストはそれらが無くても動くままにする
- 取り込みの関数（`install_selection`）は、将来 Colab から直接コミットするときに使うので、zip・標準出力に依らないままにする（zip の展開は `extract_pack`、表示は `main` が受け持つ）
- 設定の誤りは `ConfigError` で、`<相対パス>: <キーの位置>: <何がおかしいか>` の形で止める
- コミットは Conventional Commits + 日本語の要約

## テスト

```sh
uv venv .venv && uv pip install -e '.[dev]' && .venv/bin/pytest
```

## ドキュメントの扱い

- `docs/superpowers/` 配下（設計書・計画）は作成時点のログ。後から実装が変わっても遡って更新しない
- 最新の状態の正典は `README.md` と `CLAUDE.md`。両者が食い違うときは正典が正しい
