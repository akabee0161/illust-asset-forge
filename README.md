# illust-asset-forge

ゲーム用のキャラクターのイラスト（顔・表情の差分・全身絵）を、Google Colab の GPU で生成するリポジトリ。

- キャラクターの設定を YAML で持ち、画風・絵の種類（型）と組み合わせてプロンプトを組み立てる
- 生成の条件（プロンプト・シード・版）を PNG に埋め込み、採用した絵を作り直せるようにする
- 採用した絵を、受け渡しの形（顔は 256×256、全身は元の大きさ、`params.json`）に書き出す

GPU を使う処理は依頼者が Colab で動かし、GPU を使わない処理（設定・プロンプト・書き出し）はサーバーで Claude が動かす。
この README は**人間向けの手引き**である。Claude が守る作業規約は [`CLAUDE.md`](CLAUDE.md) にある。

**目次**

1. [ファイルの地図](#1-ファイルの地図)
2. [Colab での使い方](#2-colab-での使い方)
3. [採用した絵の渡し方](#3-採用した絵の渡し方)
4. [Claude 側の流れ](#4-claude-側の流れ)
5. [初めて Colab で動かすときの確認](#5-初めて-colab-で動かすときの確認)
6. [ライセンスと再現の限界](#6-ライセンスと再現の限界)
7. [テストの実行](#7-テストの実行)

---

## 1. ファイルの地図

| ファイル | 内容 |
|---|---|
| `styles/<name>.yaml` | 画風（プロンプトとネガティブ） |
| `templates/face.yaml`・`fullbody.yaml` | 絵の種類（大きさ・構図のタグ・書き出しの大きさ） |
| `characters/<id>/character.yaml` | キャラクターの設定（外見・表情の一覧・描き直す範囲 `inpaint_box`） |
| `characters/<id>/source/` | 採用した元の画像（受け取ったもの） |
| `characters/<id>/face.png` など・`params.json` | 書き出した受け渡しの形 |
| `src/artgen/` | `config`（読み込み）・`prompt`（組み立て）・`pnginfo`（条件の読み書き）・`mask`（描き直す範囲）・`export`（書き出し）・`forge`（Colab 側） |
| `notebooks/run.ipynb` | Colab の入口 |
| `requirements-colab.txt` | Colab で入れるライブラリ（版を固定） |

## 2. Colab での使い方

1. `notebooks/run.ipynb` を Colab で開き、ランタイムを GPU（T4）にする
2. 上から順にセルを実行する
   - **準備**: Drive をつなぎ、リポジトリを clone（2 回目以降は pull）、ライブラリを入れ、モデルを読み込む
   - **顔**: `forge.generate("ally05", "face", n=8)` で 8 枚作って 1 枚選ぶ
   - **表情**: `forge.inpaint("ally05", "hurt", source=<採用した顔>, n=4)`。`inpaint_box` が要るので、顔を採用して Claude が書いた後に実行する
   - **全身**: `forge.generate("ally05", "fullbody", n=4)`
3. 候補は Drive の `illust-asset-forge/runs/<run_id>/` に `<id>_<型>[_<表情>]_<seed>.png` で保存される

試すときの引数（YAML は書き換えない）:

```python
forge.generate("ally05", "face", extra="scar on cheek", remove="stubble")  # タグを足す・消す
forge.generate("ally05", "face", extra_negative="hat", remove_negative="lowres")
forge.generate("ally05", "face", seeds=[123, 456])  # シードを固定（n は無視される）
forge.generate("ally05", "face", guidance=6, steps=30)
```

**YAML は Colab で書き換えない。** Colab はリポジトリを GitHub から取るので、書き換えは GitHub に戻らず、次の pull でぶつかる。上書きした内容は PNG の条件に `overrides` として残るので、採用した絵を受け取った Claude が設定に戻す。

## 3. 採用した絵の渡し方

Drive から採用した PNG を手元にダウンロードし、zip にしてチャットに添付する。zip にするのは、PNG の条件（シードなど）が添付で消えたり、画像が縮められたりするのを避けるため。

## 4. Claude 側の流れ

1. 受け取った PNG を `characters/<id>/source/` に `face.png`・`face_<表情>.png`・`fullbody.png` の名前で置く
2. `python -m artgen.pnginfo characters/<id>/source/*.png` で条件が残っているか確かめる（`overrides` は先頭に出る）
3. `overrides` を `character.yaml` などの設定に戻す
4. 採用した顔を見て、目と口を囲む `inpaint_box`（1024 の座標）を `character.yaml` に書く
5. `python -m artgen.export <id>` で書き出す（顔は 256×256 に縮め、`params.json` を作る）
6. push は依頼者が頼んだときだけ行う

## 5. 初めて Colab で動かすときの確認

GPU を使う部分は Claude が動かして確かめられない。次を 1 回ずつ実行して確かめる。

1. **準備のセル**: エラーなく終わり、最後に `forge` ができる（ライブラリの版の組み合わせ・コミュニティパイプラインの取得もここで確かめられる）
2. **顔のセル**: 8 枚が表示され、各ファイル名とシードが出る。Drive の `runs/` に PNG ができる。`python -m artgen.pnginfo` か Colab 上で PNG の条件を読めること
3. **表情のセル**: 顔を採用し、Claude が `inpaint_box` を書いて push した後に実行する。目と口のあたりだけが変わり、境目が目立たない
4. **全身のセル**: 832×1216 の全身絵が出る
5. 添付した PNG に条件が残っているか（Claude が `pnginfo` で確かめる）

## 6. ライセンスと再現の限界

- モデル `cagliostrolab/animagine-xl-4.0` は CreativeML Open RAIL++-M。ライセンスを出した側は生成物に権利を主張せず、商用利用も認められている。生成物の使い方の責任は使う側が負い、禁止用途には使えない。モデルを配らないので、ライセンスの写しを付ける決まりは関係しない（Hugging Face のモデルのページと SDXL の LICENSE.md、2026-11-21 取得。法律の専門家の判断ではない）
- 同じシードでも、GPU の種類やライブラリの版が変わると同じ絵にならないことがある。版を固定し条件を記録しても、完全な再現は保証できない。採用した元の画像を `source/` に残すのはそのため

## 7. テストの実行

```sh
uv venv .venv && uv pip install -e '.[dev]' && .venv/bin/pytest
```

`src/artgen/forge.py` の `load()` と `show()` 以外は torch・diffusers が無くても動く。
