# illust-asset-forge 引き継ぎ

## 進捗
作業: forge 本体に切り出しと背景消しを入れる
工程:
1. 設計（docs/superpowers/specs/2026-11-24-cutout-design.md）（完了）
2. 実装計画（docs/superpowers/plans/2026-11-24-cutout.md）（完了）
3. 実装と、一時フォルダでの本物の Modal での確かめ（完了）
4. コピー先の UPSTREAM.md の該当の行を「戻した」に直す（コピー先の PR）（完了）
段階: 完了
次にやること:
- なし

以後このリポジトリは、各作品のリポジトリへ丸ごとコピーして使う（作品ごとの作り込みはコピーの側で行う。`README.md` 8）。

## 概要

ゲーム用のキャラクターのイラストを Colab か Modal（T4）の GPU で生成する汎用のリポジトリ。Modal の道（`artgen.cloud`）は、最初のコピー先で作ったものを持ち帰った（コピー先の顔の切り出し `face_crop`・`crop_review` は、テンプレートごとの切り出しと背景消し `cutout`・`cutout_review`・`matte` に広げて、持ち帰った。README 11）。設計は `docs/superpowers/specs/2026-11-21-illust-asset-forge-design.md`。使い方は `README.md`、作業規約は `CLAUDE.md`。

## 状態

- Modal の道（`python -m artgen.cloud`）を PR #6 で入れた。main（`fc4e601`）を新しく clone し、README の節 10 の手順で騎士の顔 2 枚を T4 で作れた（2026-11-24）。シード 3034254740 は、前回 Modal で作った絵と画素まで一致した（Colab の絵とは平均差約 4 で一致しない。README に記載）。Modal の Volume `illust-asset-forge-models` にモデルが入っている
- GPU なしで動く部分（設定・プロンプト・PNG の条件・マスク・書き出し・`forge.py` の偽パイプラインでのテスト）は pytest で確かめ済み
- ブランチ全体の最終レビュー済み（初期実装の分）。Critical 1・Important 4・Minor 7 をすべて直した
- Colab で、モデルの読み込み・顔の生成・inpaint（表情）・全身の生成が動くことを、依頼者が確かめた（2026-11-21）。1 枚あたり、顔・全身は約 20 秒、表情は約 11 秒（T4）
- `requirements-colab.txt` の版は、diffusers 0.42.1・transformers 5.22.2・accelerate 1.15.0・safetensors 0.8.0。Colab で動いた
- ノートブックで採用・範囲決め・画風の見比べ・zip 化まで済ませ、Claude は `python -m artgen.intake <zip>` で取り込む流れに直した（`selection`・`picker`・`intake` を追加、`forge.load()` は `F.load_pipeline()` と `F.attach()` に分けた）。ノートブックと `picker` の画面は、偽のウィジェット・パイプラインでしか確かめていない（サーバーに GPU も ipywidgets の実行環境も無い）
- 型 face（左向き・頭から胸まで）の初期値・既定の範囲（`inpaint.box`）・画風の候補は、Colab で試して調整が要る
- 候補の置き場は `runs/<id>/<種類>/<日時>/` になった。古い `runs/<日時>/` のフォルダは `P.pick(forge, run_id="<日時>")` だけで開ける（char・kind では絞れない）
- 画風 soft-cel を足した。画風はその場で辞書に書いても使える（`style=` と `compare_styles(styles=[...])`）
- 画風 thick-line-flat を、Colab の見比べの結果で洗練した（設計は `docs/superpowers/specs/2026-11-22-thick-line-flat-refine-design.md`）。元の定義に、髪の描き込み（`messy hair` など 5 つ）と鮮やかすぎる色（`colorful` など 3 つ）を避けるネガティブを足した。色の指定で服の色が設定より薄くなることがあるが、依頼者は許容した
- 画風は全キャラクター共通のデフォルト（`defaults.yaml` の `style`、今は thick-line-flat）を使う。`character.yaml` の `style` は、1 人だけ別の画風にするときだけ書く（設計は `docs/superpowers/specs/2026-11-22-default-style-design.md`）。見本 4 人とも `style` を書かず、デフォルトに従う
- 生成は `lpw_stable_diffusion_xl` を使うので、75 トークンを超えるプロンプトも切り詰められずに届き、`(tag:1.3)` の重みも効くと判断している（lpw のソースは未確認。標準の SDXL パイプラインは 77 トークンで切り詰める）
- ノートブックの準備のセルに `REPO_URL`・`REPO_BRANCH`・`SUBDIR`・`DRIVE_DIR` の 4 つの変数を置いた。コピーした先では、この 4 つを書き換える（`README.md` 8）。Colab でコピーした先から動くことは、まだ確かめていない
- キャラクターは汎用の見本 4 人（`knight`・`mage`・`schoolboy`・`schoolgirl`）だけ。見本の絵は 8 枚（4 人 × 顔・全身）を Colab（T4）で作って取り込み、書き出し済み。騎士・魔法使いの 4 枚は、画風を洗練したときに選んだ絵と、同じシード・同じ GPU で画素まで一致した（顔は左右反転して比べた）。制服の 2 人は候補から選んだ（シードは `characters/<id>/params.json`）
- 候補を選ぶ画面（`P.pick`）の画像が、出力欄の幅まで引き伸ばされて大きすぎた。縮小した画像の実寸を渡すように直した（Colab での見え方は未確認）。`edit_box`・`show_styles` の画像は同じ作りで、直していない
