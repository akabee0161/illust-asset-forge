# illust-asset-forge 引き継ぎ

## 進捗
作業: illust-asset-forge の作業フローの効率化
工程:
1. 設計（ノートブックで完結する流れ・顔の構図・画風の見比べ・複数キャラクター）（完了）
2. 実装（完了）
3. Colab で試す ← 今ここ
段階: 実装（8 タスク中 8）
次にやること:
- 依頼者が Colab で試す（階層分け・画風 soft-cel・その場で書く画風・セラの仮設定を含む）
- thick-line-flat を軸にした生成プロンプトの洗練（依頼者の希望）
- お試しの変更を main に入れるかは依頼者が決める

## 概要

ゲーム用のキャラクターのイラストを Colab の GPU で生成する汎用のリポジトリ。設計は `docs/superpowers/specs/2026-11-21-illust-asset-forge-design.md`。使い方は `README.md`、作業規約は `CLAUDE.md`。

## 状態

- GPU なしで動く部分（設定・プロンプト・PNG の条件・マスク・書き出し・`forge.py` の偽パイプラインでのテスト）は pytest で確かめ済み
- ブランチ全体の最終レビュー済み（初期実装の分）。Critical 1・Important 4・Minor 7 をすべて直した
- Colab で、モデルの読み込み・顔の生成・inpaint（表情）・全身の生成が動くことを、依頼者が確かめた（2026-11-21）。1 枚あたり、顔・全身は約 20 秒、表情は約 11 秒（T4）
- `requirements-colab.txt` の版は、diffusers 0.42.1・transformers 5.22.2・accelerate 1.15.0・safetensors 0.8.0。Colab で動いた
- ノートブックで採用・範囲決め・画風の見比べ・zip 化まで済ませ、Claude は `python -m artgen.intake <zip>` で取り込む流れに直した（`selection`・`picker`・`intake` を追加、`forge.load()` は `F.load_pipeline()` と `F.attach()` に分けた）。ノートブックと `picker` の画面は、偽のウィジェット・パイプラインでしか確かめていない（サーバーに GPU も ipywidgets の実行環境も無い）
- 型 face（左向き・頭から胸まで）の初期値・既定の範囲（`inpaint.box`）・画風の候補は、Colab で試して調整が要る
- 候補の置き場は `runs/<id>/<種類>/<日時>/` になった。古い `runs/<日時>/` のフォルダは `P.pick(forge, run_id="<日時>")` だけで開ける（char・kind では絞れない）
- 画風 soft-cel を足した。画風はその場で辞書に書いても使える（`style=` と `compare_styles(styles=[...])`）
- ally06（セラ）の `character.yaml` は仮の設定。外見は役割とテーマの色から推測で書いたので、依頼者が直してよい
- ボルグ（`characters/ally05/`）: 顔（シード 1630744298）・表情 `hurt`（シード 135213639）・全身（シード 4074111561）を採用して書き出し済み。`inpaint_box` は `[440, 170, 820, 600]`
- 全身の髪の色・髪型・鎧の色は、顔に近づけるタグを `character.yaml` に入れた。髪型は、顔が後ろへ流した形、全身は立ち上がった短髪で、合っていない。鎧は、顔が青みの少ない灰色、全身は青みのある鋼色と金色の装飾が多く、合っていない。依頼者は「概ね合っている」と判断して採用した
