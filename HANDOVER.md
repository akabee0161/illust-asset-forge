# illust-asset-forge 引き継ぎ

## 進捗
作業: illust-asset-forge の初期実装
工程:
1. 設計（完了）
2. 実装（設定・プロンプト・書き出し・Colab の入口）（完了）
3. ボルグ（ally05）で顔・表情・全身絵を試す ← 今ここ
段階: 実装（7 タスク中 7）
次にやること:
- 依頼者がボルグの設定の下書き（`characters/ally05/character.yaml`）を確認する
- GitHub にリポジトリを作って push する（依頼者が頼んだら）
- 依頼者が Colab で準備・顔のセルを試す（表情は `inpaint_box` が要るので顔を採用した後）

## 概要

ゲーム用のキャラクターのイラストを Colab の GPU で生成する汎用のリポジトリ。設計は `docs/superpowers/specs/2026-11-21-illust-asset-forge-design.md`。使い方は `README.md`、作業規約は `CLAUDE.md`。

## 状態

- GPU なしで動く部分（設定・プロンプト・PNG の条件・マスク・書き出し・`forge.py` の偽パイプラインでのテスト）は pytest で確かめ済み
- ブランチ全体の最終レビュー済み。Critical 1・Important 4・Minor 7 をすべて直した（23790c5）
- **GPU を使う部分（`forge.load()` と実際の生成・inpaint）は未確認。** 依頼者が Colab で動かす
- `requirements-colab.txt` の版（diffusers 0.42.1・transformers 5.22.2・accelerate 1.15.0・safetensors 0.8.0）は、`uv pip install --dry-run --exclude-newer 2026-11-07` で解決できた組み合わせ。Colab 上で互いに動くか、`lpw_stable_diffusion_xl` が `v0.42.1` の版で取れるかは未確認
- `characters/ally05/character.yaml` の `inpaint_box` は null。顔を採用した後に書く

## 保留している課題

- **Colab で画像のパスを渡す手間（2026-11-21 に依頼者が保留）**: 表情・全身のセルへ進むたびに、Drive にある採用した画像のパスを手で書き換える必要がある。スマホでは Drive の画面でパスをコピーできず、Colab のファイルのパネルも使いにくい。効率的にする方法は未定で、あとで検討する
