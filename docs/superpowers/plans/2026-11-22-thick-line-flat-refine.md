# thick-line-flat の洗練 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 依頼者と Colab で見比べを繰り返し、当たりの絵（ally06・シード 3034254740）の塗り方の雰囲気が安定して出る画風の定義を決め、`styles/` に入れる。

**Architecture:** コードは変えない。Claude がチャットに貼れるセル（`forge.compare_styles` にその場の画風の辞書を渡す）を出し、依頼者の見たままの感想で次の変種を作る。決まった定義だけを YAML にしてコミットする。

**Tech Stack:** 既存の `artgen`（`forge.compare_styles`・`picker.show_styles`・`config.load_style`）、pytest

**Spec:** `docs/superpowers/specs/2026-11-22-thick-line-flat-refine-design.md`

## Global Constraints

- 目の形は画風に入れない（`large eyes` など目の形・大きさのタグは使わない）。目の塗り方は塗り方の雰囲気に含めてよい
- 変種の名前は `tlf-r<回>-<a|b|c>`（半角英数字・`_`・`-` だけ）
- 毎回の比べる基準は `"thick-line-flat"`（YAML のまま）。当たりのシード 3034254740 を毎回入れる
- 1 回は、セラ（ally06）とボルグ（ally05）で、同じシード 3 つ・画風 4 つ（計 24 枚）
- 絵に疑問があっても Claude が勝手に調整しない。見たままを報告し、扱いを依頼者と合意してから次へ
- リポジトリにゲーム側のリポジトリの名前・ファイル名・定数を書かない
- push と PR は依頼者が頼んだときだけ。コミットは Conventional Commits + 日本語の要約

## Review Focus

- セルを貼ったときの誤り: 辞書の `name` が重複・使えない文字 → `compare_styles` が `ConfigError` で止まる。出す前に名前を目で確かめる
- 変種で目の形が変わってボルグに合わなくなる → ボルグの行も毎回見る
- 当たりのシードでも基準と変種で構図が変わる（タグの数でずれる）→ 構図の違いは評価から外すと依頼者に伝える
- 仕上げの YAML にカンマを含む重み付きタグがある → 引用符で囲む（README の決まり）

---

### Task 1: 顔の見比べを回す（繰り返し）

**Files:** なし（チャットだけ）

- [ ] **Step 1: 第 1 回のセルを出す。** 変種は、髪の描き込みを抑える・影の段数を絞る・色の鮮やかさを抑える、の 3 方向。形:

```python
SEEDS = [3034254740, <シード2>, <シード3>]
STYLES = ["thick-line-flat", {"name": "tlf-r1-a", "prompt": [...], "negative": [...]}, ...]
for c in ["ally06", "ally05"]:
    P.show_styles(forge.compare_styles(c, styles=STYLES, seeds=SEEDS))
```

- [ ] **Step 2: 依頼者の感想を待ち、次の回の変種を作る。** 依頼者が「これで顔は良い」と言うまで繰り返す

### Task 2: 全身で確かめる

- [ ] **Step 1:** 顔で決まった定義と基準で、`template="fullbody"` のセルを出す（セラ・ボルグ、シード 2 つ）
- [ ] **Step 2:** 依頼者が良いと言えば Task 3、問題があれば Task 1 に戻る

### Task 3: 定義を `styles/` に入れる

**Files:**
- Create or Modify: `styles/<名前>.yaml`（上書きか別名かは依頼者に聞く）
- Modify（別名のとき）: `tests/test_repo_configs.py:65` の画風名の一覧

- [ ] **Step 1:** 依頼者に、`thick-line-flat.yaml` を上書きするか別の名前にするかを聞く
- [ ] **Step 2:** 決まった定義を YAML に書く（`prompt:`・`negative:` の配列。既存の YAML と同じ形）。別名なら一覧のテストに足す
- [ ] **Step 3:** `.venv/bin/pytest` を実行し、全部通ることを確かめる
- [ ] **Step 4:** コミット `feat: 画風 <名前> を見比べの結果で洗練する`（上書きなら `feat: 画風 thick-line-flat を見比べの結果で洗練する`）

### Task 4: 引き継ぎを直す

- [ ] **Step 1:** `HANDOVER.md` の `## 進捗` を `~/.claude/rules/progress.md` の書式で今の状態に直し、状態の欄に決まった画風を 1 行足す
- [ ] **Step 2:** コミット `docs: 引き継ぎに画風の洗練の結果を書く`
