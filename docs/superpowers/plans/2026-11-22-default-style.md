# 共通のデフォルトの画風 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `defaults.yaml` の `style` を全キャラクターのデフォルトにし、`character.yaml` の `style:` を省略できるようにする。

**Architecture:** `load_character` が `style:` の無いときに `defaults.yaml` を読んで `Character.style` に入れる。`Character.style` の意味（そのキャラクターが使う画風名）は変わらないので、`forge`・`intake` は変えない。

**Tech Stack:** Python、PyYAML、pytest

**Spec:** `docs/superpowers/specs/2026-11-22-default-style-design.md`

## Global Constraints

- 設定の誤りは `ConfigError` で、`<相対パス>: <キーの位置>: <何がおかしいか>` の形で止める
- `defaults.yaml` はリポジトリ直下。中身は `style: thick-line-flat`
- `character.yaml` に `style:` があればそれが優先する
- コミットは Conventional Commits + 日本語の要約。push はしない

## Review Focus

- `style:` を省略し `defaults.yaml` も無い → `defaults.yaml: ファイルが無い` で止まる（Task 1 のテスト）
- `defaults.yaml` の `style` が空文字・リスト → `defaults.yaml: style: ...` で止まる（Task 1 のテスト）
- `character.yaml` に `style: null` → 今の `r.string` と同じく誤りで止める（省略とは扱わない）（Task 1 のテスト）
- `defaults.yaml` の画風が `styles/` に無い → リポジトリの設定のテストで落とす（Task 2）

---

### Task 1: `load_character` でデフォルトを当てはめる

**Files:**
- Modify: `src/artgen/config.py`（`load_character`）
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_default_style(root: Path) -> str`（`defaults.yaml` の `style`）。`load_character` は `style:` が無いときにこれを使う。`Character.style: str` は変えない

- [ ] **Step 1: 失敗するテストを書く**（`write(tmp_path, "defaults.yaml", ...)` と `write_character` を使う）
  - `test_character_without_style_uses_default`: `style:` 行を消した設定 + `defaults.yaml` に `style: flat` → `load_character(...).style == "flat"`
  - `test_character_style_overrides_default`: `style: color-anime` のまま + `defaults.yaml` に `style: flat` → `"color-anime"`
  - `test_character_without_style_and_no_defaults_fails`: `ConfigError`、文に `defaults.yaml` と `ファイルが無い`
  - `test_defaults_style_must_be_string`（`style: ""`・`style: [a]`・キー無し）: `ConfigError`、文が `defaults.yaml: style:` で始まる
  - `test_character_style_null_is_rejected`: `style: null` → `ConfigError`、文に `characters/c1/character.yaml: style:`
  - 既存の `test_missing_required_key` は、消すキーを `name` に変える
- [ ] **Step 2:** `.venv/bin/pytest tests/test_config.py -q` で新しいテストが落ちることを確かめる
- [ ] **Step 3:** `load_default_style` を足し、`load_character` で `"style" in r.data` なら `r.string("style")`、無ければ `load_default_style(root)` にする。`defaults.yaml` は `_Reader(root, "defaults.yaml").string("style")` で読む
- [ ] **Step 4:** `.venv/bin/pytest -q` で全部通ることを確かめる
- [ ] **Step 5:** コミット `feat: character.yaml の style を省略したら defaults.yaml の画風を使う`

### Task 2: リポジトリの設定とドキュメントを直す

**Files:**
- Create: `defaults.yaml`
- Modify: `characters/ally05/character.yaml`・`characters/ally06/character.yaml`（`style:` の行を消す）
- Modify: `tests/test_repo_configs.py`、`README.md`、`CLAUDE.md`、`notebooks/run.ipynb`（`STYLE = None` の説明）

- [ ] **Step 1: 失敗するテストを書く**: `test_default_style_exists`（`load_style(ROOT, load_default_style(ROOT))` が通る）、`test_characters_follow_default_style`（全 `CHARACTER_IDS` の `character.yaml` に `style` キーが無い）
- [ ] **Step 2:** `.venv/bin/pytest tests/test_repo_configs.py -q` で落ちることを確かめる
- [ ] **Step 3:** `defaults.yaml` を作り、2 人の `style:` を消す
- [ ] **Step 4:** README（ファイルの地図に `defaults.yaml`、画風はデフォルト → 個別の上書き → `style=` 引数の順で決まること）、CLAUDE.md（新しいキャラクターに `style:` を書かない、書き戻しで `style` を書くのはデフォルトと違う画風を採用したときだけ）、ノートブックの説明（`STYLE = None` ならデフォルト、キャラクターに個別の指定があればそれ）を直す
- [ ] **Step 5:** `.venv/bin/pytest -q` で全部通ることを確かめる
- [ ] **Step 6:** コミット `feat: デフォルトの画風を thick-line-flat にし、セラ・ボルグを従わせる`

### Task 3: 引き継ぎを直す

- [ ] **Step 1:** `HANDOVER.md` の `## 進捗` を `~/.claude/rules/progress.md` の書式で直し、状態の欄の「ally06 は soft-cel」に当たる記述とボルグの画風の記述を今の状態に合わせる
- [ ] **Step 2:** コミット `docs: 引き継ぎにデフォルトの画風を書く`
