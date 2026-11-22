# runs/ の階層分け 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新しく作る候補を `runs/<id>/<種類>/<日時>/` に保存し、候補を開き直すときにキャラクター・種類で絞れるようにする。古い `runs/<日時>/` も今までどおり開ける。

**Architecture:** 階層の決まり（種類の名前・保存先・探し方）は `selection.py` の 1 か所に置く。`forge.py` は保存先を `selection.py` の関数で決め、`picker.py` は絞り込みの引数を `run_paths` に渡すだけにする。

**Tech Stack:** Python 3.12、Pillow、pytest。

**Spec:** `docs/superpowers/specs/2026-11-22-runs-hierarchy-design.md`

## Global Constraints

- ファイル名（`<id>_<型>[_<表情 or 画風>]_<seed>.png`）と PNG の条件のキーは変えない。`run_id` は `%Y%m%d_%H%M%S` の日時のまま
- 保存先は `runs/<キャラクターID>/<種類>/<日時>/`。種類は `face`・`face-<表情>`・`fullbody`・`styles`（顔以外の型の見比べは `styles-<型>`）
- 複数キャラクターを一度に作ったときは、各キャラクターのフォルダに同じ日時で分けて入れる
- 古い階層 `runs/<日時>/`（名前が `\d{8}_\d{6}`）は読めるまま。`char`・`kind` で絞ったときは新しい階層だけが対象
- `selected/`・`packs/`・`intake`・`export` は変えない。古いフォルダの移行・削除はしない
- 入力の誤りは `ConfigError`（何をすればよいかを書く）。フォルダ名に使う値（キャラクター ID・表情名・型名）は、`selection.py` の既存のパス検査（`_check_name_part`）を通す
- リポジトリにゲーム側のリポジトリの名前・ファイル名・定数を書かない
- コミットは Conventional Commits + 日本語の要約。author は akabee0161。push しない。ブランチは今の `feat/style-soft-cel` のまま
- テスト: `.venv/bin/pytest`（各タスクでは対象のファイルだけで足りる）

## Review Focus

1. `runs/` の下に、階層に当てはまらないもの（ファイル、`.ipynb_checkpoints`、日時でない名前のフォルダ）がある → 無視して、探す処理が落ちない（Task 1 のテスト）
2. `char`・`kind` の打ち間違い → 空の画面でなく、あるキャラクター・種類を挙げた `ConfigError`（Task 1 のテスト）
3. 複数キャラクターを ■ で止めた → できた分のパスが返り、保存先が複数のフォルダに分かれていても `P.pick(forge, paths)` に渡せる（Task 2 のテスト）
4. 表情名にパス区切りや `..` がある → GPU を回す前に止まる（Task 2 のテスト）
5. 新旧の階層で同じ日時がある → 開き直しは両方の候補を返す（Task 1 のテスト）

---

### Task 1: 探す側と階層の決まり（selection.py）

**Files:**
- Modify: `src/artgen/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Produces:
  - `run_kind(template: str, expression: str | None = None, *, compare: bool = False) -> str` — `compare` なら顔は `styles`・それ以外は `styles-<型>`、`expression` があれば `face-<表情>`、無ければ型の名前。`expression`・`template` は `_check_name_part` を通す
  - `run_dir(runs_dir: Path, char_id: str, kind: str, run_id: str) -> Path` — `runs_dir/char_id/kind/run_id`（`char_id` も検査する）
  - `run_ids(runs_dir: Path) -> list[str]` — 新旧の階層の日時（重複なし・古い順）
  - `run_paths(runs_dir: Path, run_id: str | None = None, *, char: str | None = None, kind: str | None = None) -> list[Path]` — 条件に合う日時のうち最新（`run_id` 指定ならその日時）の、全フォルダの PNG をパス順に返す
  - `find_candidate(runs_dir, key, run_id=None)` — 引数は変えない。新旧の階層をたどる（ファイル名は全部の日時を新しい順、シードは最新の日時）

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_selection.py`。既存の `_make`・`FACE` を使い、新しい階層の fixture `runs_new` を足す）

```python
def test_run_kind_names():
    assert run_kind("face") == "face"
    assert run_kind("fullbody") == "fullbody"
    assert run_kind("face", "hurt") == "face-hurt"
    assert run_kind("face", compare=True) == "styles"
    assert run_kind("fullbody", compare=True) == "styles-fullbody"

@pytest.mark.parametrize("bad", ["../x", "a/b", "a\\b", ".."])
def test_run_kind_rejects_unsafe_names(bad):
    with pytest.raises(ConfigError, match="ファイル名に使えない"):
        run_kind("face", bad)

def test_run_dir_layout(tmp_path):
    assert run_dir(tmp_path, "c1", "face-hurt", "20261122_010203") == tmp_path / "c1" / "face-hurt" / "20261122_010203"

def test_run_paths_filters_by_char_and_kind_on_new_layout(runs_new):
    # runs_new: c1/face/20261120_090000/c1_face_1.png, c1/face/20261121_090000/c1_face_7.png,
    #           c2/face/20261121_090000/c2_face_8.png, c1/fullbody/20261121_100000/c1_fullbody_5.png
    assert [p.name for p in run_paths(runs_new, char="c1", kind="face")] == ["c1_face_7.png"]
    assert [p.name for p in run_paths(runs_new, run_id="20261121_090000")] == ["c1_face_7.png", "c2_face_8.png"]  # 複数キャラクターの同じ日時は全部
    assert [p.name for p in run_paths(runs_new)] == ["c1_fullbody_5.png"]  # 絞らなければ最新の日時
    assert [p.name for p in run_paths(runs_new, char="c1", kind="face", run_id="20261120_090000")] == ["c1_face_1.png"]

def test_run_paths_error_lists_what_exists(runs_new):  # Review Focus 2
    with pytest.raises(ConfigError, match=r"候補が無い.*c1.*c2"):
        run_paths(runs_new, char="c9")
    with pytest.raises(ConfigError, match=r"候補が無い.*face.*fullbody"):
        run_paths(runs_new, char="c1", kind="nope")

def test_legacy_layout_still_readable(runs):  # 既存の fixture（runs/<日時>/ のフラットな階層）
    assert [p.name for p in run_paths(runs, run_id="20261121_090000")] == ["c1_face_7.png", "c2_face_8.png"]
    assert find_candidate(runs, "c1_face_1.png").name == "c1_face_1.png"
    with pytest.raises(ConfigError, match="候補が無い"):
        run_paths(runs, char="c1")  # 古い階層は絞り込みの対象外

def test_mixed_layouts_same_run_id_returns_both(runs_new):  # Review Focus 5
    _run(runs_new, "20261121_090000", [("c3_face_9.png", 9)])  # 古い階層に同じ日時
    assert {p.name for p in run_paths(runs_new, run_id="20261121_090000")} == {"c1_face_7.png", "c2_face_8.png", "c3_face_9.png"}

def test_stray_entries_under_runs_are_ignored(runs_new):  # Review Focus 1
    (runs_new / ".ipynb_checkpoints").mkdir()
    (runs_new / "note.txt").write_text("x")
    (runs_new / "c1" / "face" / "not-a-run").mkdir()
    assert run_ids(runs_new) == sorted(set(run_ids(runs_new)))
    assert "20261121_100000" in run_ids(runs_new)
    assert [p.name for p in run_paths(runs_new)] == ["c1_fullbody_5.png"]

def test_find_candidate_walks_new_layout(runs_new):
    assert find_candidate(runs_new, "c2_face_8.png").parent.parent.parent.name == "c2"
    assert find_candidate(runs_new, 5).name == "c1_fullbody_5.png"  # シードは最新の日時
    with pytest.raises(ConfigError, match="見つからない"):
        find_candidate(runs_new, "nope.png")
```

既存の `test_run_paths_latest_and_named`（古い階層）は、そのまま通ること。

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_selection.py -v`
Expected: 新しいテストが FAIL（`run_kind` などが無い）

- [ ] **Step 3: `selection.py` に実装する**
  - 日時の判定は `re.fullmatch(r"\d{8}_\d{6}", name)`。新しい階層は `runs/<char>/<kind>/<日時>/`（`char`・`kind` のフォルダの中の、日時の名前のフォルダだけを拾う。ファイルや他の名前は無視）、古い階層は `runs/<日時>/`
  - 内部の共通部品は「(日時, char | None, kind | None, フォルダ) の一覧を返す関数」1 つにし、`run_ids`・`run_paths`・`find_candidate` はそれを使う。古い階層は `char`・`kind` が None で、絞り込みのときは除く
  - 絞り込みで 1 件も無いときの `ConfigError`: `<runs_dir>: 候補が無い: char=…, kind=…（あるキャラクター: …、そのキャラクターの種類: …）`。日時が見つからないときと、run が 1 つも無いときの文言は既存のまま（`run が無い`）

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: コミット**

```bash
git add src/artgen/selection.py tests/test_selection.py
git commit -m "feat: runs/ を新しい階層でも古い階層でも探せるようにし、種類の決まりを足す"
```

---

### Task 2: 保存先を新しい階層にする（forge.py）

**Files:**
- Modify: `src/artgen/forge.py`
- Test: `tests/test_forge.py`

**Interfaces:**
- Consumes: `run_kind`・`run_dir`（Task 1）
- Produces: `generate`・`inpaint`・`compare_styles` が `run_dir(self.out_dir, char_id, run_kind(...), run_id) / <ファイル名>` に保存する。戻り値・ファイル名・PNG の条件は変えない

- [ ] **Step 1: テストを直し、失敗するテストを書く**（`tests/test_forge.py`。既存の `paths[0].parent.parent == tmp_path / "runs"` は `paths[0].parent.parent.parent.parent == tmp_path / "runs"` に直す）

```python
def test_generate_saves_under_char_kind_run(forge, tmp_path):
    p = forge.generate("c1", "face", seeds=[7])[0]
    assert p.parent.parent.parent.parent == tmp_path / "runs"
    assert (p.parent.parent.parent.name, p.parent.parent.name) == ("c1", "face")
    assert re.fullmatch(r"\d{8}_\d{6}", p.parent.name) and read_params(p)["run_id"] == p.parent.name

def test_multi_character_run_splits_dirs_with_same_run_id(forge):
    paths = forge.generate(["c1", "c2"], "face", seeds=[7])
    assert paths[0].parent.parent.parent.name == "c1" and paths[1].parent.parent.parent.name == "c2"
    assert paths[0].parent.name == paths[1].parent.name  # 同じ日時

def test_kinds_for_fullbody_expression_and_styles(forge):
    assert forge.generate("c1", "fullbody", seeds=[1])[0].parent.parent.name == "fullbody"
    forge.adopt(forge.generate("c1", "face", seeds=[2])[0])
    assert forge.inpaint("c1", "hurt", seeds=[3])[0].parent.parent.name == "face-hurt"
    assert forge.compare_styles("c1", styles=["s"], seeds=[4])[0].parent.parent.name == "styles"

def test_unsafe_expression_name_stops_before_gpu(forge, pipe):  # Review Focus 4
    # character.yaml の表情名が "a/b" のキャラクター fixture（repo に足す）で inpaint する
    forge.adopt(forge.generate("badexpr", "face", seeds=[1])[0])
    pipe.calls.clear()
    with pytest.raises(ConfigError, match="ファイル名に使えない"):
        forge.inpaint("badexpr", "a/b", seeds=[2])
    assert pipe.calls == []

def test_interrupted_multi_character_run_returns_paths_usable_by_pick(forge, pipe):  # Review Focus 3
    def stop_on_third(path):
        if len(pipe.calls) == 3:
            raise KeyboardInterrupt
    paths = forge.generate(["c1", "c2"], "face", seeds=[7, 8], on_image=stop_on_third)
    assert [p.name for p in paths] == ["c1_face_7.png", "c1_face_8.png", "c2_face_7.png"]
    assert all(p.is_file() for p in paths)
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_forge.py -v`
Expected: 新しいテストと、直した既存テストが FAIL

- [ ] **Step 3: 実装する**
  - `_text2img_batch` と `inpaint` の保存先を `run_dir(self.out_dir, char_id, run_kind(...), run_id) / <ファイル名>` にする。`run_kind` の引数: `generate` は `(template)`、`compare_styles` は `(template, compare=True)`、`inpaint` は `("face", expression)`。`_text2img_batch` に `compare: bool` を足して渡す
  - `_run` の `run_dir` 引数は、中断のメッセージに使うだけなので `self.out_dir` を渡し、メッセージは「N 枚が `<out_dir>` の下に保存済み」にする
  - 検査（`run_kind`・`run_dir` が出す `ConfigError`）は、GPU を回す前に全部の entries で済ませる（`_text2img_batch` は plans を作る段階、`inpaint` は prompt を作る段階）

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: コミット**

```bash
git add src/artgen/forge.py tests/test_forge.py
git commit -m "feat: 候補の保存先を runs/<id>/<種類>/<日時>/ にする"
```

---

### Task 3: 開き直しの絞り込みと、ノートブック・ドキュメント

**Files:**
- Modify: `src/artgen/picker.py`、`notebooks/run.ipynb`、`README.md`、`CLAUDE.md`
- Test: `tests/test_picker.py`、`tests/test_notebook.py`

**Interfaces:**
- Consumes: `run_paths(runs_dir, run_id, *, char, kind)`（Task 1）
- Produces: `pick(forge, paths=None, run_id=None, *, char=None, kind=None, widgets=None, display=None, thumb=320)`。`paths` を渡したときに `char`・`kind` も渡されたら `ConfigError`（どちらかだけにする）

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_picker.py`。`forge.generate` で作った新しい階層の候補を使う）

```python
def test_pick_with_char_and_kind_opens_latest_matching_run(forge, fake_widgets):
    forge.generate(["c1", "c2"], "face", seeds=[1])
    forge.generate("c1", "fullbody", seeds=[2])
    pick(forge, char="c1", kind="face", widgets=fake_widgets, display=lambda *a: None)
    assert [b.description for b in fake_widgets.created("Button")] == ["採用", "左右反転して採用"]  # c1 の顔 1 枚

def test_pick_char_and_kind_with_paths_is_an_error(forge, fake_widgets):
    p = forge.generate("c1", "face", seeds=[1])
    with pytest.raises(ConfigError, match="paths"):
        pick(forge, p, char="c1", widgets=fake_widgets, display=lambda *a: None)

def test_pick_without_filters_opens_latest_run_across_characters(forge, fake_widgets):
    forge.generate(["c1", "c2"], "face", seeds=[1])
    pick(forge, widgets=fake_widgets, display=lambda *a: None)
    assert len(fake_widgets.created("Button")) == 4  # 2 人分（各 [採用]・[左右反転して採用]）
```

`tests/test_notebook.py` に追加: 「候補を開き直す」のセルのコードに `P.pick(forge, char=` が入っていること。

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_picker.py tests/test_notebook.py -v`
Expected: 新しいテストが FAIL

- [ ] **Step 3: 実装する**
  - `pick` に `char`・`kind`（キーワード専用）を足し、`paths is None` のとき `run_paths(forge.out_dir, run_id, char=char, kind=kind)` を呼ぶ。`paths` と `char`/`kind` が一緒に渡されたら `ConfigError("paths を渡したときは char・kind は使えない")`（文言に `paths` を含める）
  - `notebooks/run.ipynb` の「5. 候補を開き直す」: 説明に、階層（`runs/<キャラクター>/<種類>/<日時>/`）と、絞り込みの書き方を足す。コードのセルは `P.pick(forge, char="ally05", kind="face")` にし、その下にコメント行で `run_id=` を足す例と `P.pick(forge, run_id=None)`（最新の日時の全部）を示す。出力は空のまま、JSON の整形は今のファイルに合わせる（スクリプトは scratchpad に置いて実行）
  - `README.md`: 置き場の表の `runs/<run_id>/` の行を新しい階層に直し、候補を開き直す説明を `char`・`kind` に合わせる。古い階層も開けることを 1 行書く
  - `CLAUDE.md`: 変更が要る記述があれば直す（無ければ触らない）

- [ ] **Step 4: 全部を確かめる**

Run: `.venv/bin/pytest -q`
Expected: 全部 PASS（`test_repo_does_not_mention_the_game` も）

- [ ] **Step 5: コミット**

```bash
git add src/artgen/picker.py notebooks/run.ipynb README.md CLAUDE.md tests/test_picker.py tests/test_notebook.py
git commit -m "feat: 候補を開き直すときキャラクター・種類で絞れるようにし、ノートブックと README を直す"
```
