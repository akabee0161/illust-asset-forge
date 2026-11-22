# ノートブックで完結する作業フロー 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 顔の生成・採用・表情の範囲・表情・全身を Colab のノートブックの中で済ませ、採用した絵を Drive の `selected/` にまとめ、最後に zip で Claude に渡せるようにする。あわせて顔の構図（左向き・頭から胸まで）、画風の見比べ、複数キャラクターの一括生成を入れる。

**Architecture:** GPU も widgets も使わない「採用した絵の置き場」（`selection.py`）を土台にし、`Forge` はそれを使って生成・採用・範囲の解決を行う。ボタンの画面（`picker.py`）は `Forge` のメソッドを呼ぶだけの薄い層。取り込み（`intake.py`）は「フォルダを `source/` に入れる関数」と「zip を展開してそれを呼ぶコマンド」に分け、将来 Colab から直接コミットするときに関数を使い回せるようにする。

**Tech Stack:** Python 3.12、Pillow、PyYAML、pytest。Colab 側で diffusers・torch・ipywidgets（import は関数の中だけ）。

**Spec:** `docs/superpowers/specs/2026-11-21-notebook-workflow-design.md`

## Global Constraints

- `diffusers`・`torch`・`transformers` は `forge.load_pipeline()`・`forge.attach()` の中でだけ import する。`ipywidgets`・`IPython` は `picker.py`・`forge.show*` の関数の中でだけ import する（テストは偽物を渡す）
- 設定の誤りは `ConfigError`、メッセージは `<相対パス>: <キーの位置>: <何がおかしいか>`。それ以外の入力の誤り（候補が見つからない等）も `ConfigError` で、何をすればよいかを書く
- 顔は正方形（1024×1024）で生成し `export_size: 256` に縮める決まりは変えない。`export.py` は変えない
- Drive の置き場: `<base>/runs/`・`<base>/selected/<id>/`・`<base>/packs/`。`<base>` は既定で `/content/drive/MyDrive/illust-asset-forge`
- `selected/<id>/` のファイル名は `face.png`・`face_<表情>.png`・`fullbody.png`・`selection.json`（中身は `{"inpaint_box": [x0, y0, x1, y1]}`）、退避先は `selected/<id>/replaced/<YYYYmmdd_HHMMSS>/`
- 採用した PNG の条件には `flipped`（bool）を足す
- 画風の差し替えは条件の `overrides` に `"style": "<名前>"` として記録する（キャラクターの設定と同じ画風なら記録しない）
- リポジトリにゲーム側のリポジトリの名前・ファイル名・定数を書かない
- コミットは Conventional Commits + 日本語の要約。author は akabee0161。push しない
- テスト: `uv venv .venv && uv pip install -e '.[dev]' && .venv/bin/pytest`（各タスクでは `.venv/bin/pytest <ファイル>` で足りる）

## Review Focus

1. 条件の無い PNG（スマホで保存し直した等）を採用しようとした → `selected/` に何も書かず、条件が無いことを言って止まる（Task 2 のテスト）
2. 複数キャラクターの生成を ■ で止めた → できた分のパスが返り、そのまま選ぶ画面に渡せる（Task 3 のテスト）
3. 複数キャラクターの run で、シードだけで採用した（別のキャラクターに同じシードがある）→ 推測で選ばず、候補のファイル名を並べて止まる（Task 2 のテスト）
4. zip に `../` や絶対パス、知らないキャラクター、受け付けない名前が入っている → 何も展開せずに止まる（Task 6 のテスト）
5. 範囲を決める画面で、画像からはみ出す枠や feather より小さい枠で [決める] を押した → 保存せず、理由を画面に出す（Task 5 のテスト）

---

### Task 1: 型 face の新しい構図・既定の枠・画風の候補

**Files:**
- Modify: `src/artgen/config.py`（`Template` に `inpaint_box`、`load_template` で `inpaint.box` を読む）
- Modify: `templates/face.yaml`
- Create: `styles/thick-line-flat.yaml`・`styles/soft-painterly.yaml`・`styles/retro-anime.yaml`・`styles/watercolor.yaml`
- Test: `tests/test_config.py`・`tests/test_repo_configs.py`

**Interfaces:**
- Produces: `Template.inpaint_box: tuple[int, int, int, int] | None`（`inpaint.box` が無ければ None）。`styles/` の 5 つの画風

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_config.py
def test_template_reads_inpaint_box(tmp_path):
    write(tmp_path, "templates/face.yaml",
          "width: 1024\nheight: 1024\nprompt: [p]\nnegative: []\n"
          "inpaint: {strength: 0.6, feather: 24, box: [300, 200, 700, 600]}\n")
    assert load_template(tmp_path, "face").inpaint_box == (300, 200, 700, 600)

@pytest.mark.parametrize("box", ["[1, 2, 3]", "[5, 5, 5, 9]", "[-1, 0, 10, 10]", "[0, 0, 1.5, 9]"])
def test_template_rejects_bad_inpaint_box(tmp_path, box):
    write(tmp_path, "templates/face.yaml",
          f"width: 1024\nheight: 1024\nprompt: [p]\nnegative: []\n"
          f"inpaint: {{strength: 0.6, feather: 24, box: {box}}}\n")
    with pytest.raises(ConfigError, match=r"templates/face.yaml: inpaint.box: "):
        load_template(tmp_path, "face")

# tests/test_repo_configs.py
def test_face_template_is_left_facing_bust_with_default_box():
    face = load_template(ROOT, "face")
    assert "upper body" not in face.prompt
    assert "upper body" in face.negative
    assert "looking to the side" in face.prompt
    assert face.inpaint_box is not None
    check_box(face.inpaint_box, (face.width, face.height))
    check_feather(face.inpaint_box, face.inpaint_feather)

def test_style_candidates_exist():
    names = sorted(p.stem for p in (ROOT / "styles").glob("*.yaml"))
    assert names == ["color-anime", "retro-anime", "soft-painterly", "thick-line-flat", "watercolor"]
    for name in names:
        load_style(ROOT, name)
```

（`write` は `tests/test_config.py` に既にある補助関数を使う。無ければ `test_forge.py` と同じ形で足す）

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_config.py tests/test_repo_configs.py -v`
Expected: FAIL（`inpaint_box` 属性が無い、画風が足りない）

- [ ] **Step 3: 実装する**

- `Template` に `inpaint_box: tuple[int, int, int, int] | None` を足す。`inpaint.box` は `load_character` の `inpaint_box` と同じ検査（4 つの 0 以上の整数、x0<x1・y0<y1）で、エラーの位置は `inpaint.box`。検査は `_Reader` に `box(where, value)` として切り出し、`load_character` もそれを使う
- `templates/face.yaml` を次の値にする（初期値。依頼者が Colab で調整する）:

```yaml
width: 1024
height: 1024
prompt: [portrait, looking to the side, simple background, white background]
negative: [full body, cowboy shot, upper body, from behind, profile, head out of frame, cropped head]
export_size: 256
use_items: false
inpaint: {strength: 0.6, feather: 24, box: [320, 260, 720, 660]}
```

- 画風の候補（`negative` はどれも `color-anime.yaml` と同じ一覧を写す）:
  - `thick-line-flat.yaml` prompt: `[thick outlines, flat color, limited palette, bold lineart, simple shading, masterpiece, high score, great score, absurdres]`
  - `soft-painterly.yaml` prompt: `[painterly, soft shading, soft lighting, detailed face, masterpiece, high score, great score, absurdres]`
  - `retro-anime.yaml` prompt: `[retro artstyle, 1990s (style), anime screencap, cel shading, masterpiece, high score, great score, absurdres]`
  - `watercolor.yaml` prompt: `[watercolor (medium), traditional media, soft colors, lineart, masterpiece, high score, great score, absurdres]`

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: コミット**

```bash
git add src/artgen/config.py templates/face.yaml styles tests/test_config.py tests/test_repo_configs.py
git commit -m "feat: 顔の型を左向き・頭から胸までの構図にし、既定の範囲と画風の候補を足す"
```

---

### Task 2: 採用した絵の置き場（selection.py）

**Files:**
- Create: `src/artgen/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Produces:
  - `class SelectionStore(selected_dir: Path, packs_dir: Path)`
  - `SelectionStore.adopt(candidate: Path, flip: bool = False, *, now: Callable[[], datetime] = datetime.now) -> Path` — 採用して `selected/<id>/<名前>` に保存し、そのパスを返す
  - `SelectionStore.face_path(char_id: str) -> Path`（存在は問わない）
  - `SelectionStore.face_style(char_id: str) -> str | None` — 採用した顔の条件の `style`。顔が無ければ None
  - `SelectionStore.read_box(char_id: str) -> tuple[int, int, int, int] | None`
  - `SelectionStore.save_box(char_id: str, box: tuple[int, int, int, int]) -> Path`
  - `SelectionStore.characters() -> list[str]` — PNG のある `selected/<id>/` の ID（名前順）
  - `SelectionStore.pack(char_ids: list[str] | None = None, *, now=datetime.now) -> Path` — `packs/selected_<日時>.zip`
  - `selected_name(params: dict) -> str` — 条件から `face.png`・`face_<表情>.png`・`fullbody.png` を決める
  - `run_ids(runs_dir: Path) -> list[str]`（古い順）、`run_paths(runs_dir: Path, run_id: str | None = None) -> list[Path]`（None は最新の run、名前順）
  - `find_candidate(runs_dir: Path, key: str | int, run_id: str | None = None) -> Path` — ファイル名かシードで run の中の候補を 1 つ探す

- [ ] **Step 1: 失敗するテストを書く**

テストの PNG は `save_png(Image.new("RGB", (1024, 1024)), path, params)` で作る。左半分を赤・右半分を青にした絵で反転を確かめる。

```python
def test_selected_name_from_params():
    assert selected_name({"template": "face", "expression": "normal"}) == "face.png"
    assert selected_name({"template": "face", "expression": "hurt", "source": "face.png"}) == "face_hurt.png"
    assert selected_name({"template": "fullbody"}) == "fullbody.png"

def test_adopt_copies_with_params_and_flipped_false(store, candidate):  # candidate: c1_face_7.png
    dest = store.adopt(candidate)
    assert dest == store.selected_dir / "c1" / "face.png"
    assert read_params(dest)["seed"] == 7
    assert read_params(dest)["flipped"] is False

def test_adopt_flip_mirrors_and_records(store, half_red_candidate):
    dest = store.adopt(half_red_candidate, flip=True)
    with Image.open(dest) as im:
        assert im.getpixel((0, 0)) == (0, 0, 255)  # 元は左が赤
    assert read_params(dest)["flipped"] is True
    assert read_params(half_red_candidate).get("flipped") is None  # 候補は変えない

def test_adopt_flip_only_for_face(store, expression_candidate, fullbody_candidate):
    for c in (expression_candidate, fullbody_candidate):
        with pytest.raises(ConfigError, match="左右反転は顔"):
            store.adopt(c, flip=True)

def test_readopting_face_moves_expressions_and_box(store, candidate, expression_candidate):
    store.adopt(candidate); store.adopt(expression_candidate); store.save_box("c1", (1, 2, 300, 400))
    store.adopt(candidate, now=lambda: datetime(2026, 11, 22, 10, 0, 0))
    moved = store.selected_dir / "c1" / "replaced" / "20261122_100000"
    assert (moved / "face_hurt.png").is_file() and (moved / "selection.json").is_file()
    assert not (store.selected_dir / "c1" / "face_hurt.png").exists()
    assert store.read_box("c1") is None

def test_adopt_without_params_writes_nothing(store, tmp_path):  # Review Focus 1
    plain = tmp_path / "plain.png"; Image.new("RGB", (8, 8)).save(plain)
    with pytest.raises(MissingParamsError):
        store.adopt(plain)
    assert not store.selected_dir.exists() or not any(store.selected_dir.rglob("*.png"))

def test_box_roundtrip_and_face_style(store, candidate):  # candidate の条件の style は "s"
    assert store.face_style("c1") is None
    store.adopt(candidate)
    assert store.face_style("c1") == "s"
    store.save_box("c1", (10, 20, 300, 400))
    assert store.read_box("c1") == (10, 20, 300, 400)

def test_pack_zips_without_replaced(store, candidate, expression_candidate):
    store.adopt(candidate); store.adopt(expression_candidate); store.adopt(candidate)
    z = store.pack(now=lambda: datetime(2026, 11, 22, 10, 0, 0))
    assert z == store.packs_dir / "selected_20261122_100000.zip"
    assert sorted(zipfile.ZipFile(z).namelist()) == ["c1/face.png"]

def test_pack_with_nothing_selected_says_so(store):
    with pytest.raises(ConfigError, match="採用した絵が無い"):
        store.pack()

def test_find_candidate_by_name_and_seed(runs):  # runs に 2 つの run、最新に c1_face_7.png・c2_face_8.png
    assert find_candidate(runs, "c1_face_7.png").name == "c1_face_7.png"
    assert find_candidate(runs, 8).name == "c2_face_8.png"
    with pytest.raises(ConfigError, match="見つからない"):
        find_candidate(runs, 999)

def test_find_candidate_ambiguous_seed_lists_names(runs_same_seed):  # Review Focus 3: c1_face_7・c2_face_7
    with pytest.raises(ConfigError, match=r"c1_face_7\.png.*c2_face_7\.png"):
        find_candidate(runs_same_seed, 7)

def test_run_paths_latest_and_named(runs):
    assert run_ids(runs) == sorted(run_ids(runs))
    assert run_paths(runs) == run_paths(runs, run_ids(runs)[-1])
    with pytest.raises(ConfigError, match="run が無い"):
        run_paths(runs, "19990101_000000")
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_selection.py -v`
Expected: FAIL（`artgen.selection` が無い）

- [ ] **Step 3: 実装する**

- `selected_name`: `template == "fullbody"` → `fullbody.png`。`template == "face"` で条件に `source` があれば `face_<expression>.png`、無ければ `face.png`。それ以外は `ConfigError`
- `adopt`: 先に `read_params` と `selected_name` と flip の可否を検査してから書く。顔を採用するとき、`selected/<id>/` に `face_*.png` か `selection.json` があれば `replaced/<日時>/` へ移し、移したファイル名を `print` する。保存は `save_png`
- `find_candidate`: `key` が文字列ならファイル名の一致、整数なら条件の `seed` の一致。1 つでなければ `ConfigError`（0 件: 「見つからない」、2 件以上: 候補のファイル名を並べ「ファイル名で指定する」）
- `pack`: `char_ids` が None なら `characters()`。各 `selected/<id>/` の PNG と `selection.json` を `<id>/<名前>` で入れる（`replaced/` は入れない）

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest tests/test_selection.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/artgen/selection.py tests/test_selection.py
git commit -m "feat: 採用した絵を Drive の決まった場所にまとめる置き場を足す"
```

---

### Task 3: Forge の生成を複数キャラクター・画風の差し替えに対応させる

**Files:**
- Modify: `src/artgen/forge.py`
- Test: `tests/test_forge.py`

**Interfaces:**
- Consumes: `SelectionStore`（Task 2）、`Template.inpaint_box`（Task 1）
- Produces:
  - `Forge.__init__(..., selected_dir: Path | None = None, packs_dir: Path | None = None)` — 既定は `out_dir.parent / "selected"`・`out_dir.parent / "packs"`。`self.selected: SelectionStore`
  - `Forge.generate(char_ids: str | list[str], template: str, n=4, seeds=None, extra="", remove="", extra_negative="", remove_negative="", guidance=None, steps=None, style: str | None = None, on_image=None) -> list[Path]`
  - `Forge.compare_styles(char_id: str, template: str = "face", styles: list[str] | None = None, seeds: list[int] | None = None, n: int = 2, on_image=None) -> list[Path]` — ファイル名 `<id>_<template>_<style>_<seed>.png`
  - `Forge.resolve_style(char_id: str, style: str | None, inherit: bool) -> tuple[str, str]` — (使う画風, 出どころ `"引数"|"採用した顔"|"設定"`)。`generate` は `inherit = template != "face"`、`inpaint`（Task 4）は `inherit=True` で呼ぶ

- [ ] **Step 1: 失敗するテストを書く**

`repo` の fixture に `characters/c2/character.yaml`（c1 と同じ形）と `styles/t.yaml`（prompt: [watercolor]）を足す。

```python
def test_generate_multiple_characters_in_one_run(forge):
    paths = forge.generate(["c1", "c2"], "face", seeds=[7, 8])
    assert [p.name for p in paths] == ["c1_face_7.png", "c1_face_8.png", "c2_face_7.png", "c2_face_8.png"]
    assert len({p.parent for p in paths}) == 1

def test_generate_checks_all_characters_before_gpu(forge, pipe):
    with pytest.raises(ConfigError, match="characters/nope/character.yaml"):
        forge.generate(["c1", "nope"], "face", seeds=[1])
    assert pipe.calls == []

def test_generate_interrupted_returns_saved_paths(forge, pipe):  # Review Focus 2
    def stop_on_third(path):
        if len(pipe.calls) == 3:
            raise KeyboardInterrupt
    paths = forge.generate(["c1", "c2"], "face", seeds=[7, 8], on_image=stop_on_third)
    assert [p.name for p in paths] == ["c1_face_7.png", "c1_face_8.png", "c2_face_7.png"]

def test_style_argument_overrides_and_is_recorded(forge, pipe):
    p = forge.generate("c1", "face", seeds=[1], style="t")[0]
    assert "watercolor" in pipe.calls[0][1]["prompt"]
    assert read_params(p)["style"] == "t"
    assert read_params(p)["overrides"]["style"] == "t"

def test_same_style_as_config_is_not_recorded(forge):
    p = forge.generate("c1", "face", seeds=[1], style="s")[0]
    assert "style" not in read_params(p)["overrides"]

def test_fullbody_inherits_adopted_face_style(forge, pipe, capsys):
    face = forge.generate("c1", "face", seeds=[1], style="t")[0]
    forge.selected.adopt(face)
    p = forge.generate("c1", "fullbody", seeds=[2])[0]
    assert read_params(p)["style"] == "t"
    assert "採用した顔" in capsys.readouterr().out

def test_face_generation_does_not_inherit(forge):
    forge.selected.adopt(forge.generate("c1", "face", seeds=[1], style="t")[0])
    assert read_params(forge.generate("c1", "face", seeds=[2])[0])["style"] == "s"

def test_compare_styles_same_seeds_per_style(forge):
    paths = forge.compare_styles("c1", styles=["s", "t"], seeds=[5, 6])
    assert [p.name for p in paths] == ["c1_face_s_5.png", "c1_face_s_6.png", "c1_face_t_5.png", "c1_face_t_6.png"]
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_forge.py -v`
Expected: 新しいテストが FAIL

- [ ] **Step 3: 実装する**

- `generate`: `char_ids` が文字列なら 1 件の一覧にする。全キャラクターの設定・画風・プロンプトを先に組み立ててから、キャラクターの順にシードを回す（シードの一覧は全員で共通。`seeds=None` なら n 個を 1 回だけ作る）。1 つの `_run` に流し、■ で止めたら保存済みを返す
- `resolve_style`: 引数があればそれ。無く、`inherit` が真で `self.selected.face_style(char_id)` があればそれ。どちらも無ければ `character.style`。出どころが「採用した顔」のときだけ `print(f"{char_id}: 画風 {name}（採用した顔に合わせた）")`
- 使った画風がキャラクターの設定と違えば `overrides` に `"style"` を入れる（`_recorded_overrides` に `style=` を渡す）
- `compare_styles`: `styles` が None なら `styles/*.yaml` の全部（名前順）。シードは `_seeds(n, seeds)` で 1 回だけ作り、画風の順に回す

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest tests/test_forge.py -v`
Expected: PASS（既存のテストも）

- [ ] **Step 5: コミット**

```bash
git add src/artgen/forge.py tests/test_forge.py
git commit -m "feat: 複数キャラクターの一括生成と、画風の差し替え・見比べを足す"
```

---

### Task 4: inpaint の既定の顔・範囲の解決・採用と、パイプラインの持ち方

**Files:**
- Modify: `src/artgen/forge.py`
- Test: `tests/test_forge.py`

**Interfaces:**
- Consumes: Task 2・3
- Produces:
  - `Forge.inpaint(char_id, expression, source: str | Path | None = None, ..., style: str | None = None, on_image=None)` — `source` 省略時は `self.selected.face_path(char_id)`
  - `Forge.resolve_box(char_id: str) -> tuple[tuple[int, int, int, int], str]` — (枠, 出どころ `"selection.json"|"character.yaml"|"型の既定"`)。どれも無ければ `ConfigError`
  - `Forge.adopt(candidate: str | int | Path, flip: bool = False, run_id: str | None = None) -> Path` — Path で存在すればそれ、そうでなければ `find_candidate(self.out_dir, ...)`
  - `Forge.pack(char_ids: list[str] | None = None) -> Path`
  - `load_pipeline(low_memory: bool = False) -> Any`（今の `load()` のパイプライン部分）
  - `attach(pipe, root: str | Path = ".", base_dir: str | Path = DEFAULT_BASE_DIR) -> Forge`（env を作り直して `Forge` を作る。`DEFAULT_BASE_DIR = "/content/drive/MyDrive/illust-asset-forge"`、`out_dir = base_dir / "runs"`）
  - `load()` は削除する

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_inpaint_uses_adopted_face_by_default(forge, pipe):
    forge.adopt(forge.generate("c1", "face", seeds=[1])[0])
    p = forge.inpaint("c1", "hurt", seeds=[3])[0]
    assert read_params(p)["source"] == "face.png"

def test_inpaint_without_adopted_face_says_what_to_do(forge, pipe):
    with pytest.raises(ConfigError, match="顔を採用"):
        forge.inpaint("c1", "hurt", seeds=[3])
    assert pipe.calls == []

def test_resolve_box_precedence(forge, repo):
    assert forge.resolve_box("c1") == ((100, 100, 400, 300), "character.yaml")
    assert forge.resolve_box("nobox") == ((320, 260, 720, 660), "型の既定")  # repo の face.yaml に box を足しておく
    forge.selected.save_box("c1", (10, 10, 500, 500))
    assert forge.resolve_box("c1") == ((10, 10, 500, 500), "selection.json")

def test_inpaint_prints_and_records_box_source(forge, capsys):
    forge.adopt(forge.generate("nobox", "face", seeds=[1])[0])
    p = forge.inpaint("nobox", "hurt", seeds=[3])[0]
    assert read_params(p)["inpaint_box"] == [320, 260, 720, 660]
    assert "型の既定" in capsys.readouterr().out

def test_adopt_by_seed_from_latest_run(forge):
    forge.generate("c1", "face", seeds=[11])
    assert forge.adopt(11, flip=True) == forge.selected.selected_dir / "c1" / "face.png"

def test_expression_inherits_face_style(forge):
    forge.adopt(forge.generate("c1", "face", seeds=[1], style="t")[0])
    assert read_params(forge.inpaint("c1", "hurt", seeds=[3])[0])["style"] == "t"
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_forge.py -v`
Expected: 新しいテストが FAIL

- [ ] **Step 3: 実装する**

- `inpaint`: 今の `character.inpaint_box is None` の検査を `resolve_box` に置き換え、出どころを `print`。`source` が None で採用した顔が無ければ `ConfigError(f"source: {char_id} の顔を採用していない。顔のセルで採用するか、source= にパスを渡す")`。画風は `resolve_style(char_id, style, inherit=True)` で、採用した顔を引き継ぐ
- `load_pipeline` / `attach`: 今の `load()` を 2 つに割る。`attach` は `diffusers`・`transformers`・`torch` の版と git の commit を毎回読み直す（pull 後の commit を記録するため）
- 既存テストの `load` への参照があれば直す

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: コミット**

```bash
git add src/artgen/forge.py tests/test_forge.py
git commit -m "feat: 表情を採用した顔と決めた範囲から作れるようにし、パイプラインを使い回す"
```

---

### Task 5: 選ぶ画面・範囲を決める画面・画風の一覧（picker.py）

**Files:**
- Create: `src/artgen/picker.py`
- Modify: `src/artgen/mask.py`（`overlay_box` を足す）
- Test: `tests/test_picker.py`・`tests/test_mask.py`

**Interfaces:**
- Consumes: `Forge.adopt`・`Forge.resolve_box`・`Forge.selected`（Task 2・4）、`check_box`・`check_feather`
- Produces:
  - `overlay_box(image: Image.Image, box, width: int = 6) -> Image.Image`（赤い枠を描いた写し。元は変えない）
  - `pick(forge, paths: list[Path] | None = None, run_id: str | None = None, *, widgets=None, display=None, thumb: int = 320) -> None` — `paths` が None なら `run_paths(forge.out_dir, run_id)`。キャラクターごとに見出しを付け、各候補に画像・ファイル名とシード・[採用]・（顔だけ）[左右反転して採用]・結果の文字を出す
  - `edit_box(forge, char_id: str, *, widgets=None, display=None) -> None` — 採用した顔に赤枠を重ねて表示し、x0・y0・x1・y1 のスライダー（0〜1024、step 8、初期値は `resolve_box`）と [この範囲で決める]
  - `show_styles(paths: list[Path], *, display=None, thumb: int = 256) -> None` — 行＝画風・列＝シードの一覧 1 枚と、行ごとの画風名

- [ ] **Step 1: 失敗するテストを書く**

偽の widgets（`Button`・`IntSlider`・`Image`・`Label`・`HTML`・`VBox`・`HBox` を、引数を属性に持つだけのクラスで作る。`Button.on_click(f)` は f を保存し、テストで `button.click()` で呼ぶ。`IntSlider.observe(f, names="value")` も同様）を `tests/test_picker.py` に置く。

```python
def test_overlay_box_draws_red_border_without_touching_original():
    im = Image.new("RGB", (100, 100), (255, 255, 255))
    out = overlay_box(im, (10, 10, 50, 50), width=2)
    assert out.getpixel((10, 30)) == (255, 0, 0) and out.getpixel((30, 30)) == (255, 255, 255)
    assert im.getpixel((10, 30)) == (255, 255, 255)

def test_pick_adopt_button_adopts(forge, fake_widgets):
    paths = forge.generate("c1", "face", seeds=[1, 2])
    pick(forge, paths, widgets=fake_widgets, display=lambda *a: None)
    buttons = fake_widgets.created("Button")
    assert [b.description for b in buttons] == ["採用", "左右反転して採用"] * 2
    buttons[1].click()
    assert read_params(forge.selected.face_path("c1"))["flipped"] is True

def test_pick_has_no_flip_for_expressions(forge, fake_widgets):
    forge.adopt(forge.generate("c1", "face", seeds=[1])[0])
    paths = forge.inpaint("c1", "hurt", seeds=[3])
    pick(forge, paths, widgets=fake_widgets, display=lambda *a: None)
    assert [b.description for b in fake_widgets.created("Button")] == ["採用"]

def test_pick_shows_error_text_instead_of_raising(forge, fake_widgets, tmp_path):
    p = forge.generate("c1", "face", seeds=[1])[0]
    pick(forge, [p], widgets=fake_widgets, display=lambda *a: None)
    p.unlink()
    fake_widgets.created("Button")[0].click()
    assert "採用できない" in fake_widgets.created("Label")[-1].value

def test_edit_box_saves_selection(forge, fake_widgets):
    forge.adopt(forge.generate("c1", "face", seeds=[1])[0])
    edit_box(forge, "c1", widgets=fake_widgets, display=lambda *a: None)
    x0, y0, x1, y1 = fake_widgets.created("IntSlider")
    assert (x0.value, y0.value, x1.value, y1.value) == (100, 100, 400, 300)
    x1.value = 600
    fake_widgets.created("Button")[0].click()
    assert forge.selected.read_box("c1") == (100, 100, 600, 300)

def test_edit_box_rejects_bad_box(forge, fake_widgets):  # Review Focus 5
    forge.adopt(forge.generate("c1", "face", seeds=[1])[0])
    edit_box(forge, "c1", widgets=fake_widgets, display=lambda *a: None)
    x0, y0, x1, y1 = fake_widgets.created("IntSlider")
    x1.value = x0.value + 16  # feather 24 の 2 倍より狭い
    fake_widgets.created("Button")[0].click()
    assert forge.selected.read_box("c1") is None
    assert "決められない" in fake_widgets.created("Label")[-1].value

def test_edit_box_without_face_says_what_to_do(forge, fake_widgets):
    with pytest.raises(ConfigError, match="顔を採用"):
        edit_box(forge, "c1", widgets=fake_widgets, display=lambda *a: None)
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_picker.py tests/test_mask.py -v`
Expected: FAIL

- [ ] **Step 3: 実装する**

- `widgets`・`display` が None なら関数の中で `import ipywidgets as widgets`・`from IPython.display import display`
- 画像は縮小した PNG のバイト列を `widgets.Image(value=..., format="png")` に渡す。スマホで崩れないよう、1 候補を `VBox`（画像・文字・ボタンの `HBox`）にして縦に並べる
- ボタンの処理は `forge.adopt(path, flip=...)` を呼び、成功なら結果の文字に `採用: <保存先の名前>`、`ConfigError`・`MissingParamsError`・`OSError` なら `採用できない: <理由>` を出す（例外を外に出さない。ボタンの中の例外は Colab で見えにくいため）
- `edit_box`: スライダーが変わるたびに `overlay_box` で描き直す。[この範囲で決める] で `check_box`・`check_feather` を通してから `save_box`。だめなら `決められない: <理由>`
- `show_styles`: 条件の `style`・`seed` で並べる

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: コミット**

```bash
git add src/artgen/picker.py src/artgen/mask.py tests/test_picker.py tests/test_mask.py
git commit -m "feat: ノートブックで採用・範囲決め・画風の見比べをするボタンの画面を足す"
```

---

### Task 6: 取り込み（intake.py）

**Files:**
- Create: `src/artgen/intake.py`
- Test: `tests/test_intake.py`

**Interfaces:**
- Consumes: `load_character`、`read_params`、`_accepted_names` 相当の名前の一覧（`export.py` の `_accepted_names` を import して使う）
- Produces:
  - `@dataclass CharacterReport: char_id: str; installed: list[str]; removed: list[str]; write_back: list[str]`
  - `install_selection(root: Path, selected_dir: Path) -> list[CharacterReport]` — `selected_dir/<id>/` ごとに `characters/<id>/source/` を丸ごと置き換える。将来 Colab から呼ぶ関数なので、zip・標準出力に依らない
  - `extract_pack(zip_path: Path, dest: Path) -> Path` — 安全に展開して、`selected/` と同じ形のフォルダを返す
  - `main(argv) -> int` — `python -m artgen.intake <zip> [--root .]`。一時ディレクトリに展開 → `install_selection` → 報告を出す

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_install_replaces_source_and_reports(repo, selected):  # selected/c1/ に face.png・face_hurt.png・selection.json
    write_png(repo / "characters/c1/source/fullbody.png", template="fullbody")  # 古いもの
    [r] = install_selection(repo, selected)
    assert r.char_id == "c1"
    assert sorted(r.installed) == ["face.png", "face_hurt.png"]
    assert r.removed == ["fullbody.png"]
    assert sorted(p.name for p in (repo / "characters/c1/source").iterdir()) == ["face.png", "face_hurt.png"]

def test_report_lists_write_backs(repo, selected):  # face.png の overrides: {extra: [scar], style: t}、selection.json の box が yaml と違う
    [r] = install_selection(repo, selected)
    text = "\n".join(r.write_back)
    assert "face.png" in text and "scar" in text
    assert "style" in text and "t" in text
    assert "inpaint_box" in text and "[10, 20, 300, 400]" in text

def test_install_checks_everything_before_writing(repo, selected):
    (selected / "c1" / "face_shout2.png").write_bytes((selected / "c1" / "face.png").read_bytes())
    before = sorted((repo / "characters/c1/source").iterdir())
    with pytest.raises(ConfigError, match="face_shout2.png"):
        install_selection(repo, selected)
    assert sorted((repo / "characters/c1/source").iterdir()) == before

def test_install_rejects_png_of_other_character(repo, selected_with_c2_png_in_c1):
    with pytest.raises(ConfigError, match="c2"):
        install_selection(repo, selected_with_c2_png_in_c1)

def test_install_rejects_png_without_params(repo, selected_plain_png):
    with pytest.raises(MissingParamsError):
        install_selection(repo, selected_plain_png)

@pytest.mark.parametrize("name", ["../evil.png", "/abs/face.png", "c1/sub/face.png", "nope/face.png"])
def test_extract_rejects_unsafe_or_unknown_entries(repo, tmp_path, name):  # Review Focus 4
    z = tmp_path / "p.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr(name, b"x")
    before = sorted((repo / "characters/c1/source").iterdir())
    assert main([str(z), "--root", str(repo)]) == 1
    assert sorted((repo / "characters/c1/source").iterdir()) == before
    assert not (tmp_path / "evil.png").exists()

def test_main_end_to_end(repo, pack_zip, capsys):  # Forge.pack と同じ形の zip
    assert main([str(pack_zip), "--root", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "c1" in out and "書き戻す" in out
```

`main` は `ConfigError`・`MissingParamsError` を受けて理由を出し、`1` を返す。

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_intake.py -v`
Expected: FAIL

- [ ] **Step 3: 実装する**

- `extract_pack`: 各エントリが `<id>/<名前>` の 2 段で、`..`・絶対パス・それ以上の深さが無いことを全部確かめてから展開する。`<id>` は `characters/<id>/character.yaml` がある ID だけ（無ければ `ConfigError`）
- `install_selection`: すべてのキャラクター・ファイルを検査（名前が受け付ける一覧にあるか、条件があるか、`character` が一致するか）してから書く。`selection.json` は `source/` に入れず、`inpaint_box` を書き戻しの一覧に使う
- 書き戻しの一覧: 各 PNG の `overrides`（`style` を含む）が空でなければ `"<名前>: overrides: <JSON>"`、`selection.json` の枠が `character.inpaint_box` と違えば `"inpaint_box: <枠> を character.yaml へ"`（`<枠>` は `[x0, y0, x1, y1]` の形の文字。タプルの `(…)` にしない。テストが見る）
- `main`: 報告の後に「次: 上の一覧を character.yaml に書き戻す。そのあと python -m artgen.export <id>」と出す（テストが「書き戻す」の文字列を見る）

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: コミット**

```bash
git add src/artgen/intake.py tests/test_intake.py
git commit -m "feat: 採用した絵の zip を source に取り込み、書き戻すものを一覧にするコマンドを足す"
```

---

### Task 7: ノートブックを新しい流れにする

**Files:**
- Modify: `notebooks/run.ipynb`
- Modify: `tests/test_notebook.py`

**Interfaces:**
- Consumes: `F.load_pipeline`・`F.attach`（Task 4）、`forge.generate`・`forge.compare_styles`・`forge.inpaint`・`forge.pack`（Task 3・4）、`P.pick`・`P.edit_box`・`P.show_styles`（Task 5、`from artgen import picker as P`）

- [ ] **Step 1: テストを新しい流れに直す（失敗する）**

```python
def test_notebook_uses_new_api():
    code = "\n".join(code_cells()[1])
    for needle in ("F.load_pipeline(", "F.attach(", "forge.generate(CHARS", "P.pick(forge", "P.edit_box(forge",
                   "forge.inpaint(CHAR, ", "forge.compare_styles(", "P.show_styles(", "forge.pack("):
        assert needle in code
    assert "/content/drive/MyDrive/illust-asset-forge/runs/<" not in code  # パスを書き換える場面が無い

def test_setup_cell_rebuilds_forge_from_kept_pipeline():
    _, cells = code_cells()
    setup = next(c for c in cells if "from artgen import forge" in c)
    assert "F.load_pipeline(" not in setup
    assert 'if "pipe" in globals()' in setup and "F.attach(pipe" in setup
    load = next(c for c in cells if "F.load_pipeline(" in c)
    assert 'if "pipe" not in globals()' in load and "F.attach(pipe" in load
```

既存の `test_notebook_uses_existing_api`・`test_notebook_load_cell_is_separate_and_skips_if_already_loaded` はこの 2 つに置き換える。`sys.path`・`sys.modules` の 2 つのテストは残す。

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_notebook.py -v`
Expected: FAIL

- [ ] **Step 3: ノートブックを書き直す**

セルの並び（設計書 5.9）。書き換えるのは各セル先頭の変数だけにする:

1. 説明（全体の流れ）
2. 準備: 今のセル + 末尾に `from artgen import picker as P` と `if "pipe" in globals(): forge = F.attach(pipe, ".")`
3. モデル: `if "pipe" not in globals(): pipe = F.load_pipeline()` と `forge = F.attach(pipe, ".")`（メモリ不足のときの `low_memory=True` を説明に書く）
4. 画風の見比べ（任意）: `CHAR = "ally05"` / `paths = forge.compare_styles(CHAR, seeds=None, n=2)` / `P.show_styles(paths)`
5. 顔: `CHARS = ["ally05"]` / `STYLE = None` / `paths = forge.generate(CHARS, "face", n=4, style=STYLE, on_image=F.show_one)` / `P.pick(forge, paths)`
6. 候補を開き直す: `P.pick(forge, run_id=None)`（説明: 中断・切断の後や、後から選ぶとき。`run_id` に Drive の runs のフォルダ名を書けばその run）
7. 表情の範囲: `CHAR = "ally05"` / `P.edit_box(forge, CHAR)`
8. 表情: `CHAR = "ally05"` / `EXPRESSION = "hurt"` / `paths = forge.inpaint(CHAR, EXPRESSION, n=4, on_image=F.show_one)` / `P.pick(forge, paths)`
9. 全身: `paths = forge.generate(CHARS, "fullbody", n=4, on_image=F.show_one)` / `P.pick(forge, paths)`
10. まとめ: `z = forge.pack()` / `print(z)` / `try: from google.colab import files; files.download(str(z))` `except Exception as e: print(f"ダウンロードできなかった（Drive の packs から取る）: {e}")`
11. 試すときの書き方（`extra=`・`remove=`・`seeds=`・`style=`・`forge.adopt("<ファイル名>", flip=True)` で採用する方法）

出力は空のまま保存する（`test_notebook_has_no_outputs`）。

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: コミット**

```bash
git add notebooks/run.ipynb tests/test_notebook.py
git commit -m "feat: ノートブックを、採用まで中で済ませる流れに書き直す"
```

---

### Task 8: 正典と引き継ぎを直す

**Files:**
- Modify: `README.md`・`CLAUDE.md`・`HANDOVER.md`

- [ ] **Step 1: README.md を新しい流れに直す**

- 2 章（Colab での使い方）を Task 7 のセルの並びに合わせる。`F.load(` の記述を `F.load_pipeline(low_memory=True)` に
- 3 章（採用した絵の渡し方）を「まとめのセルで zip を作り、Drive の `packs/` かダウンロードした zip を添付する」に
- 4 章（Claude 側の流れ）を `python -m artgen.intake <zip>` → 書き戻し → `python -m artgen.export <id>` に
- 1 章の地図に `selection`・`picker`・`intake`、Drive の置き場（`runs/`・`selected/`・`packs/`）を足す
- 型 face の構図（左向き・右向きは反転して採用）と、画風の候補の見比べを書く

- [ ] **Step 2: CLAUDE.md を直す**

- 「`inpaint_box` は、採用した顔の画像を見て決める」を「範囲は依頼者がノートブックで決め、`selection.json` で届く。Claude は `intake` の一覧を見て `character.yaml` に書き戻す」に
- 受け取りの手順を `intake` に。`overrides` の `style` を `character.yaml` の `style` に戻すことを足す
- 「取り込みの関数（`install_selection`）は、将来 Colab から直接コミットするときに使うので、zip・標準出力に依らないままにする」を足す
- `ipywidgets`・`IPython` の import を関数の中だけにする決まりを足す

- [ ] **Step 3: HANDOVER.md を直す**

- `## 進捗` を `~/.claude/rules/progress.md` の書式で、工程 2 を `（完了）`、工程 3 を ` ← 今ここ`、`段階: 実装（8 タスク中 8）`、次にやることを「依頼者が Colab で試す（push が要る）」などに
- 保留している課題（パスを手で書き換える手間）を、解決したので消す
- 状態の欄に、型 face の初期値と既定の枠は Colab で調整が要ることを書く

- [ ] **Step 4: 確かめる**

Run: `.venv/bin/pytest -q`
Expected: 全部 PASS（`test_repo_does_not_mention_the_game` も）

- [ ] **Step 5: コミット**

```bash
git add README.md CLAUDE.md HANDOVER.md
git commit -m "docs: README・CLAUDE.md・引き継ぎを、ノートブックで完結する流れに直す"
```
