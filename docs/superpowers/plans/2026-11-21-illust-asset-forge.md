# illust-asset-forge 初期実装 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** YAML の設定（画風・絵の種類・キャラクター）から Colab の GPU で顔・表情の差分・全身絵を生成し、採用した画像を受け渡しの形に書き出せるようにする。

**Architecture:** GPU を使わない部分（設定・プロンプト・PNG の条件・描き直す範囲の貼り戻し・書き出し）を純粋な Python モジュールにしてテストする。GPU を使う `forge.py` はパイプラインを外から受け取る作りにして、偽のパイプラインでテストする。Colab のノートブックは clone・導入・`forge.load()` と 1 行の呼び出しだけを持つ。

**Tech Stack:** Python 3.12、PyYAML、Pillow、pytest（サーバーでのテスト）。Colab では加えて diffusers（コミュニティパイプライン `lpw_stable_diffusion_xl`）・transformers・accelerate・safetensors・torch（Colab のもの）。

**Spec:** `docs/superpowers/specs/2026-11-21-illust-asset-forge-design.md`

## Global Constraints

- モデル `cagliostrolab/animagine-xl-4.0`、スケジューラ `EulerAncestralDiscreteScheduler`（記録名 `"Euler a"`）、guidance 5、28 steps が既定
- シードは CPU の `torch.Generator().manual_seed(seed)`。ランダムのときは 0〜2**32-1
- 顔の型は 1024×1024 で生成し、書き出しは 256×256（Lanczos で 1 回だけ縮める）。全身は 832×1216 で生成し、書き出しで縮めない
- PNG の条件は tEXt の `parameters` に JSON（`ensure_ascii=False`）。キー名は今のノートブックのもの（`prompt`・`negative_prompt`・`seed`・`guidance_scale`・`num_inference_steps`・`sampler`・`model`・`width`・`height`・`run_id`）を引き継ぐ
- diffusers・torch は `forge.load()` の中でだけ import する。`src/artgen/` のほかのモジュールとテストは、それらが無くても動く
- リポジトリは character-tactics を知らない（ファイル名・`FACE_PX`・JSON の形を書かない）。キャラクターの ID はただの名前
- 設定の誤りは `ConfigError` で止め、メッセージは `<リポジトリからの相対パス>: <キーの位置>: <何がおかしいか>` の形にする（例 `characters/ally05/character.yaml: expressions.hurt: タグの一覧でない`）
- ドキュメント・コメント・エラーメッセージは日本語。コミットは Conventional Commits + 日本語の要約

## Review Focus

1. **添付で PNG の条件が消えた画像**: `source/` に条件のない PNG を置いて書き出すと、黙って進まず「条件が無い（添付で消えた可能性）」と言って止まる → Task 3・Task 5 のテスト
2. **重み付きのタグとカンマ**: `(thin eyebrows:1.15)` や `(a, b:1.1)` をノートブックの文字列で渡しても、カンマで割れず、重みを無視して重複・削除が効く → Task 2 のテスト
3. **inpaint_box が画像の外・逆向き・未記入**: 画像の外に出る四角、x0≥x1 の四角、まだ書いていない（null）ときに、GPU を回す前に止まる → Task 4・Task 6 のテスト
4. **正方形でない顔・知らないファイル名**: `source/` に正方形でない顔や `face_xxx.png`（xxx が表情の一覧に無い）を置くと止まる → Task 5 のテスト
5. **リポジトリの中身の設定が壊れている**: 実際の `styles/`・`templates/`・`characters/ally05/` が読み込めてプロンプトが組めることを、テストでいつも確かめる → Task 7 のテスト

---

## ファイルの地図

| ファイル | 受け持ち |
|---|---|
| `pyproject.toml` | パッケージ `artgen`（src 配置）、依存 PyYAML・Pillow、`[dev]` に pytest |
| `src/artgen/config.py` | YAML の読み込みと検証。`Style`・`Template`・`Character`・`ConfigError` |
| `src/artgen/prompt.py` | タグの分割・比較・組み立て。`Overrides`・`Prompt`・`build_prompt` |
| `src/artgen/pnginfo.py` | 条件の書き込みと読み出し。`python -m artgen.pnginfo <png>` で中身を表示 |
| `src/artgen/mask.py` | 描き直す範囲の四角のマスクと、ぼかした貼り戻し |
| `src/artgen/export.py` | `source/` → 受け渡しの形。`python -m artgen.export <id>` |
| `src/artgen/forge.py` | Colab 側。`load()`・`Forge.generate`・`Forge.inpaint`・`show` |
| `styles/color-anime.yaml`、`templates/face.yaml`、`templates/fullbody.yaml`、`characters/ally05/character.yaml` | 中身の設定 |
| `notebooks/run.ipynb` | Colab の入口 |
| `requirements-colab.txt` | Colab で入れるライブラリ（版を固定） |
| `README.md`・`CLAUDE.md`・`HANDOVER.md` | 正典と引き継ぎ |

---

### Task 1: 土台と設定の読み込み

**Files:**
- Create: `pyproject.toml`、`.gitignore`（`.venv/`・`__pycache__/`・`*.egg-info/`・`.pytest_cache/`）、`src/artgen/__init__.py`（空）、`src/artgen/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `class ConfigError(Exception)`
  - `@dataclass(frozen=True) Style(name: str, prompt: list[str], negative: list[str])`
  - `@dataclass(frozen=True) Template(name: str, width: int, height: int, prompt: list[str], negative: list[str], export_size: int | None, steps: int, guidance: float, use_items: bool, inpaint_strength: float | None, inpaint_feather: int | None)`
  - `@dataclass(frozen=True) Character(id: str, name: str, style: str, subject: list[str], appearance: list[str], outfit: list[str], items: list[str], negative: list[str], default_expression: str, expressions: dict[str, list[str]], inpaint_box: tuple[int, int, int, int] | None)`
  - `load_style(root: Path, name: str) -> Style`（`styles/<name>.yaml`）
  - `load_template(root: Path, name: str) -> Template`（`templates/<name>.yaml`）
  - `load_character(root: Path, char_id: str) -> Character`（`characters/<id>/character.yaml`）

YAML の形（キー名はこのとおり）:

```yaml
# styles/<name>.yaml
prompt: [tag, ...]
negative: [tag, ...]
# templates/<name>.yaml
width: 1024
height: 1024
prompt: [...]
negative: [...]
export_size: 256        # null なら縮めない
steps: 28               # 省略時 28
guidance: 5             # 省略時 5
use_items: false        # true なら character.items を入れる。省略時 false
inpaint: {strength: 0.6, feather: 24}   # 省略可（顔の型だけが持つ）
# characters/<id>/character.yaml
name: ボルグ
style: color-anime
subject: [...]
appearance: [...]
outfit: [...]
items: [...]            # 省略時 []
negative: [...]         # 省略時 []
default_expression: normal
expressions: {normal: [...], hurt: [...]}
inpaint_box: null       # または [x0, y0, x1, y1]
```

- [ ] **Step 1: 開発環境を作る**

Run: `cd /home/ubuntu/workspace/illust-asset-forge && uv venv .venv && uv pip install -e '.[dev]'`（pyproject を先に書く）
Expected: `.venv/bin/pytest --version` が版を出す

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_config.py`（YAML は `tmp_path` に書く）:
- `test_load_character_reads_all_fields`: 上の形の YAML を読み、`c.expressions["hurt"] == ["pain"]`、`c.inpaint_box is None`、`c.items == []`（省略時）
- `test_load_template_defaults`: `steps`・`guidance`・`use_items`・`inpaint` を省いた YAML で `t.steps == 28`、`t.guidance == 5`、`t.use_items is False`、`t.inpaint_strength is None`
- `test_missing_file_names_path`: 無いキャラクターで `ConfigError`、メッセージに `characters/nobody/character.yaml` を含む
- `test_tags_must_be_list_of_strings`: `expressions: {hurt: "pain"}` で `ConfigError`、メッセージに `expressions.hurt` を含む
- `test_default_expression_must_exist`: `default_expression: smile` が `expressions` に無いと `ConfigError`、メッセージに `default_expression` と既知の表情名を含む
- `test_inpaint_box_shape`: `inpaint_box: [10, 10, 5, 20]`（x0≥x1）と `[1, 2, 3]` で `ConfigError`
- `test_missing_required_key`: `style` を省くと `ConfigError`、メッセージに `style` を含む

- [ ] **Step 3: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL（`artgen.config` が無い）

- [ ] **Step 4: `src/artgen/config.py` を実装する**

`yaml.safe_load` で読み、キーごとに型を検査する小さな関数（タグの一覧・整数・四角）で組む。パスはエラー用に `root` からの相対で出す。

- [ ] **Step 5: 通ることを確かめる**

Run: `.venv/bin/pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "feat: 設定ファイルの読み込みと検証を追加"
```

---

### Task 2: プロンプトの組み立て

**Files:**
- Create: `src/artgen/prompt.py`
- Test: `tests/test_prompt.py`

**Interfaces:**
- Consumes: Task 1 の `Style`・`Template`・`Character`・`ConfigError`
- Produces:
  - `split_tags(value: str | list[str]) -> list[str]` — 文字列はカッコの外のカンマだけで割り、前後の空白を取り、空を捨てる。一覧はそのまま（空白だけ取る）
  - `tag_key(tag: str) -> str` — 比較用の名前。外側のカッコ（`()` `[]` の重ね）と末尾の `:<数値>` を取り、小文字にし、`_` を空白にする
  - `@dataclass(frozen=True) Overrides(extra: list[str] = [], remove: list[str] = [], extra_negative: list[str] = [], remove_negative: list[str] = [])`（`field(default_factory=list)`）と `Overrides.from_args(extra="", remove="", extra_negative="", remove_negative="") -> Overrides`（`split_tags` を通す）、`Overrides.is_empty() -> bool`、`Overrides.to_dict() -> dict[str, list[str]]`（空の項目は出さない）
  - `@dataclass(frozen=True) Prompt(positive: str, negative: str)`（タグを `", "` でつないだ文字列）
  - `build_prompt(style: Style, template: Template, character: Character, expression: str | None = None, overrides: Overrides = Overrides()) -> Prompt`

並び（spec 5.2）: `subject` → `appearance` → `outfit` → `items`（`template.use_items` のときだけ）→ `template.prompt` → 表情のタグ（`expression` が None なら `default_expression`）→ `overrides.extra` → `style.prompt`。そのあと `overrides.remove` に `tag_key` が一致するタグを除き、`tag_key` が同じタグは最初の 1 つだけ残す。
ネガティブ: `style.negative` → `template.negative` → `character.negative` → `overrides.extra_negative`、`remove_negative` を除き、同じく重複を除く。
知らない表情名は `ConfigError`（メッセージに既知の表情名を並べる）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_prompt.py`（`Style`・`Template`・`Character` はテストの中で直接作る）:
- `test_split_tags_keeps_commas_inside_parens`: `split_tags("a, (b, c:1.1), d ,")` が `["a", "(b, c:1.1)", "d"]`
- `test_tag_key_ignores_weight_and_parens`: `tag_key("(Thin_Eyebrows:1.15)") == "thin eyebrows"`、`tag_key("((smile))") == "smile"`、`tag_key("[sweat]") == "sweat"`
- `test_order`: subject `["1boy"]`、appearance `["brown hair"]`、outfit `["armor"]`、items `["shield"]`、template.prompt `["portrait"]`（`use_items=False`）、表情 normal `["serious"]`、style.prompt `["masterpiece"]` で `positive == "1boy, brown hair, armor, portrait, serious, masterpiece"`
- `test_items_only_when_template_uses_them`: 同じ設定で `use_items=True` なら `"armor, shield, portrait"` の並びを含む
- `test_expression_replaces_default`: `expression="hurt"` で `"pain"` を含み `"serious"` を含まない
- `test_overrides_extra_and_remove_by_key`: `Overrides.from_args(extra="scar on cheek", remove="(serious:1.2)")` で `"scar on cheek"` を含み `"serious"` を含まない
- `test_dedupe_by_key_keeps_first`: appearance に `"(brown hair:1.2)"`、style.prompt に `"brown hair"` があると、`positive` に `brown hair` は `"(brown hair:1.2)"` の 1 回だけ
- `test_negative_merge_and_remove`: style.negative `["lowres"]`、template.negative `["full body"]`、`extra_negative="blurry"`、`remove_negative="lowres"` で `negative == "full body, blurry"`
- `test_unknown_expression`: `expression="laugh"` で `ConfigError`、メッセージに `normal` を含む
- `test_overrides_to_dict_skips_empty`: `Overrides.from_args(extra="a").to_dict() == {"extra": ["a"]}`

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_prompt.py -v`
Expected: FAIL（`artgen.prompt` が無い）

- [ ] **Step 3: `src/artgen/prompt.py` を実装する**

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/artgen/prompt.py tests/test_prompt.py
git commit -m "feat: 画風・型・キャラクターからプロンプトを組み立てる"
```

---

### Task 3: PNG の生成の条件

**Files:**
- Create: `src/artgen/pnginfo.py`
- Test: `tests/test_pnginfo.py`

**Interfaces:**
- Produces:
  - `PARAM_KEY = "parameters"`
  - `class MissingParamsError(Exception)`
  - `save_png(image: PIL.Image.Image, path: Path, params: dict) -> None`（親フォルダは作る）
  - `read_params(path: Path) -> dict` — `parameters` が無い・JSON でないときは `MissingParamsError`。メッセージにファイル名と「生成の条件が無い。添付やダウンロードで消えた可能性がある」を含む
  - `main(argv: list[str] | None = None) -> int` と `python -m artgen.pnginfo <png>...` — 各ファイルの条件を整形して表示する（`overrides` があれば目立つよう先頭に出す）。条件の無いファイルはメッセージを出し、終了コード 1

- [ ] **Step 1: 失敗するテストを書く**

- `test_roundtrip`: 8×8 の画像を `{"seed": 1, "prompt": "ボルグ, 1boy"}` で保存し、`read_params` が同じ dict を返す（日本語を含む）
- `test_missing_params_raises`: 条件なしで保存した PNG で `MissingParamsError`、メッセージにファイル名を含む
- `test_cli_reports_missing`: 条件なしの PNG で `main([str(p)]) == 1`、条件ありなら 0（`capsys` で `seed` が出力に含まれる）

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_pnginfo.py -v`
Expected: FAIL

- [ ] **Step 3: `src/artgen/pnginfo.py` を実装する**（`PngImagePlugin.PngInfo.add_text`、読み出しは `Image.open(path).text`）

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/artgen/pnginfo.py tests/test_pnginfo.py
git commit -m "feat: 生成の条件を PNG に書き込み・読み出す"
```

---

### Task 4: 描き直す範囲のマスクと貼り戻し

背景: `lpw_stable_diffusion_xl` の `inpaint` はマスクを 2 値にしてしまう（`mask_processor` が `do_binarize=True`）。そのため、パイプラインには四角のままのマスクを渡し、生成後に**ぼかしたマスクで元の絵に貼り戻す**ことで境目を目立たなくする（spec 4.4）。

**Files:**
- Create: `src/artgen/mask.py`
- Test: `tests/test_mask.py`

**Interfaces:**
- Consumes: `ConfigError`（Task 1）
- Produces:
  - `check_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> None` — 四角が画像の中に収まり、x0<x1・y0<y1 でなければ `ConfigError`（メッセージに四角と画像の大きさ）
  - `make_mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> PIL.Image.Image` — モード `L`、四角の中 255・外 0
  - `feather_composite(original: Image, generated: Image, box: tuple[int, int, int, int], feather: int) -> Image` — 四角を内側に `feather` だけ縮めたマスクを半径 `feather / 2` のガウスぼかしにかけ、それで `generated` を `original` に重ねる。2 枚の大きさが違えば `ValueError`

- [ ] **Step 1: 失敗するテストを書く**

- `test_check_box_rejects_outside_and_reversed`: 大きさ (100, 100) で `(0, 0, 101, 50)`・`(50, 10, 40, 20)` は `ConfigError`、`(10, 10, 90, 90)` は通る
- `test_make_mask_values`: `(100, 100)`・`(10, 20, 30, 40)` で (15, 25) が 255、(5, 5) が 0
- `test_feather_composite_keeps_outside_and_replaces_center`: 元は全面赤 (200, 0, 0)、生成は全面青 (0, 0, 200)、大きさ (200, 200)、四角 (50, 50, 150, 150)、feather 16 で、(100, 100) は青、(10, 10) と (49, 100)（四角のすぐ外）は赤のまま、(66, 100)（縮めた四角の端）は赤と青の両方の成分が 0 より大きい
- `test_feather_composite_size_mismatch`: 大きさが違うと `ValueError`

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_mask.py -v`
Expected: FAIL

- [ ] **Step 3: `src/artgen/mask.py` を実装する**（`ImageDraw.rectangle`、`ImageFilter.GaussianBlur`、`Image.composite`）

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/artgen/mask.py tests/test_mask.py
git commit -m "feat: 描き直す範囲のマスクとぼかした貼り戻しを追加"
```

---

### Task 5: 書き出し（受け渡しの形）

**Files:**
- Create: `src/artgen/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `load_character`・`load_template`・`ConfigError`（Task 1）、`read_params`・`MissingParamsError`（Task 3）
- Produces:
  - `class ExportError(Exception)`
  - `export_character(root: Path, char_id: str) -> list[Path]` — `characters/<id>/source/*.png` を読み、`characters/<id>/` に同じファイル名で書き出し、`characters/<id>/params.json` を書く。書いたパスの一覧を返す
  - `main(argv: list[str] | None = None) -> int` と `python -m artgen.export <id> [--root PATH]`（既定の root はカレントディレクトリ）。エラーはメッセージを出して終了コード 1

決まり:
- 受け付けるファイル名は `face.png`・`face_<表情>.png`（表情は `character.expressions` のキー）・`fullbody.png` だけ。それ以外は `ExportError`（ファイル名と受け付ける名前を示す）
- 各ファイルの条件を `read_params` で読む（無ければ `MissingParamsError` をそのまま上げる → CLI はメッセージを出して 1）
- 条件の `template` で型を読み、`export_size` が整数なら幅・高さとも `export_size` に Lanczos で縮める（元が正方形でなければ `ExportError`）。`None` なら元のファイルをそのまま写す
- 縮めた画像は条件を書き込まずに保存する（条件は `params.json` に集める）
- `params.json` は `{ファイル名: 条件}`、キーを並べ替え、`ensure_ascii=False`、`indent=2`、末尾に改行
- `source/` が無い・空なら `ExportError`

- [ ] **Step 1: 失敗するテストを書く**

テスト用に `tmp_path` へ `templates/face.yaml`（1024×1024、export_size 256）・`templates/fullbody.yaml`（832×1216、export_size null）・`styles/s.yaml`・`characters/c1/character.yaml`（表情 normal・hurt）を書くフィクスチャを作り、`source/` には `save_png` で条件（`"template": "face"` など）付きの画像を置く。

- `test_face_is_resized_to_256`: `face.png`（1024×1024）→ 書き出しは 256×256、`read_params` すると `MissingParamsError`（条件を書き込まない）
- `test_expression_face_and_fullbody`: `face_hurt.png` は 256×256、`fullbody.png`（832×1216）は 832×1216 でバイト列が元と同じ
- `test_params_json`: `params.json` のキーが `["face.png", "fullbody.png"]`、`["face.png"]["seed"]` が元の条件どおり
- `test_rejects_unknown_name`: `face_laugh.png` で `ExportError`、メッセージに `face_laugh.png` と `face_hurt.png` を含む
- `test_rejects_non_square_face`: 1024×1000 の `face.png`（条件の template は face）で `ExportError`
- `test_missing_params`: 条件なしの `face.png` で `MissingParamsError`
- `test_cli_exit_code`: `main(["c1", "--root", str(tmp_path)])` が成功で 0、`source/` が無いキャラクターで 1

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_export.py -v`
Expected: FAIL

- [ ] **Step 3: `src/artgen/export.py` を実装する**

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/artgen/export.py tests/test_export.py
git commit -m "feat: 採用した画像を受け渡しの形に書き出す"
```

---

### Task 6: Colab 側（生成・inpaint・ノートブック）

**Files:**
- Create: `src/artgen/forge.py`、`requirements-colab.txt`、`notebooks/run.ipynb`
- Test: `tests/test_forge.py`、`tests/test_notebook.py`

**Interfaces:**
- Consumes: Task 1〜4 のすべて
- Produces:
  - `MODEL_NAME = "cagliostrolab/animagine-xl-4.0"`、`SAMPLER_NAME = "Euler a"`
  - `class Forge`:
    - `__init__(self, pipe, root: Path, out_dir: Path, make_generator: Callable[[int], Any], env: dict[str, str])`
    - `generate(self, char_id: str, template: str, n: int = 4, seeds: list[int] | None = None, extra: str = "", remove: str = "", extra_negative: str = "", remove_negative: str = "", guidance: float | None = None, steps: int | None = None) -> list[Path]`
    - `inpaint(self, char_id: str, expression: str, source: str | Path, n: int = 4, seeds: list[int] | None = None, extra: str = "", remove: str = "", extra_negative: str = "", remove_negative: str = "", strength: float | None = None, guidance: float | None = None, steps: int | None = None) -> list[Path]`
  - `load(root: str | Path = ".", out_dir: str | Path = "/content/drive/MyDrive/illust-asset-forge/runs") -> Forge`
  - `show(paths: list[Path], cols: int = 4, thumb: int = 256) -> None` — 縮小した一覧を 1 枚にして `IPython.display` で出し、各ファイル名とシードを print する

決まり:
- 1 回の呼び出しで `run_id = datetime.now().strftime("%Y%m%d_%H%M%S")`、保存先 `out_dir/run_id/`。ファイル名は generate が `<id>_<template>_<seed>.png`、inpaint が `<id>_face_<expression>_<seed>.png`
- `seeds` が None なら `n` 個を `random.SystemRandom().randint(0, 2**32 - 1)` で作る。渡されたら `n` は無視する
- generate は `pipe.text2img(prompt=, negative_prompt=, width=, height=, guidance_scale=, num_inference_steps=, generator=make_generator(seed), output_type="pil").images[0]`
- inpaint は型 `face` を使う。`character.inpaint_box` が None なら GPU を回す前に `ConfigError`（「採用した顔を Claude に渡し、inpaint_box を書いてもらう」旨）。入力画像が型の大きさ（1024×1024）でなければ `ConfigError`。`check_box` を通したうえで `pipe.inpaint(prompt=, negative_prompt=, image=, mask_image=make_mask(...), width=, height=, strength=, guidance_scale=, num_inference_steps=, generator=, output_type="pil").images[0]` を呼び、`feather_composite(元, 生成, box, template.inpaint_feather)` を保存する。強さの既定は `template.inpaint_strength`
- 条件（spec 5.3）: 今のキーに加えて `character`・`template`・`style`・`expression`（generate では `default_expression`）・`overrides`（`Overrides.to_dict()`、空なら `{}`）と `env` の中身（`commit`・`diffusers`・`transformers`・`torch`・`gpu`）。inpaint では加えて `source`（入力のファイル名）・`inpaint_box`・`strength`・`feather`
- 1 枚ごとに `gc.collect()` と（`load` が渡した）後片付けの関数を呼べるよう、`Forge` は `cleanup: Callable[[], None] = lambda: None` を `__init__` のキーワード引数で受け取る
- `load()` の中身: `torch`・`diffusers` を import、`StableDiffusionXLPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, custom_pipeline="lpw_stable_diffusion_xl", custom_revision=<入れた diffusers の版>, use_safetensors=True, add_watermarker=False)`、`EulerAncestralDiscreteScheduler.from_config`、`.to("cuda")`、`enable_attention_slicing()`、`pipe.vae.enable_slicing()`・`pipe.vae.enable_tiling()`（パイプラインの同名メソッドは廃止予定の警告が出るため）、`torch.backends.cudnn.deterministic = True`・`benchmark = False`。`make_generator = lambda s: torch.Generator().manual_seed(s)`、`cleanup = torch.cuda.empty_cache`、`env` は `git rev-parse --short HEAD`（失敗したら `"unknown"`）・各ライブラリの `__version__`・`torch.cuda.get_device_name(0)`。`out_dir` が無ければ作る
- `requirements-colab.txt`: diffusers・transformers・accelerate・safetensors を `==` で固定（実装の時点の最新の安定版で、公開から 2 週間以上たったもの。transformers は選んだ diffusers と組めるもの）、PyYAML・Pillow は版を書かない（Colab に入っている）。torch は書かない
- `notebooks/run.ipynb` のセル（Markdown の見出し + コード）:
  1. 準備: Drive をつなぐ（`google.colab.drive.mount("/content/drive")`）→ `/content/illust-asset-forge` が無ければ `git clone https://github.com/akabee0161/illust-asset-forge.git`、あれば `git -C ... pull` → `pip install -q -r requirements-colab.txt` と `pip install -q -e . --no-deps` → `os.chdir` → `from artgen import forge as F; forge = F.load(".")`
  2. 顔: `paths = forge.generate("ally05", "face", n=8)` と `F.show(paths)`
  3. 表情: `paths = forge.inpaint("ally05", "hurt", source="/content/drive/MyDrive/illust-asset-forge/runs/<run_id>/<採用した顔>.png", n=4)` と `F.show(paths)`
  4. 全身: `paths = forge.generate("ally05", "fullbody", n=4)` と `F.show(paths)`
  5. 試すときの書き方の例（Markdown）: `extra="..."`・`remove="..."`・`seeds=[...]` の使い方と、「YAML は書き換えない。採用した PNG を zip にして Claude に渡す」

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_forge.py`。偽のパイプライン `FakePipe` は `text2img(**kw)` と `inpaint(**kw)` で呼び出しの引数を記録し、`SimpleNamespace(images=[Image.new("RGB", (kw["width"], kw["height"]), 色)])` を返す。`make_generator=lambda s: ("gen", s)`。設定は `tmp_path` に書く（Task 5 のフィクスチャと同じ形。キャラクターに `inpaint_box: [100, 100, 400, 300]`）。
- `test_generate_saves_with_params`: `generate("c1", "face", seeds=[7, 8])` が 2 つのパスを返し、名前は `c1_face_7.png`・`c1_face_8.png`、同じ `run_id` のフォルダ。`read_params` で `seed == 7`・`width == 1024`・`sampler == "Euler a"`・`model == MODEL_NAME`・`expression == "normal"`・`overrides == {}`・`commit` が `env` の値
- `test_generate_passes_prompt_and_generator`: `FakePipe` の記録で `generator == ("gen", 7)`、`prompt` が `build_prompt` の結果と同じ、`guidance_scale == 5`、`num_inference_steps == 28`
- `test_overrides_recorded_and_applied`: `extra="scar on cheek", guidance=6` で記録の `prompt` に `"scar on cheek"`、`overrides == {"extra": ["scar on cheek"]}`、`guidance_scale == 6`
- `test_random_seeds_count_and_range`: `n=3`・`seeds=None` で 3 枚、各 seed が 0〜2**32-1
- `test_inpaint_requires_box`: `inpaint_box: null` のキャラクターで `ConfigError`、`FakePipe` は呼ばれていない
- `test_inpaint_rejects_wrong_source_size`: 512×512 の入力で `ConfigError`、`FakePipe` は呼ばれていない
- `test_inpaint_composites_and_records`: 1024×1024 の赤い入力、`FakePipe` が青を返すとき、保存した画像の (250, 200) が青・(10, 10) が赤、名前が `c1_face_hurt_5.png`、記録に `expression == "hurt"`・`source` が入力のファイル名・`inpaint_box == [100, 100, 400, 300]`・`strength` が型の既定値、`mask_image` が渡されている

`tests/test_notebook.py`:
- `test_notebook_cells_parse`: `notebooks/run.ipynb` を JSON で読み、コードセルから `!` と `%` で始まる行を除いたものが `ast.parse` を通る。コードセルが 4 つ以上ある
- `test_notebook_uses_existing_api`: ノートブックのコードに `forge.generate(`・`forge.inpaint(`・`F.show(`・`F.load(` を含む

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_forge.py tests/test_notebook.py -v`
Expected: FAIL

- [ ] **Step 3: `src/artgen/forge.py` を実装する**（`load` と `show` の中の import は関数の中に置く。モジュールの先頭で torch・diffusers・IPython を import しない）

- [ ] **Step 4: `requirements-colab.txt` と `notebooks/run.ipynb` を書く**

版は `uv pip index versions diffusers` などで確かめる（使えなければ `pip index versions`）。確かめられなかった場合は、そのことを報告に書く。ノートブックは出力なし（`outputs: []`、`execution_count: null`）で保存する。

- [ ] **Step 5: 通ることを確かめる**

Run: `.venv/bin/pytest -v`
Expected: PASS。加えて `.venv/bin/python -c "import artgen.forge"` が torch なしで成功する

- [ ] **Step 6: Commit**

```bash
git add src/artgen/forge.py requirements-colab.txt notebooks tests/test_forge.py tests/test_notebook.py
git commit -m "feat: Colab で生成・inpaint する入口とノートブックを追加"
```

---

### Task 7: 中身の設定（ボルグの下書き）と正典

**Files:**
- Create: `styles/color-anime.yaml`、`templates/face.yaml`、`templates/fullbody.yaml`、`characters/ally05/character.yaml`、`README.md`、`CLAUDE.md`
- Modify: `HANDOVER.md`
- Test: `tests/test_repo_configs.py`

**Interfaces:**
- Consumes: Task 1・2 の `load_*`・`build_prompt`

中身（最初の版。依頼者が Colab で試しながら直す前提）:

```yaml
# styles/color-anime.yaml
prompt: [anime coloring, cel shading, clean lineart, safe, masterpiece, high score, great score, absurdres]
negative: [monochrome, greyscale, sketch, lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry, 3d, realistic]
# templates/face.yaml
width: 1024
height: 1024
prompt: [portrait, face focus, upper body, looking at viewer, front view, simple background, white background]
negative: [full body, cowboy shot, from side, from behind]
export_size: 256
use_items: false
inpaint: {strength: 0.6, feather: 24}
# templates/fullbody.yaml
width: 832
height: 1216
prompt: [full body, standing, front view, looking at viewer, simple background, white background]
negative: [close-up, portrait, cropped legs, out of frame]
export_size: null
use_items: true
# characters/ally05/character.yaml
name: ボルグ
style: color-anime
subject: [1boy, solo, adult]
appearance: [mature male, muscular, broad shoulders, square jaw, short hair, brown hair, stubble, brown eyes, thick eyebrows]
outfit: [grey armor, steel shoulder armor, breastplate, gauntlets]
items: [large shield, holding shield]
default_expression: normal
expressions:
  normal: [serious, closed mouth]
  hurt: [pain, clenched teeth, one eye closed, sweat]
  shout: [shouting, open mouth, v-shaped eyebrows]
inpaint_box: null
```

根拠（ボルグ）: 盾役の男性前衛・近接（ally05.json）、色 `#7a8a9a`（灰色がかった鋼色 → grey armor）、セリフ「まだまだ 硬くなるぞ」「ぐっ……まだ 倒れん！」（hurt）、「こっちだ！ 俺が 相手だ！」（引きつける → shout）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_repo_configs.py`（root はリポジトリ自身）:
- `test_all_characters_build_for_every_template_and_expression`: `characters/*/character.yaml` のすべてについて、`face`・`fullbody` の型と、すべての表情で `build_prompt` が通り、`positive` が空でない
- `test_face_template_matches_spec`: `face` が 1024×1024・`export_size == 256`、`fullbody` が 832×1216・`export_size is None`

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/bin/pytest tests/test_repo_configs.py -v`
Expected: FAIL（設定ファイルが無い）

- [ ] **Step 3: 上の設定ファイルを書く**

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/bin/pytest -v`
Expected: PASS

- [ ] **Step 5: README.md・CLAUDE.md・HANDOVER.md を書く**

- README.md（人間向け、pixel-asset-forge の README と同じ調子）: 何をするリポジトリか／ファイルの地図／Colab での使い方（ノートブックの各セル、試すときの `extra`・`remove`・`seeds`、YAML は書き換えないこと）／採用した絵の渡し方（Drive からダウンロード → zip → Claude に添付）／Claude 側の流れ（`source/` に `face.png`・`face_<表情>.png`・`fullbody.png` の名前で置く → `python -m artgen.pnginfo` で条件を確かめる → `overrides` を設定に戻す → `inpaint_box` を書く → `python -m artgen.export <id>`）／初めて Colab で動かすときの確認の手順（準備・顔・表情・全身の各セルを 1 回ずつ。表情は inpaint_box が要るので顔の採用の後）／ライセンス（spec 3 章の要約と出典）／再現の限界（spec 8 章）／テストの実行（`uv venv .venv && uv pip install -e '.[dev]' && .venv/bin/pytest`）
- CLAUDE.md（Claude 向けの作業規約）: GPU の処理は依頼者が Colab で動かす／Colab は GitHub から取るので設定の変更は push まで届かない、push は依頼者が頼んだときだけ／候補はリポジトリに入れない／受け取った PNG は `source/` に決まった名前で置き、必ず条件を確かめる／`overrides` は設定に戻してからコミット／inpaint_box は採用した顔を見て決める／character-tactics の名前や事情を書かない／コミットは Conventional Commits + 日本語／ドキュメントの扱い（`docs/superpowers/` はログ、正典は README.md と CLAUDE.md）
- HANDOVER.md: `## 進捗` を「工程 2（完了）、3 ← 今ここ」「段階: 実装（7 タスク中 7）」に直し、次にやることに「依頼者がボルグの設定の下書きを確認する」「GitHub にリポジトリを作って push（依頼者が頼んだら）」「依頼者が Colab で準備・顔のセルを試す」を書く

- [ ] **Step 6: Commit**

```bash
git add styles templates characters tests/test_repo_configs.py README.md CLAUDE.md HANDOVER.md
git commit -m "feat: 色付きの画風・顔と全身の型・ボルグの設定の下書きと正典を追加"
```
