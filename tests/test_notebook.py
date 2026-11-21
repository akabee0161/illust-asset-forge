import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "run.ipynb"


def code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return nb, ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_cells_parse():
    _, cells = code_cells()
    assert len(cells) >= 4
    for source in cells:
        lines = [ln for ln in source.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        ast.parse("\n".join(lines))


def test_notebook_uses_existing_api():
    _, cells = code_cells()
    code = "\n".join(cells)
    for needle in ("forge.generate(", "forge.inpaint(", "on_image=F.show_one", "F.load("):
        assert needle in code


def test_notebook_has_no_outputs():
    nb, _ = code_cells()
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == []
            assert c["execution_count"] is None


def test_notebook_makes_package_importable_in_running_kernel():
    # pip install -e は動いているカーネルの import に効かないので、src を sys.path に足す
    _, cells = code_cells()
    setup = next(c for c in cells if "from artgen import forge" in c)
    assert "sys.path.insert" in setup
    assert setup.index("sys.path.insert") < setup.index("from artgen import forge")


def test_notebook_load_cell_is_separate_and_skips_if_already_loaded():
    _, cells = code_cells()
    setup = next(c for c in cells if "from artgen import forge" in c)
    assert "F.load(" not in setup  # 設定を pull し直すだけのときにモデルを読み込まない
    load = next(c for c in cells if "F.load(" in c)
    assert 'in globals()' in load


def test_notebook_purges_cached_artgen_modules_before_import():
    # 実行し直しても古いコードのままになるのを防ぐ。import より前に sys.modules から消す
    _, cells = code_cells()
    setup = next(c for c in cells if "from artgen import forge" in c)
    assert "sys.modules" in setup
    assert setup.index("sys.modules") < setup.index("from artgen import forge")
