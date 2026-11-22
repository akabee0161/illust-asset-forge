import zipfile
from datetime import datetime

import pytest
from PIL import Image

from artgen.config import ConfigError
from artgen.pnginfo import MissingParamsError, read_params, save_png
from artgen.selection import (
    SelectionStore,
    find_candidate,
    run_ids,
    run_paths,
    selected_name,
)


def _half_red() -> Image.Image:
    im = Image.new("RGB", (16, 16), (0, 0, 255))
    for x in range(8):
        for y in range(16):
            im.putpixel((x, y), (255, 0, 0))
    return im


def _make(path, params, image=None):
    save_png(image or Image.new("RGB", (16, 16)), path, params)
    return path


FACE = {"template": "face", "expression": "normal", "character": "c1", "seed": 7, "style": "s"}


@pytest.fixture
def store(tmp_path):
    return SelectionStore(tmp_path / "selected", tmp_path / "packs")


@pytest.fixture
def candidate(tmp_path):
    return _make(tmp_path / "cand" / "c1_face_7.png", dict(FACE))


@pytest.fixture
def half_red_candidate(tmp_path):
    return _make(tmp_path / "cand" / "c1_face_9.png", {**FACE, "seed": 9}, _half_red())


@pytest.fixture
def expression_candidate(tmp_path):
    p = {**FACE, "expression": "hurt", "source": "face.png", "seed": 3}
    return _make(tmp_path / "cand" / "c1_face_hurt_3.png", p)


@pytest.fixture
def fullbody_candidate(tmp_path):
    p = {"template": "fullbody", "character": "c1", "seed": 4, "style": "s"}
    return _make(tmp_path / "cand" / "c1_fullbody_4.png", p)


def _run(runs, run_id, files):
    for name, seed in files:
        _make(runs / run_id / name, {**FACE, "character": name.split("_")[0], "seed": seed})


@pytest.fixture
def runs(tmp_path):
    r = tmp_path / "runs"
    _run(r, "20261120_090000", [("c1_face_1.png", 1)])
    _run(r, "20261121_090000", [("c1_face_7.png", 7), ("c2_face_8.png", 8)])
    return r


@pytest.fixture
def runs_same_seed(tmp_path):
    r = tmp_path / "runs"
    _run(r, "20261121_090000", [("c1_face_7.png", 7), ("c2_face_7.png", 7)])
    return r


def test_selected_name_from_params():
    assert selected_name({"template": "face", "expression": "normal"}) == "face.png"
    assert selected_name({"template": "face", "expression": "hurt", "source": "face.png"}) == "face_hurt.png"
    assert selected_name({"template": "fullbody"}) == "fullbody.png"
    with pytest.raises(ConfigError):
        selected_name({"template": "other"})


def test_adopt_copies_with_params_and_flipped_false(store, candidate):
    dest = store.adopt(candidate)
    assert dest == store.selected_dir / "c1" / "face.png"
    assert read_params(dest)["seed"] == 7
    assert read_params(dest)["flipped"] is False


def test_adopt_flip_mirrors_and_records(store, half_red_candidate):
    dest = store.adopt(half_red_candidate, flip=True)
    with Image.open(dest) as im:
        assert im.convert("RGB").getpixel((0, 0)) == (0, 0, 255)
    assert read_params(dest)["flipped"] is True
    assert read_params(half_red_candidate).get("flipped") is None


def test_adopt_flip_only_for_face(store, expression_candidate, fullbody_candidate):
    for c in (expression_candidate, fullbody_candidate):
        with pytest.raises(ConfigError, match="左右反転は顔"):
            store.adopt(c, flip=True)


def test_readopting_face_moves_expressions_and_box(store, candidate, expression_candidate):
    store.adopt(candidate)
    store.adopt(expression_candidate)
    store.save_box("c1", (1, 2, 300, 400))
    store.adopt(candidate, now=lambda: datetime(2026, 11, 22, 10, 0, 0))
    moved = store.selected_dir / "c1" / "replaced" / "20261122_100000"
    assert (moved / "face_hurt.png").is_file() and (moved / "selection.json").is_file()
    assert not (store.selected_dir / "c1" / "face_hurt.png").exists()
    assert store.read_box("c1") is None


def test_adopt_without_params_writes_nothing(store, tmp_path):
    plain = tmp_path / "plain.png"
    Image.new("RGB", (8, 8)).save(plain)
    with pytest.raises(MissingParamsError):
        store.adopt(plain)
    assert not store.selected_dir.exists() or not any(store.selected_dir.rglob("*.png"))


def test_box_roundtrip_and_face_style(store, candidate):
    assert store.face_style("c1") is None
    store.adopt(candidate)
    assert store.face_style("c1") == "s"
    store.save_box("c1", (10, 20, 300, 400))
    assert store.read_box("c1") == (10, 20, 300, 400)


def test_read_box_broken_file_is_config_error(store):
    d = store.selected_dir / "c1"
    d.mkdir(parents=True)
    (d / "selection.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ConfigError, match="selection.json"):
        store.read_box("c1")
    (d / "selection.json").write_text('{"inpaint_box": [1, 2]}', encoding="utf-8")
    with pytest.raises(ConfigError, match="inpaint_box"):
        store.read_box("c1")


def test_pack_zips_without_replaced(store, candidate, expression_candidate):
    store.adopt(candidate)
    store.adopt(expression_candidate)
    store.adopt(candidate)
    z = store.pack(now=lambda: datetime(2026, 11, 22, 10, 0, 0))
    assert z == store.packs_dir / "selected_20261122_100000.zip"
    assert sorted(zipfile.ZipFile(z).namelist()) == ["c1/face.png"]


def test_pack_with_nothing_selected_says_so(store):
    with pytest.raises(ConfigError, match="採用した絵が無い"):
        store.pack()


def test_find_candidate_by_name_and_seed(runs):
    assert find_candidate(runs, "c1_face_7.png").name == "c1_face_7.png"
    assert find_candidate(runs, 8).name == "c2_face_8.png"
    with pytest.raises(ConfigError, match="見つからない"):
        find_candidate(runs, 999)


def test_find_candidate_ambiguous_seed_lists_names(runs_same_seed):
    with pytest.raises(ConfigError, match=r"c1_face_7\.png.*c2_face_7\.png"):
        find_candidate(runs_same_seed, 7)


def test_run_paths_latest_and_named(runs):
    assert run_ids(runs) == sorted(run_ids(runs))
    assert run_paths(runs) == run_paths(runs, run_ids(runs)[-1])
    with pytest.raises(ConfigError, match="run が無い"):
        run_paths(runs, "19990101_000000")


def test_adopt_save_failure_keeps_previous_state(store, candidate, expression_candidate, monkeypatch):
    store.adopt(candidate)
    store.adopt(expression_candidate)
    store.save_box("c1", (1, 2, 300, 400))

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("artgen.selection.save_png", boom)
    with pytest.raises(OSError):
        store.adopt(candidate)
    d = store.selected_dir / "c1"
    assert (d / "face_hurt.png").is_file() and (d / "selection.json").is_file()
    assert not (d / "replaced").exists()


def test_adopt_leaves_no_temp_file(store, candidate):
    store.adopt(candidate)
    assert sorted(p.name for p in (store.selected_dir / "c1").iterdir()) == ["face.png"]


def test_adopt_same_second_does_not_overwrite_replaced(store, candidate, expression_candidate):
    when = lambda: datetime(2026, 11, 22, 10, 0, 0)  # noqa: E731
    store.adopt(candidate)
    store.adopt(expression_candidate)
    store.adopt(candidate, now=when)
    store.adopt(expression_candidate)
    store.adopt(candidate, now=when)
    replaced = store.selected_dir / "c1" / "replaced"
    assert (replaced / "20261122_100000" / "face_hurt.png").is_file()
    assert (replaced / "20261122_100000_2" / "face_hurt.png").is_file()


def test_adopt_missing_character(store, tmp_path):
    p = {k: v for k, v in FACE.items() if k != "character"}
    c = _make(tmp_path / "x.png", p)
    with pytest.raises(ConfigError, match="character が無い"):
        store.adopt(c)


@pytest.mark.parametrize("bad", ["", "../evil", "a/b", "a\\b", "..", 5, ["c1"]])
def test_adopt_bad_character(store, tmp_path, bad):
    c = _make(tmp_path / "x.png", {**FACE, "character": bad})
    with pytest.raises(ConfigError, match="character"):
        store.adopt(c)
    assert not store.selected_dir.exists()
    assert not (tmp_path / "evil").exists()


def test_selected_name_source_without_expression():
    with pytest.raises(ConfigError, match="expression"):
        selected_name({"template": "face", "source": "face.png"})


def test_adopt_source_without_expression_writes_nothing(store, tmp_path):
    p = {k: v for k, v in FACE.items() if k != "expression"}
    c = _make(tmp_path / "x.png", {**p, "source": "face.png"})
    with pytest.raises(ConfigError, match="expression"):
        store.adopt(c)
    assert not store.selected_dir.exists()


def _expression_from(face_path, tmp_path, name="c1_face_hurt_5.png", with_hash=True):
    from artgen.selection import file_sha256

    p = {**FACE, "expression": "hurt", "source": "face.png", "seed": 5}
    if with_hash:
        p["source_sha256"] = file_sha256(face_path)
    return _make(tmp_path / "cand" / name, p)


def test_adopt_expression_made_from_current_face(store, candidate, tmp_path):
    face = store.adopt(candidate)
    expr = _expression_from(face, tmp_path)
    assert store.adopt(expr).name == "face_hurt.png"


def test_adopt_expression_from_old_face_is_rejected(store, candidate, half_red_candidate, tmp_path):
    old_face = store.adopt(candidate)
    expr = _expression_from(old_face, tmp_path)
    store.adopt(half_red_candidate)  # 顔を採用し直す
    with pytest.raises(ConfigError, match="今の顔から作っていない"):
        store.adopt(expr)
    assert not (store.selected_dir / "c1" / "face_hurt.png").exists()


def test_adopt_expression_without_face_is_rejected(store, candidate, tmp_path):
    face = store.adopt(candidate)
    expr = _expression_from(face, tmp_path)
    face.unlink()
    with pytest.raises(ConfigError, match="顔"):
        store.adopt(expr)


def test_adopt_expression_without_recorded_hash_is_not_blocked(store, candidate, tmp_path):
    """source_sha256 が無い古い PNG は、後方互換のため止めない。"""
    face = store.adopt(candidate)
    expr = _expression_from(face, tmp_path, with_hash=False)
    assert store.adopt(expr).name == "face_hurt.png"


def test_adopt_records_moved_files_and_resets(store, candidate, expression_candidate):
    store.adopt(candidate)
    store.adopt(expression_candidate)
    store.save_box("c1", (1, 2, 300, 400))
    assert store.last_moved == []
    store.adopt(candidate, now=lambda: datetime(2026, 11, 22, 10, 0, 0))
    assert sorted(store.last_moved) == ["face_hurt.png", "selection.json"]
    assert store.last_replaced == "20261122_100000"
    store.adopt(candidate)
    assert store.last_moved == [] and store.last_replaced is None


def test_find_candidate_by_name_searches_older_runs_newest_first(tmp_path):
    r = tmp_path / "runs"
    _run(r, "20261120_090000", [("c1_face_1.png", 1), ("c1_face_5.png", 5)])
    _run(r, "20261121_090000", [("c1_face_5.png", 5)])
    assert find_candidate(r, "c1_face_1.png").parent.name == "20261120_090000"
    assert find_candidate(r, "c1_face_5.png").parent.name == "20261121_090000"  # 新しい方
    assert find_candidate(r, "c1_face_5.png", run_id="20261120_090000").parent.name == "20261120_090000"
    with pytest.raises(ConfigError, match="見つからない"):
        find_candidate(r, "c1_face_1.png", run_id="20261121_090000")
    with pytest.raises(ConfigError, match="見つからない"):
        find_candidate(r, "nope.png")


def test_find_candidate_seed_still_searches_latest_run_only(runs):
    with pytest.raises(ConfigError, match="見つからない"):
        find_candidate(runs, 1)
    assert find_candidate(runs, 1, run_id="20261120_090000").name == "c1_face_1.png"


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "a\\b", ".."])
def test_selected_name_rejects_unsafe_expression(bad):
    with pytest.raises(ConfigError):
        selected_name({"template": "face", "source": "face.png", "expression": bad})


def test_pack_explicit_id_without_adopted_pngs_is_an_error(store, candidate):
    store.adopt(candidate)
    with pytest.raises(ConfigError, match="c9"):
        store.pack(["c1", "c9"])
    assert store.pack(["c1"]).is_file()


def test_adopt_expression_made_from_adopted_candidate(store, candidate, tmp_path):
    """候補の顔を source に表情を作り、その候補を採用した場合は通す。"""
    expr = _expression_from(candidate, tmp_path)
    store.adopt(candidate)
    assert store.adopt(expr).name == "face_hurt.png"
    assert read_params(store.face_path("c1"))["adopted_from_sha256"]


def test_adopt_expression_made_from_candidate_rejected_when_flipped(store, candidate, tmp_path):
    expr = _expression_from(candidate, tmp_path)
    store.adopt(candidate, flip=True)
    assert "adopted_from_sha256" not in read_params(store.face_path("c1"))
    with pytest.raises(ConfigError, match="今の顔から作っていない"):
        store.adopt(expr)


def test_adopt_expression_made_from_older_candidate_is_rejected(
    store, candidate, half_red_candidate, tmp_path
):
    expr = _expression_from(candidate, tmp_path)
    store.adopt(candidate)
    store.adopt(half_red_candidate)
    with pytest.raises(ConfigError, match="今の顔から作っていない"):
        store.adopt(expr)
