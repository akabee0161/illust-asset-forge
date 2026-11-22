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
