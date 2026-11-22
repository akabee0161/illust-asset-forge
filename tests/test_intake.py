import json
import zipfile

import pytest
from PIL import Image

from artgen.config import ConfigError
from artgen.intake import extract_pack, install_selection, main
from artgen.pnginfo import MissingParamsError, save_png


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def char_yaml(extra=""):
    return (
        "name: C\nstyle: s\nsubject: [1boy]\nappearance: []\noutfit: []\n"
        "default_expression: normal\nexpressions: {normal: [serious], hurt: [pain]}\n" + extra
    )


def write_png(path, template="face", character="c1", overrides=None, params=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    if params:
        save_png(image, path, {
            "seed": 1, "template": template, "character": character,
            "overrides": overrides or {},
        })
    else:
        image.save(path)


@pytest.fixture
def repo(tmp_path):
    write(tmp_path, "styles/s.yaml", "prompt: [masterpiece]\nnegative: [lowres]\n")
    write(tmp_path, "characters/c1/character.yaml", char_yaml("inpaint_box: [1, 2, 30, 40]\n"))
    write(tmp_path, "characters/c2/character.yaml", char_yaml())
    write_png(tmp_path / "characters/c1/source/face.png")
    return tmp_path


@pytest.fixture
def selected(tmp_path):
    d = tmp_path / "selected"
    write_png(d / "c1/face.png", overrides={"extra": ["scar"], "style": "t"})
    write_png(d / "c1/face_hurt.png")
    (d / "c1/selection.json").write_text(json.dumps({"inpaint_box": [10, 20, 300, 400]}), encoding="utf-8")
    return d


@pytest.fixture
def selected_with_c2_png_in_c1(tmp_path):
    d = tmp_path / "selected"
    write_png(d / "c1/face.png", character="c2")
    return d


@pytest.fixture
def selected_plain_png(tmp_path):
    d = tmp_path / "selected"
    write_png(d / "c1/face.png", params=False)
    return d


@pytest.fixture
def pack_zip(tmp_path, selected):
    z = tmp_path / "pack.zip"
    with zipfile.ZipFile(z, "w") as f:
        for p in sorted(selected.glob("c1/*")):
            f.write(p, f"c1/{p.name}")
    return z


def test_install_replaces_source_and_reports(repo, selected):
    write_png(repo / "characters/c1/source/fullbody.png", template="fullbody")  # 古いもの
    [r] = install_selection(repo, selected)
    assert r.char_id == "c1"
    assert sorted(r.installed) == ["face.png", "face_hurt.png"]
    assert r.removed == ["fullbody.png"]
    assert sorted(p.name for p in (repo / "characters/c1/source").iterdir()) == ["face.png", "face_hurt.png"]


def test_report_lists_write_backs(repo, selected):
    [r] = install_selection(repo, selected)
    text = "\n".join(r.write_back)
    assert "face.png" in text and "scar" in text
    assert "style" in text and "t" in text
    assert "inpaint_box" in text and "[10, 20, 300, 400]" in text


def test_report_omits_unchanged_box_and_empty_overrides(repo, selected):
    (selected / "c1/selection.json").write_text(json.dumps({"inpaint_box": [1, 2, 30, 40]}), encoding="utf-8")
    write_png(selected / "c1/face.png")
    [r] = install_selection(repo, selected)
    assert r.write_back == []


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


@pytest.mark.parametrize("name", ["../evil.png", "/abs/face.png", "c1/sub/face.png", "nope/face.png", "c1/face_x.png"])
def test_extract_rejects_unsafe_or_unknown_entries(repo, tmp_path, name):
    z = tmp_path / "p.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr(name, b"x")
    before = sorted((repo / "characters/c1/source").iterdir())
    assert main([str(z), "--root", str(repo)]) == 1
    assert sorted((repo / "characters/c1/source").iterdir()) == before
    assert not (tmp_path / "evil.png").exists()


def test_extract_writes_nothing_when_any_entry_is_bad(repo, tmp_path, pack_zip):
    with zipfile.ZipFile(pack_zip, "a") as f:
        f.writestr("../evil.png", b"x")
    dest = tmp_path / "out"
    with pytest.raises(ConfigError):
        extract_pack(pack_zip, dest, root=repo)
    assert not dest.exists() or not any(dest.rglob("*"))


def test_main_rejects_bad_zip(repo, tmp_path):
    z = tmp_path / "bad.zip"
    z.write_bytes(b"not a zip")
    assert main([str(z), "--root", str(repo)]) == 1


def test_main_end_to_end(repo, pack_zip, capsys):
    assert main([str(pack_zip), "--root", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "c1" in out and "書き戻す" in out
