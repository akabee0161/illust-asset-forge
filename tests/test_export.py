import json

import pytest
from PIL import Image

from artgen.config import ConfigError
from artgen.export import ExportError, export_character, main
from artgen.pnginfo import MissingParamsError, read_params, save_png


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    write(tmp_path, "styles/s.yaml", "prompt: [masterpiece]\nnegative: [lowres]\n")
    write(tmp_path, "templates/face.yaml",
          "width: 1024\nheight: 1024\nprompt: [portrait]\nnegative: []\nexport_size: 256\n")
    write(tmp_path, "templates/fullbody.yaml",
          "width: 832\nheight: 1216\nprompt: [full body]\nnegative: []\nexport_size: null\n")
    write(tmp_path, "characters/c1/character.yaml",
          "name: C\nstyle: s\nsubject: [1boy]\nappearance: []\noutfit: []\n"
          "default_expression: normal\nexpressions: {normal: [serious], hurt: [pain]}\n")
    return tmp_path


def put(repo, name, size, template, seed=1, params=True):
    path = repo / "characters/c1/source" / name
    image = Image.new("RGB", size, (10, 20, 30))
    if params:
        save_png(image, path, {"seed": seed, "template": template, "prompt": "ボルグ"})
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    return path


def test_face_is_resized_to_256(repo):
    put(repo, "face.png", (1024, 1024), "face")
    export_character(repo, "c1")
    out = repo / "characters/c1/face.png"
    assert Image.open(out).size == (256, 256)
    with pytest.raises(MissingParamsError):
        read_params(out)


def test_expression_face_and_fullbody(repo):
    src = put(repo, "fullbody.png", (832, 1216), "fullbody")
    put(repo, "face_hurt.png", (1024, 1024), "face")
    export_character(repo, "c1")
    assert Image.open(repo / "characters/c1/face_hurt.png").size == (256, 256)
    out = repo / "characters/c1/fullbody.png"
    assert Image.open(out).size == (832, 1216)
    assert out.read_bytes() == src.read_bytes()


def test_params_json(repo):
    put(repo, "face.png", (1024, 1024), "face", seed=11)
    put(repo, "fullbody.png", (832, 1216), "fullbody")
    paths = export_character(repo, "c1")
    text = (repo / "characters/c1/params.json").read_text(encoding="utf-8")
    data = json.loads(text)
    assert list(data) == ["face.png", "fullbody.png"]
    assert data["face.png"]["seed"] == 11
    assert "ボルグ" in text and text.endswith("\n")
    assert repo / "characters/c1/params.json" in paths


def test_rejects_unknown_name(repo):
    put(repo, "face_laugh.png", (1024, 1024), "face")
    with pytest.raises(ExportError) as e:
        export_character(repo, "c1")
    assert "face_laugh.png" in str(e.value)
    assert "face_hurt.png" in str(e.value)


def test_rejects_non_square_face(repo):
    put(repo, "face.png", (1024, 1000), "face")
    with pytest.raises(ExportError):
        export_character(repo, "c1")


def test_missing_params(repo):
    put(repo, "face.png", (1024, 1024), "face", params=False)
    with pytest.raises(MissingParamsError):
        export_character(repo, "c1")


def test_missing_template_key_in_params(repo):
    path = repo / "characters/c1/source/face.png"
    save_png(Image.new("RGB", (1024, 1024)), path, {"seed": 1})
    with pytest.raises(ExportError) as e:
        export_character(repo, "c1")
    assert "template" in str(e.value)


def test_unknown_template_in_params(repo):
    put(repo, "face.png", (1024, 1024), "nonexistent")
    with pytest.raises(ConfigError):
        export_character(repo, "c1")


def test_nothing_written_when_any_file_is_bad(repo):
    put(repo, "face.png", (1024, 1024), "face")
    put(repo, "face_hurt.png", (1024, 1024), "face", params=False)
    with pytest.raises(MissingParamsError):
        export_character(repo, "c1")
    assert not (repo / "characters/c1/face.png").exists()
    assert not (repo / "characters/c1/params.json").exists()


def test_empty_or_missing_source(repo):
    with pytest.raises(ExportError):
        export_character(repo, "c1")
    (repo / "characters/c1/source").mkdir()
    with pytest.raises(ExportError):
        export_character(repo, "c1")


def test_cli_exit_code(repo, capsys):
    put(repo, "face.png", (1024, 1024), "face")
    assert main(["c1", "--root", str(repo)]) == 0
    write(repo, "characters/c2/character.yaml",
          "name: D\nstyle: s\nsubject: []\nappearance: []\noutfit: []\n"
          "default_expression: normal\nexpressions: {normal: []}\n")
    assert main(["c2", "--root", str(repo)]) == 1
    assert "source" in capsys.readouterr().out
    assert main(["nobody", "--root", str(repo)]) == 1
