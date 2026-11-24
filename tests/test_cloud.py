import json
import sys
import types
from pathlib import Path

import pytest

from artgen import cloud
from artgen.local import local_dirs
from artgen.rounds import new_round
from cloud_helpers import make_candidates, make_forge, paint_left_half_red
from test_forge import forge, pipe, repo, write  # noqa: F401  (fixtures)


@pytest.fixture
def cli_env(repo, monkeypatch):
    monkeypatch.chdir(repo)
    return repo


@pytest.fixture
def round_json(cli_env):
    return new_round(cli_env / "local" / "rounds", "20261123_120000", make_candidates(cli_env))


def test_adopt_by_number(cli_env, round_json, capsys):
    assert cloud.main(["adopt", str(round_json), "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("採用した:") and "警告" not in out
    assert (cli_env / "local/selected/c1/face.png").is_file()


def test_adopt_flip_mirrors_the_face(cli_env, round_json):
    from PIL import Image, ImageChops

    original = next(local_dirs(cli_env).runs.glob("c1/face/*/c1_face_2.png"))
    paint_left_half_red(original)  # 左右が非対称の絵にする（単色だと、反転しなくても一致してしまう）
    assert cloud.main(["adopt", str(round_json), "2", "--flip"]) == 0
    adopted = Image.open(cli_env / "local/selected/c1/face.png").convert("RGB")
    source = Image.open(original).convert("RGB")
    mirrored = source.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    assert ImageChops.difference(adopted, mirrored).getbbox() is None
    assert ImageChops.difference(adopted, source).getbbox() is not None  # 反転していない実装なら、ここで落ちる


def test_adopt_without_flip_keeps_the_original_orientation(cli_env, round_json):
    from PIL import Image, ImageChops

    original = next(local_dirs(cli_env).runs.glob("c1/face/*/c1_face_2.png"))
    paint_left_half_red(original)
    assert cloud.main(["adopt", str(round_json), "2"]) == 0
    adopted = Image.open(cli_env / "local/selected/c1/face.png").convert("RGB")
    assert ImageChops.difference(adopted, Image.open(original).convert("RGB")).getbbox() is None


@pytest.mark.parametrize("args, msg", [(["9"], "9"), (["3", "--flip"], "左右反転")])
def test_adopt_errors_are_one_line(cli_env, round_json, capsys, args, msg):
    assert cloud.main(["adopt", str(round_json), *args]) == 1
    out = capsys.readouterr().out
    assert out.startswith("止まった:") and msg in out and "Traceback" not in out
    assert not (cli_env / "local/selected").exists()


def test_install_only_named_characters(cli_env):
    forge_ = make_forge(cli_env)
    store = forge_.selected
    for char in ("c1", "c2"):
        store.adopt(forge_.generate(char, "face", seeds=[5])[0])
    assert cloud.main(["install", "c1"]) == 0
    assert (cli_env / "characters/c1/source/face.png").is_file()
    assert not (cli_env / "characters/c2/source").exists()


def test_install_missing_selection_is_one_line(cli_env, capsys):
    assert cloud.main(["install", "c1"]) == 1
    assert capsys.readouterr().out.startswith("止まった:")


def test_page_rebuilds_html(cli_env, round_json, capsys):
    assert cloud.main(["page", str(round_json)]) == 0
    assert (cli_env / "local/rounds/20261123_120000.html").is_file()


def fake_modal(monkeypatch, outputs, forge_dir=None):
    calls = []
    module = types.ModuleType("artgen.modal_app")
    module.FORGE_DIR = forge_dir if forge_dir is not None else Path.cwd()  # 既定は --root と同じフォルダ

    def run(jobs, commit, faces):
        calls.append((jobs, commit, faces))
        return outputs

    module.run = run
    monkeypatch.setitem(sys.modules, "artgen.modal_app", module)
    return calls


def test_generate_bad_job_file_stops_before_modal(cli_env, monkeypatch, capsys):
    calls = fake_modal(monkeypatch, [])
    jobs = cli_env / "jobs.json"
    jobs.write_text(json.dumps([{"char": "nope", "template": "face", "n": 1}]), encoding="utf-8")
    assert cloud.main(["generate", "--jobs", str(jobs)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("止まった:") and "[0].char" in out
    assert calls == []


def test_generate_saves_outputs_and_builds_round_and_page(cli_env, monkeypatch, capsys):
    made = make_candidates(cli_env)
    outputs = [(p.relative_to(cli_env / "local/runs").as_posix(), p.read_bytes()) for p in made]
    for p in made:
        p.unlink()
    calls = fake_modal(monkeypatch, outputs)
    jobs = cli_env / "jobs.json"
    jobs.write_text(json.dumps([{"char": "c1", "template": "face", "n": 1}]), encoding="utf-8")
    assert cloud.main(["generate", "--jobs", str(jobs)]) == 0
    assert len(calls) == 1 and calls[0][0][0].char == "c1" and calls[0][1]
    out = capsys.readouterr().out
    rounds = sorted((cli_env / "local/rounds").glob("*.json"))
    pages = sorted((cli_env / "local/rounds").glob("*.html"))
    assert len(rounds) == len(pages) == 1
    assert str(rounds[0]) in out or rounds[0].name in out
    assert all(p.is_file() for p in made)


def _write_source_fullbody(cli_env):
    forge_ = make_forge(cli_env)
    full = forge_.generate("c1", "fullbody", seeds=[9])[0]
    dest = cli_env / "characters/c1/source/fullbody.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(full.read_bytes())
    return dest


def test_install_keeps_existing_fullbody_that_was_not_reselected(cli_env, capsys):
    forge_ = make_forge(cli_env)
    forge_.selected.adopt(forge_.generate("c1", "face", seeds=[5])[0])
    kept = _write_source_fullbody(cli_env)
    before = kept.read_bytes()
    assert cloud.main(["install", "c1"]) == 0
    assert (cli_env / "characters/c1/source/face.png").is_file()
    assert kept.read_bytes() == before
    assert "消した" not in capsys.readouterr().out


def test_install_replaces_fullbody_when_reselected(cli_env):
    forge_ = make_forge(cli_env)
    forge_.selected.adopt(forge_.generate("c1", "face", seeds=[5])[0])
    forge_.selected.adopt(forge_.generate("c1", "fullbody", seeds=[10])[0])
    old = _write_source_fullbody(cli_env)
    old_bytes = old.read_bytes()
    assert cloud.main(["install", "c1"]) == 0
    assert old.read_bytes() != old_bytes


def test_generate_refuses_root_other_than_the_folder_sent_to_modal(cli_env, monkeypatch, capsys, tmp_path):
    calls = fake_modal(monkeypatch, [], forge_dir=tmp_path / "elsewhere")
    jobs = cli_env / "jobs.json"
    jobs.write_text(json.dumps([{"char": "c1", "template": "face", "n": 1}]), encoding="utf-8")
    assert cloud.main(["generate", "--jobs", str(jobs)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("止まった:") and "--root" in out
    assert calls == []


def test_install_does_not_bring_back_a_fullbody_installed_before(cli_env):
    # Modal で全身 A を取り込み、その後 Colab の道で source の全身を B にした。顔だけ選び直しても B のまま
    forge_ = make_forge(cli_env)
    forge_.selected.adopt(forge_.generate("c1", "face", seeds=[5])[0])
    forge_.selected.adopt(forge_.generate("c1", "fullbody", seeds=[10])[0])
    assert cloud.main(["install", "c1"]) == 0
    colab_fullbody = _write_source_fullbody(cli_env)
    b = colab_fullbody.read_bytes()
    forge_.selected.adopt(forge_.generate("c1", "face", seeds=[6])[0])
    assert cloud.main(["install", "c1"]) == 0
    assert colab_fullbody.read_bytes() == b


def test_install_fullbody_only_keeps_the_existing_face(cli_env, capsys):
    forge_ = make_forge(cli_env)
    face = forge_.generate("c1", "face", seeds=[5])[0]
    dest = cli_env / "characters/c1/source/face.png"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(face.read_bytes())
    forge_.selected.adopt(forge_.generate("c1", "fullbody", seeds=[10])[0])
    assert cloud.main(["install", "c1"]) == 0
    assert dest.read_bytes() == face.read_bytes()
    assert (cli_env / "characters/c1/source/fullbody.png").is_file()
    assert "消した" not in capsys.readouterr().out


def _write_source_expression(cli_env, name="face_hurt.png"):
    from PIL import Image

    from artgen.pnginfo import save_png

    path = cli_env / "characters/c1/source" / name
    save_png(Image.new("RGB", (8, 8)), path, {"character": "c1", "template": "face", "expression": "hurt"})
    return path


def test_install_keeps_expressions_when_the_face_was_not_reselected(cli_env):
    forge_ = make_forge(cli_env)
    forge_.selected.adopt(forge_.generate("c1", "face", seeds=[5])[0])
    assert cloud.main(["install", "c1"]) == 0
    hurt = _write_source_expression(cli_env)
    forge_.selected.adopt(forge_.generate("c1", "fullbody", seeds=[10])[0])
    assert cloud.main(["install", "c1"]) == 0
    assert hurt.is_file()


def test_install_drops_expressions_when_the_face_was_reselected(cli_env, capsys):
    forge_ = make_forge(cli_env)
    forge_.selected.adopt(forge_.generate("c1", "face", seeds=[5])[0])
    assert cloud.main(["install", "c1"]) == 0
    hurt = _write_source_expression(cli_env)
    forge_.selected.adopt(forge_.generate("c1", "face", seeds=[6])[0])
    assert cloud.main(["install", "c1"]) == 0
    assert not hurt.exists()
    assert "face_hurt.png" in capsys.readouterr().out  # 「消した」に出る


def test_install_without_new_adoption_stops(cli_env, capsys):
    forge_ = make_forge(cli_env)
    forge_.selected.adopt(forge_.generate("c1", "face", seeds=[5])[0])
    assert cloud.main(["install", "c1"]) == 0
    capsys.readouterr()
    assert cloud.main(["install", "c1"]) == 1
    assert capsys.readouterr().out.startswith("止まった:")


def test_install_rejects_path_like_character(cli_env, capsys):
    assert cloud.main(["install", "../c1"]) == 1
    assert capsys.readouterr().out.startswith("止まった:")


def _installed(cli_env):
    return json.loads((cli_env / "local/installed/c1.json").read_text(encoding="utf-8"))


def test_install_records_the_hash_of_the_bytes_it_installed(cli_env, monkeypatch):
    # 写す直前に別の adopt が local/selected の絵を置き換えても、記録は実際に取り込んだ絵の SHA-256 になる
    from artgen.selection import file_sha256

    forge_ = make_forge(cli_env)
    forge_.selected.adopt(forge_.generate("c1", "face", seeds=[5])[0])
    newer = forge_.generate("c1", "face", seeds=[6])[0]
    paint_left_half_red(newer)
    real_copy2 = cloud.shutil.copy2

    def copy_after_replacement(src, dst, *a, **kw):
        if "selected" in str(src):
            real_copy2(newer, src)  # 写す前に置き換わった
        return real_copy2(src, dst, *a, **kw)

    monkeypatch.setattr(cloud.shutil, "copy2", copy_after_replacement)
    assert cloud.main(["install", "c1"]) == 0
    assert _installed(cli_env)["face.png"] == file_sha256(cli_env / "characters/c1/source/face.png")


def test_install_keeps_the_previous_record_when_writing_fails(cli_env, monkeypatch, capsys):
    forge_ = make_forge(cli_env)
    forge_.selected.adopt(forge_.generate("c1", "face", seeds=[5])[0])
    assert cloud.main(["install", "c1"]) == 0
    before = _installed(cli_env)
    forge_.selected.adopt(forge_.generate("c1", "fullbody", seeds=[10])[0])

    def broken_replace(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cloud.os, "replace", broken_replace)
    assert cloud.main(["install", "c1"]) == 1
    assert capsys.readouterr().out.count("止まった:") == 1
    assert _installed(cli_env) == before  # 前の完全な記録が残る
    assert sorted(p.name for p in (cli_env / "local/installed").iterdir()) == ["c1.json"]  # 一時ファイルも残らない
