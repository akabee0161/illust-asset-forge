import re
import sys

import pytest
from PIL import Image
from types import SimpleNamespace

from artgen.config import ConfigError, load_character, load_style, load_template
from artgen.forge import MODEL_NAME, Forge
from artgen.pnginfo import read_params
from artgen.prompt import build_prompt

ENV = {"commit": "abc1234", "diffusers": "0.0", "transformers": "0.0", "torch": "0.0", "gpu": "FakeGPU"}


class FakePipe:
    def __init__(self, color=(0, 0, 200)):
        self.color = color
        self.calls = []

    def _run(self, kind, kw):
        self.calls.append((kind, kw))
        return SimpleNamespace(images=[Image.new("RGB", (kw["width"], kw["height"]), self.color)])

    def text2img(self, **kw):
        return self._run("text2img", kw)

    def inpaint(self, **kw):
        return self._run("inpaint", kw)


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


CHARACTER = (
    "name: C\nstyle: s\nsubject: [1boy]\nappearance: [brown hair]\noutfit: [armor]\n"
    "default_expression: normal\nexpressions: {{normal: [serious], hurt: [pain]}}\n"
    "inpaint_box: {box}\n"
)


@pytest.fixture
def repo(tmp_path):
    write(tmp_path, "styles/s.yaml", "prompt: [masterpiece]\nnegative: [lowres]\n")
    write(tmp_path, "templates/face.yaml",
          "width: 1024\nheight: 1024\nprompt: [portrait]\nnegative: [full body]\n"
          "export_size: 256\ninpaint: {strength: 0.6, feather: 24}\n")
    write(tmp_path, "templates/fullbody.yaml",
          "width: 832\nheight: 1216\nprompt: [full body]\nnegative: []\nexport_size: null\n")
    write(tmp_path, "characters/c1/character.yaml", CHARACTER.format(box="[100, 100, 400, 300]"))
    write(tmp_path, "characters/c2/character.yaml", CHARACTER.format(box="[100, 100, 400, 300]"))
    write(tmp_path, "styles/t.yaml", "prompt: [watercolor]\nnegative: []\n")
    write(tmp_path, "characters/nobox/character.yaml", CHARACTER.format(box="null"))
    return tmp_path


@pytest.fixture
def pipe():
    return FakePipe()


@pytest.fixture
def forge(repo, pipe, tmp_path):
    return Forge(pipe, repo, tmp_path / "runs", lambda s: ("gen", s), dict(ENV))


def test_generate_saves_with_params(forge, repo, tmp_path):
    paths = forge.generate("c1", "face", seeds=[7, 8])
    assert [p.name for p in paths] == ["c1_face_7.png", "c1_face_8.png"]
    assert paths[0].parent == paths[1].parent
    assert paths[0].parent.parent == tmp_path / "runs"
    assert re.fullmatch(r"\d{8}_\d{6}", paths[0].parent.name)
    params = read_params(paths[0])
    assert params["seed"] == 7
    assert params["width"] == 1024 and params["height"] == 1024
    assert params["sampler"] == "Euler a"
    assert params["model"] == MODEL_NAME
    assert params["expression"] == "normal"
    assert params["overrides"] == {}
    assert params["commit"] == "abc1234"
    assert params["gpu"] == "FakeGPU"
    assert params["character"] == "c1" and params["template"] == "face" and params["style"] == "s"
    assert params["run_id"] == paths[0].parent.name
    assert Image.open(paths[0]).size == (1024, 1024)


def test_generate_passes_prompt_and_generator(forge, repo, pipe):
    forge.generate("c1", "face", seeds=[7])
    kind, kw = pipe.calls[0]
    expected = build_prompt(
        load_style(repo, "s"), load_template(repo, "face"), load_character(repo, "c1")
    )
    assert kind == "text2img"
    assert kw["generator"] == ("gen", 7)
    assert kw["prompt"] == expected.positive
    assert kw["negative_prompt"] == expected.negative
    assert kw["guidance_scale"] == 5
    assert kw["num_inference_steps"] == 28
    assert kw["output_type"] == "pil"
    assert (kw["width"], kw["height"]) == (1024, 1024)


def test_overrides_recorded_and_applied(forge, pipe):
    paths = forge.generate("c1", "face", seeds=[1], extra="scar on cheek", guidance=6, steps=20)
    params = read_params(paths[0])
    assert "scar on cheek" in params["prompt"]
    assert params["overrides"] == {"extra": ["scar on cheek"], "guidance": 6, "steps": 20}
    assert params["guidance_scale"] == 6
    assert params["num_inference_steps"] == 20
    assert pipe.calls[0][1]["guidance_scale"] == 6


def test_random_seeds_count_and_range(forge):
    paths = forge.generate("c1", "face", n=3)
    assert len(paths) == 3
    for p in paths:
        assert 0 <= read_params(p)["seed"] <= 2**32 - 1


def test_seeds_override_n(forge):
    assert len(forge.generate("c1", "face", n=5, seeds=[1, 2])) == 2


def test_generate_rejects_bad_seed_before_gpu(forge, pipe):
    with pytest.raises(ConfigError):
        forge.generate("c1", "face", seeds=[-1])
    with pytest.raises(ConfigError):
        forge.generate("c1", "face", n=0)
    assert pipe.calls == []


def test_generate_unknown_character_before_gpu(forge, pipe):
    with pytest.raises(ConfigError):
        forge.generate("nobody", "face", n=1)
    assert pipe.calls == []


def test_cleanup_called_per_image(repo, pipe, tmp_path):
    count = []
    f = Forge(pipe, repo, tmp_path / "runs", lambda s: s, dict(ENV), cleanup=lambda: count.append(1))
    f.generate("c1", "face", seeds=[1, 2, 3])
    assert len(count) == 3


def make_source(tmp_path, size=(1024, 1024), color=(200, 0, 0)):
    p = tmp_path / "picked.png"
    Image.new("RGB", size, color).save(p)
    return p


def test_inpaint_requires_box(forge, pipe, tmp_path):
    with pytest.raises(ConfigError) as e:
        forge.inpaint("nobox", "hurt", make_source(tmp_path), n=1)
    assert "inpaint_box" in str(e.value)
    assert pipe.calls == []


def test_inpaint_rejects_wrong_source_size(forge, pipe, tmp_path):
    with pytest.raises(ConfigError):
        forge.inpaint("c1", "hurt", make_source(tmp_path, (512, 512)), n=1)
    assert pipe.calls == []


def test_inpaint_rejects_box_outside_image_before_gpu(repo, pipe, tmp_path):
    write(repo, "characters/c1/character.yaml", CHARACTER.format(box="[100, 100, 2000, 300]"))
    f = Forge(pipe, repo, tmp_path / "runs", lambda s: s, dict(ENV))
    with pytest.raises(ConfigError):
        f.inpaint("c1", "hurt", make_source(tmp_path), n=1)
    assert pipe.calls == []


def test_inpaint_rejects_unknown_expression_before_gpu(forge, pipe, tmp_path):
    with pytest.raises(ConfigError):
        forge.inpaint("c1", "laugh", make_source(tmp_path), n=1)
    assert pipe.calls == []


def test_inpaint_missing_source_before_gpu(forge, pipe, tmp_path):
    with pytest.raises(ConfigError):
        forge.inpaint("c1", "hurt", tmp_path / "nope.png", n=1)
    assert pipe.calls == []


def test_inpaint_composites_and_records(forge, pipe, tmp_path):
    src = make_source(tmp_path)
    paths = forge.inpaint("c1", "hurt", src, seeds=[5])
    assert [p.name for p in paths] == ["c1_face_hurt_5.png"]
    out = Image.open(paths[0])
    assert out.size == (1024, 1024)
    assert out.getpixel((250, 200)) == (0, 0, 200)
    assert out.getpixel((10, 10)) == (200, 0, 0)

    params = read_params(paths[0])
    assert params["expression"] == "hurt"
    assert params["source"] == "picked.png"
    assert params["inpaint_box"] == [100, 100, 400, 300]
    assert params["strength"] == 0.6
    assert params["feather"] == 24
    assert params["template"] == "face"

    kind, kw = pipe.calls[0]
    assert kind == "inpaint"
    assert kw["mask_image"].size == (1024, 1024)
    assert kw["mask_image"].getpixel((250, 200)) == 255
    assert kw["image"].size == (1024, 1024)
    assert kw["strength"] == 0.6
    assert kw["generator"] == ("gen", 5)
    assert "pain" in kw["prompt"]


def test_module_imports_without_torch():
    import artgen.forge  # noqa: F401

    assert "torch" not in sys.modules
    assert "diffusers" not in sys.modules


def test_show_displays_one_sheet_and_prints_seeds(forge, monkeypatch, capsys):
    shown = []
    fake_display = SimpleNamespace(display=lambda obj: shown.append(obj))
    monkeypatch.setitem(sys.modules, "IPython", SimpleNamespace(display=fake_display))
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)
    from artgen.forge import show

    paths = forge.generate("c1", "face", seeds=[7, 8, 9])
    show(paths, cols=2, thumb=64)
    assert len(shown) == 1 and shown[0].size == (128, 128)
    out = capsys.readouterr().out
    assert "c1_face_7.png" in out and "seed=9" in out


def test_inpaint_rejects_feather_too_large_before_gpu(repo, pipe, tmp_path):
    write(repo, "characters/c1/character.yaml", CHARACTER.format(box="[100, 100, 140, 300]"))
    f = Forge(pipe, repo, tmp_path / "runs", lambda s: s, dict(ENV))
    with pytest.raises(ConfigError):
        f.inpaint("c1", "hurt", make_source(tmp_path), n=1)
    assert pipe.calls == []


def test_inpaint_box_error_names_the_character_file(repo, pipe, tmp_path):
    write(repo, "characters/c1/character.yaml", CHARACTER.format(box="[100, 100, 2000, 300]"))
    f = Forge(pipe, repo, tmp_path / "runs", lambda s: s, dict(ENV))
    with pytest.raises(ConfigError) as e:
        f.inpaint("c1", "hurt", make_source(tmp_path), n=1)
    assert str(e.value).startswith("characters/c1/character.yaml: inpaint_box")


@pytest.mark.parametrize("kw", [{"guidance": 0}, {"guidance": -1}, {"steps": 0}, {"steps": 1.5}])
def test_generate_rejects_bad_sampling_args_before_gpu(forge, pipe, kw):
    with pytest.raises(ConfigError):
        forge.generate("c1", "face", seeds=[1], **kw)
    assert pipe.calls == []


@pytest.mark.parametrize("kw", [{"strength": 0}, {"strength": 1.5}, {"steps": 0}, {"guidance": 0}])
def test_inpaint_rejects_bad_sampling_args_before_gpu(forge, pipe, tmp_path, kw):
    with pytest.raises(ConfigError):
        forge.inpaint("c1", "hurt", make_source(tmp_path), seeds=[1], **kw)
    assert pipe.calls == []


def test_overridden_sampling_values_are_recorded_in_overrides(forge, tmp_path):
    p = forge.generate("c1", "face", seeds=[1], guidance=6, steps=20)[0]
    assert read_params(p)["overrides"] == {"guidance": 6, "steps": 20}
    p = forge.inpaint("c1", "hurt", make_source(tmp_path), seeds=[1], strength=0.4)[0]
    assert read_params(p)["overrides"] == {"strength": 0.4}
    p = forge.generate("c1", "face", seeds=[1])[0]
    assert read_params(p)["overrides"] == {}


def fake_gpu_libs(monkeypatch):
    """torch・diffusers・transformers の偽物を入れ、from_pretrained の引数と呼ばれた設定を記録する。"""
    rec = SimpleNamespace(kwargs={}, calls=[])

    class FakeVae:
        def enable_slicing(self): rec.calls.append("vae_slicing")
        def enable_tiling(self): rec.calls.append("vae_tiling")

    class FakePipeObj:
        scheduler = SimpleNamespace(config={})
        vae = FakeVae()
        def to(self, device): return self
        def enable_attention_slicing(self): rec.calls.append("attention_slicing")

    class FakeSDXL:
        @staticmethod
        def from_pretrained(name, **kw):
            rec.kwargs.update(kw)
            return FakePipeObj()

    fake_diffusers = SimpleNamespace(
        __version__="0.42.1",
        StableDiffusionXLPipeline=FakeSDXL,
        EulerAncestralDiscreteScheduler=SimpleNamespace(from_config=lambda c: "sched"),
    )
    fake_torch = SimpleNamespace(
        __version__="2.0", float16="fp16",
        backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True)),
        cuda=SimpleNamespace(get_device_name=lambda i: "FakeGPU", empty_cache=lambda: None),
        Generator=lambda: SimpleNamespace(manual_seed=lambda s: ("gen", s)),
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="5.0"))
    return rec


def test_load_passes_custom_revision_without_v_prefix(monkeypatch, tmp_path):
    # diffusers は custom_revision を PyPI の版名（"0.42.1"）と比べ、自分で "v" を付ける。
    # "v0.42.1" を渡すと「存在しない」と止まる
    rec = fake_gpu_libs(monkeypatch)
    from artgen.forge import load

    forge = load(tmp_path, tmp_path / "runs")
    assert rec.kwargs["custom_revision"] == "0.42.1"
    assert rec.kwargs["custom_pipeline"] == "lpw_stable_diffusion_xl"
    assert forge.env["diffusers"] == "0.42.1" and forge.env["gpu"] == "FakeGPU"


def test_load_skips_memory_saving_by_default_for_speed(monkeypatch, tmp_path):
    rec = fake_gpu_libs(monkeypatch)
    from artgen.forge import load

    load(tmp_path, tmp_path / "runs")
    assert "attention_slicing" not in rec.calls and "vae_tiling" not in rec.calls


def test_load_low_memory_enables_slicing_and_tiling(monkeypatch, tmp_path):
    rec = fake_gpu_libs(monkeypatch)
    from artgen.forge import load

    load(tmp_path, tmp_path / "runs", low_memory=True)
    assert {"attention_slicing", "vae_slicing", "vae_tiling"} <= set(rec.calls)


def test_on_image_called_after_each_saved_image(forge):
    seen = []
    paths = forge.generate("c1", "face", seeds=[1, 2], on_image=lambda p: seen.append((p, p.exists())))
    assert [p for p, _ in seen] == paths
    assert all(exists for _, exists in seen)  # 保存してから呼ぶ


def test_inpaint_calls_on_image(forge, tmp_path):
    seen = []
    forge.inpaint("c1", "hurt", make_source(tmp_path), seeds=[1, 2], on_image=seen.append)
    assert len(seen) == 2


def test_interrupt_returns_images_saved_so_far(repo, tmp_path, capsys):
    class InterruptingPipe(FakePipe):
        def text2img(self, **kw):
            if len(self.calls) == 1:
                raise KeyboardInterrupt
            return super().text2img(**kw)

    f = Forge(InterruptingPipe(), repo, tmp_path / "runs", lambda s: s, dict(ENV))
    paths = f.generate("c1", "face", seeds=[1, 2, 3])
    assert [p.name for p in paths] == ["c1_face_1.png"]
    assert paths[0].exists()
    out = capsys.readouterr().out
    assert "中断" in out and "1 枚" in out


def test_show_one_displays_image_and_seed(forge, monkeypatch, capsys):
    shown = []
    fake_display = SimpleNamespace(display=lambda obj: shown.append(obj))
    monkeypatch.setitem(sys.modules, "IPython", SimpleNamespace(display=fake_display))
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)
    from artgen.forge import show_one

    path = forge.generate("c1", "face", seeds=[7])[0]
    capsys.readouterr()
    show_one(path, thumb=128)
    out = capsys.readouterr().out
    assert len(shown) == 1 and max(shown[0].size) == 128
    assert "c1_face_7.png" in out and "seed=7" in out


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
