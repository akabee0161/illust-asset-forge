"""Colab 側の入口。パイプラインを外から受け取る作りで、GPU なしでもテストできる。

torch・diffusers・IPython は `load()` と `show()` の中でだけ import する。
"""

from __future__ import annotations

import gc
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from artgen.config import ConfigError, load_character, load_style, load_template
from artgen.mask import check_box, feather_composite, make_mask
from artgen.pnginfo import MissingParamsError, read_params, save_png
from artgen.prompt import Overrides, build_prompt

MODEL_NAME = "cagliostrolab/animagine-xl-4.0"
SAMPLER_NAME = "Euler a"
DEFAULT_OUT_DIR = "/content/drive/MyDrive/illust-asset-forge/runs"
MAX_SEED = 2**32 - 1


class Forge:
    def __init__(
        self,
        pipe,
        root: Path,
        out_dir: Path,
        make_generator: Callable[[int], Any],
        env: dict[str, str],
        *,
        cleanup: Callable[[], None] = lambda: None,
    ):
        self.pipe = pipe
        self.root = Path(root)
        self.out_dir = Path(out_dir)
        self.make_generator = make_generator
        self.env = env
        self.cleanup = cleanup

    # --- 共通 ---------------------------------------------------------

    @staticmethod
    def _seeds(n: int, seeds: list[int] | None) -> list[int]:
        if seeds is None:
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ConfigError(f"n: 1 以上の整数でない: {n!r}")
            rng = random.SystemRandom()
            return [rng.randint(0, MAX_SEED) for _ in range(n)]
        seeds = list(seeds)
        if not seeds:
            raise ConfigError("seeds: 空の一覧")
        for s in seeds:
            if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= MAX_SEED:
                raise ConfigError(f"seeds: 0〜{MAX_SEED} の整数でない: {s!r}")
        return seeds

    def _base_params(self, *, prompt, template, style, character, expression, overrides, seed,
                     guidance, steps, run_id) -> dict:
        return {
            "prompt": prompt.positive,
            "negative_prompt": prompt.negative,
            "seed": seed,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
            "sampler": SAMPLER_NAME,
            "model": MODEL_NAME,
            "width": template.width,
            "height": template.height,
            "run_id": run_id,
            "character": character.id,
            "template": template.name,
            "style": style.name,
            "expression": expression,
            "overrides": overrides.to_dict(),
            **self.env,
        }

    def _finish_one(self) -> None:
        gc.collect()
        self.cleanup()

    # --- 生成 ---------------------------------------------------------

    def generate(
        self,
        char_id: str,
        template: str,
        n: int = 4,
        seeds: list[int] | None = None,
        extra: str = "",
        remove: str = "",
        extra_negative: str = "",
        remove_negative: str = "",
        guidance: float | None = None,
        steps: int | None = None,
    ) -> list[Path]:
        character = load_character(self.root, char_id)
        style = load_style(self.root, character.style)
        tpl = load_template(self.root, template)
        overrides = Overrides.from_args(extra, remove, extra_negative, remove_negative)
        prompt = build_prompt(style, tpl, character, overrides=overrides)
        guidance = tpl.guidance if guidance is None else guidance
        steps = tpl.steps if steps is None else steps
        seed_list = self._seeds(n, seeds)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.out_dir / run_id
        paths: list[Path] = []
        for seed in seed_list:
            image = self.pipe.text2img(
                prompt=prompt.positive,
                negative_prompt=prompt.negative,
                width=tpl.width,
                height=tpl.height,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=self.make_generator(seed),
                output_type="pil",
            ).images[0]
            params = self._base_params(
                prompt=prompt, template=tpl, style=style, character=character,
                expression=character.default_expression, overrides=overrides, seed=seed,
                guidance=guidance, steps=steps, run_id=run_id,
            )
            path = run_dir / f"{char_id}_{template}_{seed}.png"
            save_png(image, path, params)
            paths.append(path)
            self._finish_one()
        return paths

    # --- 表情（inpaint） -----------------------------------------------

    def inpaint(
        self,
        char_id: str,
        expression: str,
        source: str | Path,
        n: int = 4,
        seeds: list[int] | None = None,
        extra: str = "",
        remove: str = "",
        extra_negative: str = "",
        remove_negative: str = "",
        strength: float | None = None,
        guidance: float | None = None,
        steps: int | None = None,
    ) -> list[Path]:
        character = load_character(self.root, char_id)
        style = load_style(self.root, character.style)
        tpl = load_template(self.root, "face")
        overrides = Overrides.from_args(extra, remove, extra_negative, remove_negative)
        prompt = build_prompt(style, tpl, character, expression=expression, overrides=overrides)

        # GPU を回す前に、入力の誤りをすべて止める
        if character.inpaint_box is None:
            raise ConfigError(
                f"characters/{char_id}/character.yaml: inpaint_box: 未記入。"
                "採用した顔を Claude に渡し、inpaint_box を書いてもらう"
            )
        if tpl.inpaint_strength is None or tpl.inpaint_feather is None:
            raise ConfigError("templates/face.yaml: inpaint: strength と feather が無い")
        source = Path(source)
        if not source.is_file():
            raise ConfigError(f"source: ファイルが無い: {source}")
        with Image.open(source) as im:
            original = im.convert("RGB")
        if original.size != (tpl.width, tpl.height):
            raise ConfigError(
                f"source: 画像の大きさが型 face と違う: {original.size[0]}x{original.size[1]}"
                f"（必要: {tpl.width}x{tpl.height}）"
            )
        box = character.inpaint_box
        check_box(box, original.size)
        mask = make_mask(original.size, box)

        strength = tpl.inpaint_strength if strength is None else strength
        guidance = tpl.guidance if guidance is None else guidance
        steps = tpl.steps if steps is None else steps
        seed_list = self._seeds(n, seeds)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.out_dir / run_id
        paths: list[Path] = []
        for seed in seed_list:
            generated = self.pipe.inpaint(
                prompt=prompt.positive,
                negative_prompt=prompt.negative,
                image=original,
                mask_image=mask,
                width=tpl.width,
                height=tpl.height,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=self.make_generator(seed),
                output_type="pil",
            ).images[0]
            result = feather_composite(original, generated, box, tpl.inpaint_feather)
            params = self._base_params(
                prompt=prompt, template=tpl, style=style, character=character,
                expression=expression, overrides=overrides, seed=seed,
                guidance=guidance, steps=steps, run_id=run_id,
            )
            params.update(
                source=source.name,
                inpaint_box=list(box),
                strength=strength,
                feather=tpl.inpaint_feather,
            )
            path = run_dir / f"{char_id}_face_{expression}_{seed}.png"
            save_png(result, path, params)
            paths.append(path)
            self._finish_one()
        return paths


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load(root: str | Path = ".", out_dir: str | Path = DEFAULT_OUT_DIR) -> Forge:
    """Colab でモデルを読み込む。GPU が要る（サーバーでは動かさない）。"""
    import diffusers
    import torch
    import transformers
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        custom_pipeline="lpw_stable_diffusion_xl",
        custom_revision=f"v{diffusers.__version__}",
        use_safetensors=True,
        add_watermarker=False,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    # パイプラインの同名メソッドは廃止予定の警告が出るので vae 側を呼ぶ
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    root = Path(root).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "commit": _git_commit(root),
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    return Forge(
        pipe,
        root,
        out_dir,
        make_generator=lambda s: torch.Generator().manual_seed(s),
        env=env,
        cleanup=torch.cuda.empty_cache,
    )


def show(paths: list[Path], cols: int = 4, thumb: int = 256) -> None:
    """縮小した一覧を 1 枚にして表示し、各ファイル名とシードを出す。"""
    from IPython.display import display

    paths = [Path(p) for p in paths]
    if not paths:
        print("表示する画像が無い")
        return
    rows = -(-len(paths) // cols)
    sheet = Image.new("RGB", (min(cols, len(paths)) * thumb, rows * thumb), (255, 255, 255))
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    display(sheet)
    for i, path in enumerate(paths, 1):
        try:
            seed = read_params(path).get("seed")
        except (MissingParamsError, OSError):
            seed = "不明"
        print(f"{i}. {path.name}  seed={seed}")
