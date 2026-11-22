"""採用した絵の置き場（`selected/<id>/`）と、run の中の候補探し。GPU もウィジェットも使わない。"""

from __future__ import annotations

import json
import shutil
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

from artgen.config import ConfigError
from artgen.pnginfo import MissingParamsError, read_params, save_png

BOX_FILE = "selection.json"
FACE_NAME = "face.png"
_STAMP = "%Y%m%d_%H%M%S"

Box = tuple[int, int, int, int]


def selected_name(params: dict) -> str:
    template = params.get("template")
    if template == "fullbody":
        return "fullbody.png"
    if template == "face":
        if params.get("source"):
            return f"face_{params.get('expression')}.png"
        return FACE_NAME
    raise ConfigError(f"条件の template が face・fullbody のどちらでもない: {template!r}")


class SelectionStore:
    def __init__(self, selected_dir: Path, packs_dir: Path):
        self.selected_dir = Path(selected_dir)
        self.packs_dir = Path(packs_dir)

    def face_path(self, char_id: str) -> Path:
        return self.selected_dir / char_id / FACE_NAME

    def face_style(self, char_id: str) -> str | None:
        path = self.face_path(char_id)
        if not path.is_file():
            return None
        return read_params(path).get("style")

    def adopt(
        self,
        candidate: Path,
        flip: bool = False,
        *,
        now: Callable[[], datetime] = datetime.now,
    ) -> Path:
        candidate = Path(candidate)
        # 何も書く前に検査する
        params = read_params(candidate)
        name = selected_name(params)
        char_id = params.get("character")
        if not char_id:
            raise ConfigError(f"{candidate.name}: 条件に character が無い")
        if flip and name != FACE_NAME:
            raise ConfigError(f"{candidate.name}: 左右反転は顔（{FACE_NAME}）だけに使える")
        with Image.open(candidate) as im:
            image = im.copy()
        if flip:
            image = ImageOps.mirror(image)

        char_dir = self.selected_dir / str(char_id)
        if name == FACE_NAME:
            self._move_aside(char_dir, now())
        save_png(image, char_dir / name, {**params, "flipped": bool(flip)})
        return char_dir / name

    def _move_aside(self, char_dir: Path, when: datetime) -> None:
        targets = sorted(char_dir.glob("face_*.png")) + [
            p for p in [char_dir / BOX_FILE] if p.is_file()
        ]
        if not targets:
            return
        dest = char_dir / "replaced" / when.strftime(_STAMP)
        dest.mkdir(parents=True, exist_ok=True)
        for p in targets:
            shutil.move(str(p), dest / p.name)
            print(f"{char_dir.name}/{p.name} を replaced/{dest.name}/ へ移した")

    def read_box(self, char_id: str) -> Box | None:
        path = self.selected_dir / char_id / BOX_FILE
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise ConfigError(f"{char_id}/{BOX_FILE}: 読めない: {e}") from e
        box = data.get("inpaint_box") if isinstance(data, dict) else None
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in box)
        ):
            raise ConfigError(f"{char_id}/{BOX_FILE}: inpaint_box: 整数 4 つの [x0, y0, x1, y1] でない")
        return (box[0], box[1], box[2], box[3])

    def save_box(self, char_id: str, box: Box) -> Path:
        path = self.selected_dir / char_id / BOX_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"inpaint_box": list(box)}), encoding="utf-8")
        return path

    def characters(self) -> list[str]:
        if not self.selected_dir.is_dir():
            return []
        return sorted(d.name for d in self.selected_dir.iterdir() if d.is_dir() and any(d.glob("*.png")))

    def pack(
        self,
        char_ids: list[str] | None = None,
        *,
        now: Callable[[], datetime] = datetime.now,
    ) -> Path:
        ids = self.characters() if char_ids is None else list(char_ids)
        files: list[tuple[Path, str]] = []
        for cid in ids:
            d = self.selected_dir / cid
            for p in sorted(d.glob("*.png")) + [d / BOX_FILE]:
                if p.is_file():
                    files.append((p, f"{cid}/{p.name}"))
        if not files:
            raise ConfigError("採用した絵が無い")
        self.packs_dir.mkdir(parents=True, exist_ok=True)
        out = self.packs_dir / f"selected_{now().strftime(_STAMP)}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p, arc in files:
                z.write(p, arc)
        return out


def run_ids(runs_dir: Path) -> list[str]:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    return sorted(d.name for d in runs_dir.iterdir() if d.is_dir())


def run_paths(runs_dir: Path, run_id: str | None = None) -> list[Path]:
    ids = run_ids(runs_dir)
    if run_id is None:
        if not ids:
            raise ConfigError(f"{runs_dir}: run が無い")
        run_id = ids[-1]
    elif run_id not in ids:
        raise ConfigError(f"{runs_dir}: run が無い: {run_id}")
    return sorted((Path(runs_dir) / run_id).glob("*.png"))


def find_candidate(runs_dir: Path, key: str | int, run_id: str | None = None) -> Path:
    paths = run_paths(runs_dir, run_id)
    if isinstance(key, str):
        found = [p for p in paths if p.name == key]
    else:
        found = []
        for p in paths:
            try:
                if read_params(p).get("seed") == key:
                    found.append(p)
            except MissingParamsError:
                continue
    if not found:
        raise ConfigError(f"候補が見つからない: {key}")
    if len(found) > 1:
        names = "・".join(p.name for p in found)
        raise ConfigError(f"シード {key} の候補が複数ある（{names}）。ファイル名で指定する")
    return found[0]
