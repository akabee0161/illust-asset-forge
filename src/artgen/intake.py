"""採用した絵の zip を `characters/<id>/source/` に取り込む。GPU は要らない。

`install_selection` は zip・標準出力に依らない（Colab 側から直接呼べる）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from artgen.config import ConfigError, load_character
from artgen.export import _accepted_names
from artgen.pnginfo import MissingParamsError, read_params

BOX_FILE = "selection.json"


@dataclass
class CharacterReport:
    char_id: str
    installed: list[str]
    removed: list[str]
    write_back: list[str]


def _format_box(box) -> str:
    return "[" + ", ".join(str(v) for v in box) + "]"


def _read_box(selection: Path, rel: str) -> list[int] | None:
    try:
        data = json.loads(selection.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise ConfigError(f"{rel}: 読めない: {e}") from e
    box = data.get("inpaint_box") if isinstance(data, dict) else None
    if box is None:
        return None
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(isinstance(v, bool) or not isinstance(v, int) for v in box)
    ):
        raise ConfigError(f"{rel}: inpaint_box: 整数 4 つの [x0, y0, x1, y1] でない")
    return box


def _visible_files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith("."))


def install_selection(root: Path, selected_dir: Path) -> list[CharacterReport]:
    """`selected_dir/<id>/` ごとに `characters/<id>/source/` を丸ごと置き換える。

    すべてのキャラクター・ファイルを検査してから書き始めるので、途中で止まっても何も書かれない。
    """
    root, selected_dir = Path(root), Path(selected_dir)
    char_dirs = sorted(p for p in selected_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not char_dirs:
        raise ConfigError(f"{selected_dir.name}: キャラクターのフォルダが無い")

    plans: list[tuple[str, list[Path], CharacterReport]] = []
    for char_dir in char_dirs:
        char_id = char_dir.name
        character = load_character(root, char_id)
        accepted = _accepted_names(list(character.expressions))
        pngs: list[Path] = []
        write_back: list[str] = []
        selection_box: list[int] | None = None
        for p in _visible_files(char_dir):
            where = f"{char_id}/{p.name}"
            if p.name == BOX_FILE:
                selection_box = _read_box(p, where)
                continue
            if p.name not in accepted:
                raise ConfigError(f"{where}: 受け付けない名前（受け付ける名前: {', '.join(accepted)}）")
            params = read_params(p)  # 無ければ MissingParamsError のまま上げる
            owner = params.get("character")
            if owner != char_id:
                raise ConfigError(f"{where}: 条件の character が {owner!r}（{char_id!r} の絵でない）")
            pngs.append(p)
            overrides = params.get("overrides")
            if overrides:
                write_back.append(f"{p.name}: overrides: {json.dumps(overrides, ensure_ascii=False)}")
        if not pngs:
            raise ConfigError(f"{char_id}: 取り込む PNG が無い")
        if selection_box is not None and selection_box != list(character.inpaint_box or ()):
            write_back.append(f"inpaint_box: {_format_box(selection_box)} を character.yaml へ")

        source_dir = root / "characters" / char_id / "source"
        old = {p.name for p in _visible_files(source_dir)} if source_dir.is_dir() else set()
        new = {p.name for p in pngs}
        report = CharacterReport(
            char_id=char_id,
            installed=sorted(new),
            removed=sorted(old - new),
            write_back=write_back,
        )
        plans.append((char_id, pngs, report))

    for char_id, pngs, _ in plans:
        char_root = root / "characters" / char_id
        staging = char_root / ".source.new"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        for p in pngs:
            shutil.copyfile(p, staging / p.name)
        source_dir = char_root / "source"
        backup = char_root / ".source.old"
        shutil.rmtree(backup, ignore_errors=True)
        if source_dir.exists():
            source_dir.rename(backup)
        staging.rename(source_dir)
        shutil.rmtree(backup, ignore_errors=True)
    return [report for _, _, report in plans]


def extract_pack(zip_path: Path, dest: Path, *, root: Path) -> Path:
    """zip を `dest` に安全に展開し、`selected/` と同じ形のフォルダ（`dest`）を返す。

    すべてのエントリ（`<id>/<名前>` の 2 段・`..` や絶対パスが無い・既知の ID・受け付ける名前）を
    確かめてから、1 つずつ書く。
    """
    zip_path, dest, root = Path(zip_path), Path(dest), Path(root)
    expressions: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path) as z:
        members: list[tuple[zipfile.ZipInfo, str, str]] = []
        for info in z.infolist():
            raw = info.filename
            where = f"{zip_path.name}: {raw}"
            if "\\" in raw or raw.startswith("/") or (len(raw) > 1 and raw[1] == ":"):
                raise ConfigError(f"{where}: 絶対パス・不正な区切りは受け付けない")
            parts = PurePosixPath(raw).parts
            if ".." in parts or "." in parts:
                raise ConfigError(f"{where}: '..' を含むパスは受け付けない")
            if info.is_dir():
                if len(parts) != 1:
                    raise ConfigError(f"{where}: <id>/<名前> の 2 段でない")
                continue
            if len(parts) != 2:
                raise ConfigError(f"{where}: <id>/<名前> の 2 段でない")
            char_id, name = parts
            if name.startswith("."):
                continue
            if not (root / "characters" / char_id / "character.yaml").is_file():
                raise ConfigError(f"{where}: 知らないキャラクター {char_id!r}（characters/{char_id}/character.yaml が無い）")
            if char_id not in expressions:
                expressions[char_id] = list(load_character(root, char_id).expressions)
            accepted = _accepted_names(expressions[char_id])
            if name != BOX_FILE and name not in accepted:
                raise ConfigError(f"{where}: 受け付けない名前（受け付ける名前: {', '.join(accepted)}）")
            members.append((info, char_id, name))

        dest.mkdir(parents=True, exist_ok=True)
        for info, char_id, name in members:
            target = dest / char_id / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m artgen.intake", description="採用した絵の zip を取り込む")
    parser.add_argument("zip", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            selected = extract_pack(args.zip, Path(tmp) / "selected", root=args.root)
            reports = install_selection(args.root, selected)
    except (ConfigError, MissingParamsError, zipfile.BadZipFile, OSError) as e:
        print(f"取り込めない: {e}")
        return 1
    for r in reports:
        print(f"{r.char_id}: 取り込んだ: {', '.join(r.installed)}")
        if r.removed:
            print(f"{r.char_id}: 消した: {', '.join(r.removed)}")
        for line in r.write_back:
            print(f"  書き戻し: {line}")
    print("次: 上の一覧を character.yaml に書き戻す。そのあと、キャラクターごとに")
    for r in reports:
        print(f"  python -m artgen.export {r.char_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
