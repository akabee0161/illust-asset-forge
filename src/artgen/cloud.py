"""Claude がこのサーバーで叩く CLI。Modal で候補を作り、`local/` から採用・取り込みする。

    python -m artgen.cloud generate --jobs FILE
    python -m artgen.cloud page ROUND_JSON
    python -m artgen.cloud adopt ROUND_JSON NUMBER [--flip]
    python -m artgen.cloud install CHAR [CHAR ...]

forge のフォルダ（このリポジトリの根。コピー先ではコピーを置いたフォルダ）で実行する（`--root` で変えられる）。
Modal は `generate` の中でだけ import する（ほかのコマンドとテストは Modal が無くても動く）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from artgen.config import ConfigError, load_character
from artgen.export import _accepted_names
from artgen.intake import install_selection, print_reports
from artgen.jobs import collect_faces, load_jobs
from artgen.local import local_commit, local_dirs, save_outputs
from artgen.pnginfo import MissingParamsError
from artgen.review_page import build_page
from artgen.rounds import new_round, read_round, unused_run_id
from artgen.selection import SelectionStore, _check_name_part, file_sha256

_STOPS = (ConfigError, MissingParamsError, OSError, RuntimeError)


def _page_path(round_json: Path) -> Path:
    return round_json.with_suffix(".html")


def cmd_generate(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    dirs = local_dirs(root)
    jobs = load_jobs(args.jobs, root)  # Modal を起動する前に、ジョブの誤りを止める
    faces = collect_faces(dirs.selected, jobs)
    from artgen import modal_app  # Modal はここでだけ import する

    if root != Path(modal_app.FORGE_DIR).resolve():
        # Modal に送る設定は、このパッケージのフォルダ（FORGE_DIR）から取る。別のフォルダの設定・commit と
        # 食い違う絵ができるので、generate では --root で別のフォルダを指せない
        raise ConfigError(
            f"--root: generate は、Modal に送るフォルダ（{modal_app.FORGE_DIR}）でしか使えない。"
            f"そのフォルダの .venv で実行する（指したのは {root}）"
        )
    outputs = modal_app.run(jobs, local_commit(root), faces)
    paths = save_outputs(dirs.runs, outputs)
    run_id = unused_run_id(dirs.rounds, datetime.now().strftime("%Y%m%d_%H%M%S"))
    round_json = new_round(dirs.rounds, run_id, paths)
    page = build_page(read_round(round_json), _page_path(round_json))
    print(f"{len(paths)} 枚できた")
    print(f"回りの記録: {round_json}")
    print(f"候補のページ: {page}")


def cmd_page(args: argparse.Namespace) -> None:
    round_json = args.round_json
    print(f"候補のページ: {build_page(read_round(round_json), _page_path(round_json))}")


def cmd_adopt(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    dirs = local_dirs(root)
    round_ = read_round(args.round_json)
    found = [c for c in round_.candidates if c.number == args.number]
    if not found:
        numbers = [c.number for c in round_.candidates]
        raise ConfigError(f"番号 {args.number} の候補が無い（{min(numbers)}〜{max(numbers)}）")
    candidate = found[0]
    dest = SelectionStore(dirs.selected, dirs.packs).adopt(candidate.path, flip=args.flip)
    print(f"採用した: {dest}")


def _installed_record(dirs, char: str) -> Path:
    return dirs.installed / f"{char}.json"


def _read_installed(path: Path) -> dict[str, str]:
    """前回の取り込みで `local/selected/<id>/` から入れた絵の名前と SHA-256。無ければ空。"""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"{path.name}: 取り込みの記録として読めない: {e}") from e
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ConfigError(f"{path.name}: 取り込みの記録として読めない（名前と SHA-256 の組でない）")
    return data


def cmd_install(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    dirs = local_dirs(root)
    records: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "selected"
        for char in args.chars:
            _check_name_part(char, "character")
            src = dirs.selected / char
            if not src.is_dir():
                raise ConfigError(f"{char}: 採用した絵が無い（{src}）")
            previous = _read_installed(_installed_record(dirs, char))
            # 退避したフォルダ（replaced/）や一時ファイルは持ち込まない。前回の取り込みの後に採用し直した絵だけを入れる
            # （前回入れたままの絵を入れると、その後に Colab の道などで直した source/ を古い採用に戻してしまう）
            adopted = {
                f.name: (f, file_sha256(f))
                for f in sorted(src.iterdir()) if f.is_file() and not f.name.startswith(".")
            }
            fresh = {name: f for name, (f, sha) in adopted.items() if previous.get(name) != sha}
            if not fresh:
                raise ConfigError(f"{char}: 前回の取り込みの後に採用し直した絵が無い（{src}）")
            (staged / char).mkdir(parents=True)
            for name, f in fresh.items():
                shutil.copy2(f, staged / char / name)
            # 取り込みは source/ を丸ごと置き換える。採用し直していない絵（顔・全身・表情）は今の source/ から持ち越す。
            # ただし顔を採用し直したときは、表情（face_*.png）を持ち越さない（古い顔から作ったので検査で落ちる）
            # 今の設定に無い名前（expressions から消した表情など）は持ち越さず、取り込みの「消した」に出す
            accepted = set(_accepted_names(list(load_character(root, char).expressions)))
            source_dir = root / "characters" / char / "source"
            if source_dir.is_dir():
                for kept in sorted(source_dir.iterdir()):
                    if not kept.is_file() or kept.name in fresh or kept.name not in accepted:
                        continue
                    if "face.png" in fresh and kept.name.startswith("face_"):
                        continue
                    shutil.copy2(kept, staged / char / kept.name)
            records[char] = {**previous, **{name: adopted[name][1] for name in fresh}}
        reports = install_selection(root, staged)
    dirs.installed.mkdir(parents=True, exist_ok=True)
    for char, record in records.items():
        _installed_record(dirs, char).write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    print_reports(reports)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m artgen.cloud", description="Modal で作った候補を扱う")
    parser.add_argument("--root", type=Path, default=Path("."))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate", help="ジョブのファイルから Modal で生成する")
    p.add_argument("--jobs", type=Path, required=True)
    p.set_defaults(func=cmd_generate)
    p = sub.add_parser("page", help="候補のページを作り直す")
    p.add_argument("round_json", type=Path)
    p.set_defaults(func=cmd_page)
    p = sub.add_parser("adopt", help="番号の候補を採用する")
    p.add_argument("round_json", type=Path)
    p.add_argument("number", type=int)
    p.add_argument("--flip", action="store_true", help="顔を左右反転して採用する")
    p.set_defaults(func=cmd_adopt)
    p = sub.add_parser("install", help="採用した絵を characters/<id>/source/ に取り込む")
    p.add_argument("chars", nargs="+")
    p.set_defaults(func=cmd_install)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except _STOPS as e:
        print(f"止まった: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
