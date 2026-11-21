"""画風 × キャラクター × 型 (× 表情) からプロンプトとネガティブを組み立てる。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from artgen.config import Character, ConfigError, Style, Template

_OPEN, _CLOSE = "([", ")]"
_WEIGHT = re.compile(r":\s*-?\d+(?:\.\d+)?$")


def split_tags(value: str | list[str]) -> list[str]:
    """文字列はカッコの外のカンマだけで割る。空は捨て、前後の空白を取る。"""
    if isinstance(value, list):
        parts = list(value)
    else:
        parts, depth, start = [], 0, 0
        for i, ch in enumerate(value):
            if ch in _OPEN:
                depth += 1
            elif ch in _CLOSE:
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                parts.append(value[start:i])
                start = i + 1
        parts.append(value[start:])
    return [p.strip() for p in parts if p.strip()]


def tag_key(tag: str) -> str:
    """比較用の名前。外側のカッコと末尾の重みを取り、小文字にし、`_` を空白にする。"""
    key = tag.strip()
    while True:
        before = key
        if len(key) >= 2 and key[0] + key[-1] in ("()", "[]"):
            key = key[1:-1].strip()
        key = _WEIGHT.sub("", key).strip()
        if key == before:
            break
    return key.lower().replace("_", " ")


@dataclass(frozen=True)
class Overrides:
    extra: list[str] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    extra_negative: list[str] = field(default_factory=list)
    remove_negative: list[str] = field(default_factory=list)

    @classmethod
    def from_args(
        cls,
        extra: str | list[str] = "",
        remove: str | list[str] = "",
        extra_negative: str | list[str] = "",
        remove_negative: str | list[str] = "",
    ) -> "Overrides":
        return cls(
            extra=split_tags(extra),
            remove=split_tags(remove),
            extra_negative=split_tags(extra_negative),
            remove_negative=split_tags(remove_negative),
        )

    def is_empty(self) -> bool:
        return not (self.extra or self.remove or self.extra_negative or self.remove_negative)

    def to_dict(self) -> dict[str, list[str]]:
        items = {
            "extra": self.extra,
            "remove": self.remove,
            "extra_negative": self.extra_negative,
            "remove_negative": self.remove_negative,
        }
        return {k: list(v) for k, v in items.items() if v}


@dataclass(frozen=True)
class Prompt:
    positive: str
    negative: str


def _clean(tags: list[str], remove: list[str]) -> list[str]:
    removed = {tag_key(t) for t in remove}
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        key = tag_key(tag)
        if key in removed or key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def build_prompt(
    style: Style,
    template: Template,
    character: Character,
    expression: str | None = None,
    overrides: Overrides = Overrides(),
) -> Prompt:
    name = expression if expression is not None else character.default_expression
    if name not in character.expressions:
        known = ", ".join(character.expressions)
        raise ConfigError(
            f"characters/{character.id}/character.yaml: expressions: "
            f"表情 {name!r} が無い（既知: {known}）"
        )

    positive = [
        *character.subject,
        *character.appearance,
        *character.outfit,
        *(character.items if template.use_items else []),
        *template.prompt,
        *character.expressions[name],
        *overrides.extra,
        *style.prompt,
    ]
    negative = [
        *style.negative,
        *template.negative,
        *character.negative,
        *overrides.extra_negative,
    ]
    return Prompt(
        positive=", ".join(_clean(positive, overrides.remove)),
        negative=", ".join(_clean(negative, overrides.remove_negative)),
    )
