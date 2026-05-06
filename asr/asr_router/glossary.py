from __future__ import annotations
from typing import Iterator


class Glossary:
    """Read-only term -> sorted aliases mapping built by merging multiple sources.

    Each source is a dict of shape `{"terms": [{"term": str, "aliases": [str, ...]}]}`.
    Merging unions aliases for the same term across sources. None sources are skipped.
    """

    def __init__(self, terms: dict[str, list[str]]):
        self._terms = terms

    @classmethod
    def merged(cls, *sources: dict | None) -> "Glossary":
        merged: dict[str, set[str]] = {}
        for src in sources:
            if not src:
                continue
            for entry in src.get("terms", []):
                t = entry["term"]
                aliases = set(entry.get("aliases") or [])
                merged.setdefault(t, set()).update(aliases)
        return cls({t: sorted(a) for t, a in merged.items()})

    def __contains__(self, term: str) -> bool:
        return term in self._terms

    def __getitem__(self, term: str) -> list[str]:
        return self._terms[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def to_prompt_text(self) -> str:
        if not self._terms:
            return "(none)"
        lines: list[str] = []
        for t in sorted(self._terms):
            aliases = self._terms[t]
            if aliases:
                lines.append(f"- {t}（同/误：{', '.join(aliases)}）")
            else:
                lines.append(f"- {t}")
        return "\n".join(lines)
