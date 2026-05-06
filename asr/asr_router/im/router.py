from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    upstream: str       # "sense_voice" | "omlx"
    reason: str         # human-readable rule trace


class IMRouter:
    """First-match-wins rule evaluator.

    Rule shape: {"when": {<predicate>: <value>, ...}, "use": "<upstream>"}.

    Supported predicates:
    - duration_gt: float       — true if duration_sec > value
    - duration_lt: float       — true if duration_sec < value
    - event_in: list[str]      — true if event tag is in the list
    - lang_in: list[str]       — true if lang code is in the list
    - request_param: dict      — true if every k/v pair matches request_params

    All predicates in a single `when` clause must match (AND).
    Rules are evaluated in order; first match wins. If none match, the
    default upstream is used.
    """

    def __init__(self, config: dict):
        self._default = config["defaults"]["upstream"]
        self._rules = config.get("rules", [])

    def decide(
        self,
        *,
        duration_sec: float,
        event: str,
        lang: str,
        request_params: dict,
    ) -> RouteDecision:
        for i, rule in enumerate(self._rules):
            if self._match(
                rule["when"],
                duration_sec=duration_sec,
                event=event,
                lang=lang,
                request_params=request_params,
            ):
                return RouteDecision(
                    upstream=rule["use"],
                    reason=f"rule[{i}]:{rule['when']}",
                )
        return RouteDecision(upstream=self._default, reason="default")

    @staticmethod
    def _match(
        when: dict,
        *,
        duration_sec: float,
        event: str,
        lang: str,
        request_params: dict,
    ) -> bool:
        for k, v in when.items():
            if k == "duration_gt":
                if not (duration_sec > v):
                    return False
            elif k == "duration_lt":
                if not (duration_sec < v):
                    return False
            elif k == "event_in":
                if event not in v:
                    return False
            elif k == "lang_in":
                if lang not in v:
                    return False
            elif k == "request_param":
                for pk, pv in v.items():
                    if request_params.get(pk) != pv:
                        return False
            else:
                # Unknown predicate — fail closed (rule does not match)
                return False
        return True
