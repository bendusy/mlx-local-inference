from asr_router.im.router import IMRouter, RouteDecision


ROUTING = {
    "defaults": {"upstream": "sense_voice"},
    "rules": [
        {"when": {"request_param": {"quality": "high"}}, "use": "omlx"},
        {"when": {"request_param": {"quality": "fast"}}, "use": "sense_voice"},
        {"when": {"duration_gt": 30}, "use": "omlx"},
        {"when": {"event_in": ["BGM", "Applause"]}, "use": "omlx"},
    ],
}


def test_default():
    r = IMRouter(ROUTING)
    d = r.decide(duration_sec=5.0, event="Speech", lang="zh", request_params={})
    assert d.upstream == "sense_voice"
    assert d.reason == "default"


def test_quality_high_overrides():
    r = IMRouter(ROUTING)
    d = r.decide(duration_sec=2.0, event="Speech", lang="zh", request_params={"quality": "high"})
    assert d.upstream == "omlx"
    assert "request_param" in d.reason


def test_long_duration_routes_to_omlx():
    r = IMRouter(ROUTING)
    d = r.decide(duration_sec=45.0, event="Speech", lang="zh", request_params={})
    assert d.upstream == "omlx"


def test_complex_event_routes_to_omlx():
    r = IMRouter(ROUTING)
    d = r.decide(duration_sec=5.0, event="BGM", lang="zh", request_params={})
    assert d.upstream == "omlx"


def test_first_match_wins():
    """quality=fast (rule 2) beats duration>30 (rule 3)"""
    r = IMRouter(ROUTING)
    d = r.decide(duration_sec=60.0, event="Speech", lang="zh", request_params={"quality": "fast"})
    assert d.upstream == "sense_voice"
