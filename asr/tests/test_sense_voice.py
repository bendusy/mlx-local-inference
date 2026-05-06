from asr_router.models.sense_voice import SenseVoiceTranscriber, TranscribeResult


def test_zh_lid(zh_wav):
    sv = SenseVoiceTranscriber.get()
    r = sv.transcribe(zh_wav)
    assert isinstance(r, TranscribeResult)
    assert r.lang == "zh"
    assert "开放时间" in r.text
    assert r.event == "Speech"
    assert r.duration_sec > 5.0
    assert r.decode_ms < 500


def test_yue_lid(yue_wav):
    r = SenseVoiceTranscriber.get().transcribe(yue_wav)
    assert r.lang == "yue"
    assert "唔到" in r.text  # 粤语特征字


def test_en_lid(en_wav):
    r = SenseVoiceTranscriber.get().transcribe(en_wav)
    assert r.lang == "en"


def test_singleton_reuses_session():
    a = SenseVoiceTranscriber.get()
    b = SenseVoiceTranscriber.get()
    assert a is b
