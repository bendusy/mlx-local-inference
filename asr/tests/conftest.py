from pathlib import Path
import pytest

SV_DIR = Path.home() / "models/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


@pytest.fixture(scope="session")
def test_wav_dir() -> Path:
    return SV_DIR / "test_wavs"


@pytest.fixture(scope="session")
def zh_wav(test_wav_dir) -> Path:
    return test_wav_dir / "zh.wav"


@pytest.fixture(scope="session")
def en_wav(test_wav_dir) -> Path:
    return test_wav_dir / "en.wav"


@pytest.fixture(scope="session")
def yue_wav(test_wav_dir) -> Path:
    return test_wav_dir / "yue.wav"


def pytest_collection_modifyitems(config, items):
    """Skip all tests if SenseVoice models aren't installed."""
    if not SV_DIR.exists():
        skip = pytest.mark.skip(reason="Run scripts/install_models.sh first")
        for item in items:
            item.add_marker(skip)
