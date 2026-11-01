"""
配置 auto_fill 单元测试

覆盖：
- 缺失字段自动补默认值（auto_fill）；
- 越界值（越出取值范围/不可解析）回退到默认值；
- 环境变量 CXO_MODELSTATION_CONFIG 加载与解析失败降级。
"""
from __future__ import annotations

import json

import pytest

import modelstation.config as cfg_module
from modelstation.config import (
    DEFAULT_AUDITION_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_SO_VITS_SVC_DIR,
    DEFAULT_TRAINING_DATA_DIR,
    DEFAULT_VOXCPM_WORKING_DIR,
    _BASE_DIR,
    _PROJECT_ROOT,
    ModelStationSettings,
    get_settings,
    load_config,
    reset_settings,
)


def test_base_dir_anchor():
    """锚点解析：config.py 的 parents[1] = CX-ModelWorkStation，parents[2] = 项目根"""
    assert _BASE_DIR.name == "CX-ModelWorkStation"
    assert (_BASE_DIR / "modelstation" / "config.py").exists()
    assert _PROJECT_ROOT == _BASE_DIR.parent


def test_empty_config_fills_all_defaults():
    """空配置 → 全部字段补默认"""
    c = load_config({})
    assert c.server.host == "127.0.0.1"
    assert c.server.port == 8300
    assert c.server.log_level == "info"
    assert c.server.cors_origins == [
        "http://localhost:3300",
        "http://127.0.0.1:3300",
        "http://localhost:3100",
    ]
    assert c.sovits_svc.training_data_dir == DEFAULT_TRAINING_DATA_DIR
    assert c.sovits_svc.models_dir == DEFAULT_MODELS_DIR
    assert c.sovits_svc.audition_dir == DEFAULT_AUDITION_DIR
    assert c.sovits_svc.input_dir == DEFAULT_INPUT_DIR
    assert c.sovits_svc.so_vits_svc_dir == DEFAULT_SO_VITS_SVC_DIR
    assert c.sovits_svc.python_path == "python"
    assert c.voxcpm.model_path == "openbmb/VoxCPM2"
    assert c.voxcpm.working_dir == DEFAULT_VOXCPM_WORKING_DIR


def test_data_dirs_anchored_to_modelstation():
    """数据目录默认值全部锚定 CXO-ModelStation/data 下（禁 CWD 相对解析）"""
    c = load_config({})
    assert c.sovits_svc.training_data_dir == str(_BASE_DIR / "data" / "training" / "sovits_svc")
    assert c.sovits_svc.models_dir == str(_BASE_DIR / "data" / "models" / "sovits_svc")
    assert c.sovits_svc.audition_dir == str(_BASE_DIR / "data" / "audition")
    assert c.sovits_svc.input_dir == str(_BASE_DIR / "data" / "input")
    # 引擎目录锚定 CXO-ModelStation/engines（2026-09-05 自包含化迁移，禁 ../ 项目根依赖）
    assert c.sovits_svc.so_vits_svc_dir == str(_BASE_DIR / "engines" / "so-vits-svc-4.1-Stable")
    assert c.voxcpm.working_dir == str(_BASE_DIR / "engines" / "VoxCPM-main")


def test_nested_partial_config_fills_missing():
    """嵌套段部分提供 → 缺失子字段补默认，提供的值保留"""
    c = load_config({"server": {"port": 8400}})
    assert c.server.port == 8400
    assert c.server.host == "127.0.0.1"  # 缺失补默认
    assert c.sovits_svc.models_dir == DEFAULT_MODELS_DIR  # 未提供段整体补默认

    c2 = load_config({"sovits_svc": {"python_path": "python3.14"}})
    assert c2.sovits_svc.python_path == "python3.14"
    assert c2.sovits_svc.training_data_dir == DEFAULT_TRAINING_DATA_DIR


def test_port_out_of_range_falls_back():
    """port 越界（<1 或 >65535 或不可解析）回退 8300"""
    for bad in (0, -1, 99999, 65536, "abc", None, [8300]):
        c = load_config({"server": {"port": bad}})
        assert c.server.port == 8300, f"port={bad!r} 应回退 8300"
    # 边界内合法值保留
    assert load_config({"server": {"port": 1}}).server.port == 1
    assert load_config({"server": {"port": 65535}}).server.port == 65535
    assert load_config({"server": {"port": "8400"}}).server.port == 8400


def test_log_level_invalid_falls_back():
    """log_level 不在合法枚举内回退 info；大小写归一"""
    assert load_config({"server": {"log_level": "verbose"}}).server.log_level == "info"
    assert load_config({"server": {"log_level": ""}}).server.log_level == "info"
    assert load_config({"server": {"log_level": None}}).server.log_level == "info"
    assert load_config({"server": {"log_level": "INFO"}}).server.log_level == "info"
    assert load_config({"server": {"log_level": "Debug"}}).server.log_level == "debug"


def test_host_and_empty_strings_fall_back():
    """host 空值补默认；目录字段空串/None 补默认"""
    c = load_config({
        "server": {"host": "  "},
        "sovits_svc": {"training_data_dir": "", "python_path": None},
        "voxcpm": {"model_path": ""},
    })
    assert c.server.host == "127.0.0.1"
    assert c.sovits_svc.training_data_dir == DEFAULT_TRAINING_DATA_DIR
    assert c.sovits_svc.python_path == "python"
    assert c.voxcpm.model_path == "openbmb/VoxCPM2"


def test_cors_origins_string_coercion_and_default():
    """cors_origins 支持逗号分隔字符串；缺失补默认"""
    c = load_config({"server": {"cors_origins": "http://a:1, http://b:2"}})
    assert c.server.cors_origins == ["http://a:1", "http://b:2"]
    c2 = load_config({"server": {"cors_origins": None}})
    assert c2.server.cors_origins == [
        "http://localhost:3300",
        "http://127.0.0.1:3300",
        "http://localhost:3100",
    ]


def test_melotts_defaults_and_autofill():
    """melotts 段：空配置补全默认值；空串/None 目录回退默认；language 归一大写"""
    c = load_config({})
    assert c.melotts.engine_dir == str(_BASE_DIR / "engines" / "MeloTTS")
    assert c.melotts.training_data_dir == str(_BASE_DIR / "data" / "training" / "melotts")
    assert c.melotts.models_dir == str(_BASE_DIR / "data" / "models" / "melotts")
    assert c.melotts.python_path == "python"
    assert c.melotts.language == "ZH"
    assert c.melotts.base_checkpoint == ""  # 空=官方默认下载（合法空值，不回退）

    # 空串/None 回退默认
    c2 = load_config({
        "melotts": {
            "engine_dir": "",
            "training_data_dir": None,
            "models_dir": "  ",
            "python_path": "",
            "language": None,
        }
    })
    assert c2.melotts.engine_dir == str(_BASE_DIR / "engines" / "MeloTTS")
    assert c2.melotts.training_data_dir == str(_BASE_DIR / "data" / "training" / "melotts")
    assert c2.melotts.models_dir == str(_BASE_DIR / "data" / "models" / "melotts")
    assert c2.melotts.python_path == "python"
    assert c2.melotts.language == "ZH"

    # 提供的合法值保留；language 小写归一
    c3 = load_config({
        "melotts": {
            "engine_dir": r"D:\engines\MeloTTS",
            "python_path": "python3.11",
            "language": "zh",
            "base_checkpoint": r"D:\ckpts\base.pth",
        }
    })
    assert c3.melotts.engine_dir == r"D:\engines\MeloTTS"
    assert c3.melotts.python_path == "python3.11"
    assert c3.melotts.language == "ZH"
    assert c3.melotts.base_checkpoint == r"D:\ckpts\base.pth"


def test_tts_runtime_defaults_and_autofill():
    """tts_runtime 段：空配置补默认；base_url 尾斜杠归一；越界 timeout/sample_rate 回退"""
    c = load_config({})
    assert c.tts_runtime.voicedesign_base_url == "http://127.0.0.1:8091"
    assert c.tts_runtime.voicedesign_model == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert c.tts_runtime.cosyvoice_base_url == "http://127.0.0.1:8094"
    assert c.tts_runtime.cosyvoice_model == "Fun-CosyVoice3-0.5B-2512"
    assert c.tts_runtime.timeout_seconds == 120
    assert c.tts_runtime.sample_rate == 24000

    # 空串回退默认；URL 尾斜杠归一
    c2 = load_config({
        "tts_runtime": {
            "voicedesign_base_url": "http://10.0.0.5:8091/",
            "cosyvoice_base_url": "",
            "voicedesign_model": "",
        }
    })
    assert c2.tts_runtime.voicedesign_base_url == "http://10.0.0.5:8091"
    assert c2.tts_runtime.cosyvoice_base_url == "http://127.0.0.1:8094"
    assert c2.tts_runtime.voicedesign_model == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    # timeout/sample_rate 越界或不可解析回退默认；边界合法值保留
    for bad_timeout in (0, -5, 3601, "abc", None):
        c3 = load_config({"tts_runtime": {"timeout_seconds": bad_timeout}})
        assert c3.tts_runtime.timeout_seconds == 120, f"timeout={bad_timeout!r} 应回退 120"
    assert load_config({"tts_runtime": {"timeout_seconds": 30}}).tts_runtime.timeout_seconds == 30.0
    assert load_config({"tts_runtime": {"timeout_seconds": 3600}}).tts_runtime.timeout_seconds == 3600.0
    for bad_rate in (7999, 192001, "abc", None):
        c4 = load_config({"tts_runtime": {"sample_rate": bad_rate}})
        assert c4.tts_runtime.sample_rate == 24000, f"sample_rate={bad_rate!r} 应回退 24000"
    assert load_config({"tts_runtime": {"sample_rate": 16000}}).tts_runtime.sample_rate == 16000


def test_env_config_load_and_invalid_json_fallback(monkeypatch):
    """环境变量 JSON 生效；非法 JSON 按空配置降级（不抛异常）"""
    payload = {"server": {"port": 8501}}
    monkeypatch.setenv("CXO_MODELSTATION_CONFIG", json.dumps(payload))
    c = load_config(None)
    assert c.server.port == 8501

    monkeypatch.setenv("CXO_MODELSTATION_CONFIG", "{not-json")
    c2 = load_config(None)
    assert c2.server.port == 8300


def test_get_settings_singleton_and_reset():
    """get_settings 进程级单例；reset_settings 后重建"""
    reset_settings()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    reset_settings()
    s3 = get_settings()
    assert s3 is not s1
    assert isinstance(s3, ModelStationSettings)


def test_default_settings_match_pydantic_defaults():
    """无 env 干净环境下 get_settings 与空配置 load_config 等价"""
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.delenv("CXO_MODELSTATION_CONFIG", raising=False)
        reset_settings()
        s = get_settings()
        d = load_config({})
        assert s.server.port == d.server.port == 8300
        assert s.sovits_svc.audition_dir == d.sovits_svc.audition_dir
    finally:
        reset_settings()
        monkeypatch.undo()
