"""
安全验证工具模块单元测试

自 CX-O-VoiceWorkStation/tests/test_security_utils.py 迁移
（import 改写 modelstation.*；断言改用 _TRAINING_DATA_ROOT 模块常量，
 消除对进程 CWD 的依赖）。
"""
import pytest
from pathlib import Path
from modelstation.services import security_utils
from modelstation.services.security_utils import (
    validate_training_data_dir,
    _MS_ROOT,
    _TRAINING_DATA_ROOT,
)


def test_ms_root_anchored_to_modelstation():
    """锚点必须指向 CXO-ModelStation 包根（禁 CWD 相对解析）"""
    assert _MS_ROOT.name == "CXO-ModelStation"
    assert (_MS_ROOT / "modelstation").is_dir()


def test_training_root_follows_injected_settings(monkeypatch, tmp_path):
    """白名单根必须跟随注入配置（打包态=用户数据根）。

    回归背景：锚点曾固定取包内 <包根>/data/training，安装版下与注入的数据根分叉，
    生成的数据集被写进安装目录（训练侧读不到、卸载即丢失），注入目录路径被误拒。
    """
    from modelstation import config as config_module

    injected_root = tmp_path / "data" / "training"
    fake_settings = type(
        "FakeSettings",
        (),
        {"sovits_svc": type("S", (), {"training_data_dir": str(injected_root / "sovits_svc")})()},
    )()
    monkeypatch.setattr(config_module, "get_settings", lambda: fake_settings)

    assert security_utils._resolve_training_data_root() == injected_root.resolve()


def test_valid_relative_path():
    """测试有效的相对路径（相对于 data/training）"""
    # 传入的路径应该相对于 data/training 根目录
    result = validate_training_data_dir("data/training/sovits_svc")
    assert result.is_relative_to(_TRAINING_DATA_ROOT)


def test_nested_valid_path():
    """测试嵌套的有效路径"""
    result = validate_training_data_dir("data/training/f5tts/models/test")
    assert result.is_relative_to(_TRAINING_DATA_ROOT)


def test_empty_path():
    """测试空路径应抛出异常"""
    with pytest.raises(ValueError, match="must not be empty"):
        validate_training_data_dir("")


def test_absolute_path():
    """测试根目录之外的绝对路径应被拒绝（以是否落在
    _TRAINING_DATA_ROOT 之下为准，而非是否为绝对路径）"""
    # Unix 风格绝对路径（在 Windows 上会被当作相对路径，但仍会被拒绝）
    with pytest.raises(ValueError, match="must be located under"):
        validate_training_data_dir("/etc/passwd")


def test_windows_absolute_path():
    """测试根目录之外的 Windows 绝对路径应被拒绝"""
    with pytest.raises(ValueError, match="must be located under"):
        validate_training_data_dir("C:\\Windows\\System32")


def test_absolute_path_under_training_root_allowed():
    """位于 data/training 根之下的绝对路径必须放行。

    本服务自身会先 resolve 出绝对路径再传给 trainer 二次校验。
    """
    result = validate_training_data_dir(str(_TRAINING_DATA_ROOT / "sovits_svc"))
    assert result.is_relative_to(_TRAINING_DATA_ROOT)


def test_path_traversal_attack():
    """测试目录穿越攻击应被阻止"""
    with pytest.raises(ValueError, match="must be located under"):
        validate_training_data_dir("../../../etc")


def test_single_dot_path_traversal():
    """测试单层目录穿越应被阻止"""
    with pytest.raises(ValueError, match="must be located under"):
        validate_training_data_dir("../outside")


def test_complex_path_traversal():
    """测试复杂路径穿越应被阻止"""
    with pytest.raises(ValueError, match="must be located under"):
        validate_training_data_dir("data/training/../../outside")


def test_path_with_backslashes():
    """测试带反斜杠的路径（Windows风格）"""
    # Windows 风格的相对路径应该也能正常处理
    result = validate_training_data_dir("data\\training\\sovits_svc\\raw")
    assert result.is_relative_to(_TRAINING_DATA_ROOT)


def test_whitespace_only_path():
    """测试仅包含空白字符的路径"""
    # 空白字符路径应该在处理后被认为无效或被清理
    try:
        result = validate_training_data_dir("   ")
        # 如果通过了，检查结果是否合理
        assert result.is_relative_to(_TRAINING_DATA_ROOT)
    except ValueError:
        # 如果抛出异常，也是合理的行为
        pass


def test_current_directory_path():
    """测试当前目录路径"""
    # 当前目录 "." 解析为 _MS_ROOT，不在 data/training 下，应被拒绝
    with pytest.raises(ValueError, match="must be located under"):
        validate_training_data_dir(".")


def test_valid_path_with_trailing_slash():
    """测试带尾部斜杠的有效路径"""
    result = validate_training_data_dir("data/training/sovits_svc/")
    assert result.is_relative_to(_TRAINING_DATA_ROOT)


def test_multiple_valid_paths():
    """测试多个有效的训练数据目录路径"""
    valid_paths = [
        "data/training/sovits_svc",
        "data/training/f5tts",
        "data/training/f5tts/models",
        "data/training/sovits_svc/raw/speaker",
        "data/training/custom_voice",
    ]

    for path in valid_paths:
        result = validate_training_data_dir(path)
        assert result.is_relative_to(_TRAINING_DATA_ROOT)
