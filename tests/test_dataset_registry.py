"""dataset_registry 单元测试：编号分配 / 归一化 / 解析 / 元数据回填 / 列表附带"""
from __future__ import annotations

import pytest

import modelstation.services.security_utils as sec_module
from modelstation.services.dataset_registry import (
    DatasetRegistryError,
    all_entries,
    attach_dataset_ids,
    get_entry,
    normalize_dataset_id,
    register_dataset,
    registry_path,
    update_entry,
)


@pytest.fixture
def isolated_data_root(monkeypatch, tmp_path):
    """隔离训练数据根目录到 tmp_path（同 dataset_builder 测试模式）"""
    root = tmp_path / "data" / "training"
    root.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sec_module, "_MS_ROOT", tmp_path)
    monkeypatch.setattr(sec_module, "_TRAINING_DATA_ROOT", root.resolve())
    return root


def test_normalize_dataset_id():
    assert normalize_dataset_id("1") == "DS-001"
    assert normalize_dataset_id("ds-1") == "DS-001"
    assert normalize_dataset_id("DS-001") == "DS-001"
    assert normalize_dataset_id(" 12 ") == "DS-012"
    assert normalize_dataset_id("1000") == "DS-1000"
    with pytest.raises(DatasetRegistryError):
        normalize_dataset_id("abc")
    with pytest.raises(DatasetRegistryError):
        normalize_dataset_id("")
    with pytest.raises(DatasetRegistryError):
        normalize_dataset_id("0")


def test_register_sequential_and_custom_name(isolated_data_root):
    first = register_dataset(engine="pipeline", ref_text="参考")
    assert first == {"dataset_id": "DS-001", "name": "DS-001"}
    second = register_dataset("custom_name")
    assert second == {"dataset_id": "DS-002", "name": "custom_name"}

    data = all_entries()
    assert data["DS-001"]["engine"] == "pipeline"
    assert data["DS-001"]["ref_text"] == "参考"
    assert data["DS-002"]["name"] == "custom_name"
    assert registry_path().is_file()


def test_register_invalid_name(isolated_data_root):
    with pytest.raises(DatasetRegistryError):
        register_dataset("../evil")
    with pytest.raises(DatasetRegistryError):
        register_dataset("带空格 name")


def test_get_entry_and_update(isolated_data_root):
    register_dataset()
    entry = get_entry("ds-1")
    assert entry["dataset_id"] == "DS-001"
    with pytest.raises(DatasetRegistryError):
        get_entry("DS-999")

    update_entry("DS-001", ref_audio="0001_abcd1234.wav", task_id="t1")
    assert get_entry("DS-001")["ref_audio"] == "0001_abcd1234.wav"
    # 不存在的编号：静默忽略
    update_entry("DS-999", ref_audio="x.wav")


def test_attach_dataset_ids(isolated_data_root):
    register_dataset()  # DS-001（目录名同名）
    datasets = [
        {"name": "DS-001", "file_count": 1},
        {"name": "imported", "file_count": 2},
    ]
    attached = attach_dataset_ids(datasets)
    assert attached[0]["dataset_id"] == "DS-001"
    assert attached[1]["dataset_id"] is None
