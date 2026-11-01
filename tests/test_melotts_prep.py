"""
MeloTTS 数据准备（melotts_dataset_prep）单元测试

覆盖（Task 3.4）：
- 95/5 切分比例与边界（总数<5 全进 train、val≥1）；
- text=null 条目排除并计数（manifest v2 / v1 迁移语义）；
- 音频文件缺失条目排除并计数（不崩溃）；
- filelist 行格式与 manifest_prep.json 元数据；
- CWD 免疫（monkeypatch.chdir 后路径解析不变）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import modelstation.config as config_module
import modelstation.services.security_utils as sec_module
from modelstation.config import ModelStationSettings
from modelstation.services.melotts_dataset_prep import prepare_filelists


@pytest.fixture
def isolated_roots(monkeypatch, tmp_path):
    """隔离训练数据根到 tmp_path（prep 读写不触碰真实 data/training）。

    返回 (training_root, prep_root)：training_root 为 validate_training_data_dir
    允许的根（data/training），prep_root 为 melotts training_data_dir（同时
    补丁 get_settings，使缺省 output_dir 落盘 tmp 而非真实 data/training/melotts）。
    """
    training_root = (tmp_path / "data" / "training").resolve()
    training_root.mkdir(parents=True)
    prep_root = tmp_path / "data" / "training" / "melotts"
    prep_root.mkdir(parents=True)
    monkeypatch.setattr(sec_module, "_MS_ROOT", tmp_path)
    monkeypatch.setattr(sec_module, "_TRAINING_DATA_ROOT", training_root)

    def _fake_settings():
        return ModelStationSettings.model_validate(
            {"melotts": {"training_data_dir": str(prep_root)}}
        )

    monkeypatch.setattr(config_module, "get_settings", _fake_settings)
    return training_root, prep_root


def _build_dataset(
    dataset_root: Path,
    speaker: str,
    entries_spec: list[dict],
    version: int = 2,
) -> Path:
    """构建 speaker 数据集目录：按 entries_spec 生成 wav 桩文件与 manifest.json。

    entries_spec 每项: {file: str, text: Optional[str], write_file: bool(默认 True)}
    """
    ds_dir = dataset_root / "sovits_svc" / "raw" / speaker
    ds_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, spec in enumerate(entries_spec, start=1):
        file_name = spec.get("file", f"{i:04d}.wav")
        if spec.get("write_file", True):
            (ds_dir / file_name).write_bytes(b"RIFF-fake")
        entry = {"fingerprint": f"fp-{i}", "file": file_name}
        if version >= 2:
            entry["text"] = spec.get("text")
            entry["engine"] = spec.get("engine", "voxcpm")
        entries.append(entry)
    manifest = {"version": version, "entries": entries}
    (ds_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return ds_dir


def _parse_filelist(path: Path) -> list[list[str]]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [line.split("|") for line in lines]


# ---------------------------------------------------------------------------
# 切分比例与行格式
# ---------------------------------------------------------------------------


def test_split_95_5_and_line_format(isolated_roots, tmp_path):
    """100 条可用条目 → 95/5 切分；行格式为四列绝对路径|speaker|语言|文本"""
    _, prep_root = isolated_roots
    ds_dir = _build_dataset(
        tmp_path / "data" / "training",
        "spk_a",
        [{"text": f"第{i}句测试文本"} for i in range(1, 101)],
    )
    stats = prepare_filelists(str(ds_dir), speaker_name="spk_a", language="ZH")

    assert stats["total"] == 100
    assert stats["used"] == 100
    assert stats["excluded_no_text"] == 0
    assert stats["excluded_missing_file"] == 0
    assert stats["train_count"] == 95
    assert stats["val_count"] == 5

    train_rows = _parse_filelist(Path(stats["train_file"]))
    val_rows = _parse_filelist(Path(stats["val_file"]))
    assert len(train_rows) == 95
    assert len(val_rows) == 5

    # 行格式（engines/MeloTTS 实码四列）：绝对路径|speaker|ZH|文本
    for row in train_rows + val_rows:
        assert len(row) == 4
        assert Path(row[0]).is_absolute()
        assert Path(row[0]).is_file()
        assert row[1] == "spk_a"
        assert row[2] == "ZH"
        assert row[3]

    # 无交集且并集完整
    train_paths = {r[0] for r in train_rows}
    val_paths = {r[0] for r in val_rows}
    assert not (train_paths & val_paths)
    assert len(train_paths | val_paths) == 100

    # metadata.list = train + val 全量
    meta_rows = _parse_filelist(Path(stats["metadata_file"]))
    assert len(meta_rows) == 100

    # manifest_prep.json 切分元数据落盘
    prep_meta = json.loads(
        (Path(stats["output_dir"]) / "manifest_prep.json").read_text(encoding="utf-8")
    )
    assert prep_meta["train_count"] == 95
    assert prep_meta["val_count"] == 5
    assert prep_meta["val_ratio"] == 0.05


def test_split_val_at_least_one_when_total_ge_5(isolated_roots, tmp_path):
    """总数=5 → val 至少 1 条（5% 下取整为 0 时取 1）"""
    _, _ = isolated_roots
    ds_dir = _build_dataset(
        tmp_path / "data" / "training",
        "spk_b",
        [{"text": f"text-{i}"} for i in range(1, 6)],
    )
    stats = prepare_filelists(str(ds_dir))
    assert stats["train_count"] == 4
    assert stats["val_count"] == 1


@pytest.mark.parametrize("total", [1, 2, 4])
def test_split_below_min_total_all_train(isolated_roots, tmp_path, total):
    """总数<5 → 全进 train 且 split_note 报告"""
    ds_dir = _build_dataset(
        tmp_path / "data" / "training",
        f"spk_small_{total}",
        [{"text": f"text-{i}"} for i in range(1, total + 1)],
    )
    stats = prepare_filelists(str(ds_dir))
    assert stats["train_count"] == total
    assert stats["val_count"] == 0
    assert "全部进入 train" in stats["split_note"]
    assert Path(stats["val_file"]).exists()


# ---------------------------------------------------------------------------
# 排除规则
# ---------------------------------------------------------------------------


def test_exclude_null_text_entries_with_count(isolated_roots, tmp_path):
    """text=null 条目排除并计数（含 v1 迁移语义的 None 与空串）"""
    ds_dir = _build_dataset(
        tmp_path / "data" / "training",
        "spk_null",
        [{"text": f"ok-{i}"} for i in range(1, 18)]  # 17 条可用（≥5）
        + [{"text": None}, {"text": ""}, {}],        # 3 条无 text
    )
    stats = prepare_filelists(str(ds_dir))
    assert stats["total"] == 20
    assert stats["excluded_no_text"] == 3
    assert stats["used"] == 17
    assert stats["train_count"] == 16
    assert stats["val_count"] == 1


def test_exclude_missing_audio_files(isolated_roots, tmp_path):
    """manifest 有条目但音频文件缺失 → 排除并计数，不崩溃"""
    ds_dir = _build_dataset(
        tmp_path / "data" / "training",
        "spk_missing",
        # 20 条在盘 + 3 条缺席（write_file=False）
        [{"text": f"ok-{i}"} for i in range(1, 21)]
        + [{"text": f"ghost-{i}", "file": f"ghost_{i}.wav", "write_file": False}
           for i in range(1, 4)],
    )
    stats = prepare_filelists(str(ds_dir))
    assert stats["total"] == 23
    assert stats["excluded_missing_file"] == 3
    assert stats["used"] == 20
    # 缺席文件不进入任何 filelist
    all_lines = Path(stats["metadata_file"]).read_text(encoding="utf-8")
    assert "ghost_" not in all_lines


def test_exclude_all_no_text_raises_with_counts(isolated_roots, tmp_path):
    """全部条目无 text（如纯 v1 数据集）→ 明确报错且消息含计数"""
    ds_dir = _build_dataset(
        tmp_path / "data" / "training",
        "spk_v1only",
        [{"text": None}] * 6,
    )
    with pytest.raises(ValueError) as excinfo:
        prepare_filelists(str(ds_dir))
    assert "excluded_no_text=6" in str(excinfo.value)


def test_text_pipe_and_newline_sanitized(isolated_roots, tmp_path):
    """文本中的 '|' 与换行被清洗，不破坏四列解析"""
    ds_dir = _build_dataset(
        tmp_path / "data" / "training",
        "spk_pipe",
        [{"text": "a|b\nc"} for _ in range(6)],
    )
    stats = prepare_filelists(str(ds_dir))
    rows = _parse_filelist(Path(stats["train_file"]))
    assert rows
    for row in rows:
        assert len(row) == 4


# ---------------------------------------------------------------------------
# manifest v1 兼容
# ---------------------------------------------------------------------------


def test_v1_manifest_migration_semantics(isolated_roots, tmp_path):
    """v1 manifest（无 version=2 字段）经迁移语义：text 缺失 → 全部排除计数；
    混合结构（v1 条目 + 手工 text 字段）正常进入 filelist"""
    root = tmp_path / "data" / "training"
    ds_dir = _build_dataset(
        root, "spk_v1",
        [{"text": None, "file": f"{i:04d}.wav"} for i in range(1, 7)],
        version=1,
    )
    # v1 manifest 逐条手工补一条 text（模拟已转录的 v1 数据集部分条目有 text）
    manifest = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["version"] = 1
    for entry in manifest["entries"][:2]:
        entry["text"] = "v1 补录文本"
    (ds_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    stats = prepare_filelists(str(ds_dir))
    assert stats["total"] == 6
    # 仅 2 条有 text（<5 全进 train）
    assert stats["used"] == 2
    assert stats["excluded_no_text"] == 4
    assert stats["train_count"] == 2
    assert stats["val_count"] == 0


def test_default_output_dir_under_melotts_training(isolated_roots, tmp_path):
    """缺省 output_dir 落在 settings.melotts.training_data_dir/<speaker> 下"""
    training_root, prep_root = isolated_roots
    ds_dir = _build_dataset(
        tmp_path / "data" / "training",
        "spk_out",
        [{"text": f"t-{i}"} for i in range(1, 7)],
    )
    stats = prepare_filelists(str(ds_dir))
    out_dir = Path(stats["output_dir"])
    assert out_dir.parent == prep_root
    assert (out_dir / "train.txt").is_file()
    assert (out_dir / "val.txt").is_file()
    assert (out_dir / "metadata.list").is_file()
    assert (out_dir / "manifest_prep.json").is_file()


def test_cwd_immunity(isolated_roots, tmp_path, monkeypatch):
    """CWD 免疫：切换 CWD 后路径解析与校验结果不变（rules-0 §三）"""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(scratch)
    ds_dir = _build_dataset(
        tmp_path / "data" / "training",
        "spk_cwd",
        [{"text": f"t-{i}"} for i in range(1, 7)],
    )
    stats = prepare_filelists(str(ds_dir))
    assert stats["used"] == 6
    assert Path(stats["train_file"]).is_absolute()
    assert Path(stats["train_file"]).is_file()


def test_invalid_dataset_dir_rejected(isolated_roots):
    """白名单外的 dataset_dir → ValueError（防任意路径）"""
    with pytest.raises(ValueError):
        prepare_filelists("C:/Windows/System32/evil")


def test_missing_manifest_raises(isolated_roots, tmp_path):
    """数据集目录无 manifest.json → ValueError"""
    training_root, _ = isolated_roots
    empty_dir = training_root / "sovits_svc" / "raw" / "spk_nomanifest"
    empty_dir.mkdir(parents=True)
    with pytest.raises(ValueError) as excinfo:
        prepare_filelists(str(empty_dir))
    assert "manifest.json not found" in str(excinfo.value)
