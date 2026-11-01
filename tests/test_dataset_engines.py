"""数据集三引擎与 manifest v2 测试（mock 客户端，不触发真实推理/HTTP）。

覆盖（spec「数据集生成三引擎」「manifest v2 与统一数据集」SubTask 2.4）：
- manifest v1→v2 迁移：函数级幂等、_load_manifest 读侧迁移；
- 三引擎分发：voxcpm（mock 子进程，回归）+ cosyvoice3_zero / qwen3_voicedesign
  （mock RuntimeTTSClient），manifest 条目 text/engine/mode/md5 断言与去重跳过；
- 引擎参数校验：非法 engine → 4xx、cosyvoice 缺 ref_audio_path / 白名单外路径 → 400、
  voicedesign 缺 voice_description → 400；
- 运行时不可达 → 任务逐条失败汇总，错误含 base_url 与指引；
- text 完整率统计：get_manifest_stats / list_datasets（v1/v2 数据集如实报告）；
- API 层：新端点 /api/datasets/batch-generate 流程、旧路由移除（404）、
  tts_runtime 配置段未就绪防御（503，Task 5 合跑后不可达）。

数据集目录经 patch security_utils._MS_ROOT/_TRAINING_DATA_ROOT 隔离到 tmp_path；
运行时引擎配置经注入 settings / monkeypatch 提供，不依赖真实 tts_runtime 段。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import modelstation.config as config_module
import modelstation.services.dataset_builder as db_module
import modelstation.services.security_utils as sec_module
from modelstation.config import load_config
from modelstation.main import create_app
from modelstation.services.dataset_builder import (
    ENGINE_COSYVOICE3_ZERO,
    ENGINE_QWEN3_VOICEDESIGN,
    ENGINE_VOXCPM,
    DatasetBuilderService,
    get_manifest_stats,
    migrate_manifest_to_v2,
)
from modelstation.services.runtime_tts_client import RuntimeTTSError

# ---------------------------------------------------------------------------
# 测试基建
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_data_root(monkeypatch, tmp_path):
    """隔离训练数据根目录到 tmp_path（同 test_dataset_builder 模式），
    并创建 data/input 白名单根（cosyvoice 参考音频落点）。"""
    root = tmp_path / "data" / "training"
    root.mkdir(parents=True)
    input_dir = tmp_path / "data" / "input"
    input_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sec_module, "_MS_ROOT", tmp_path)
    monkeypatch.setattr(sec_module, "_TRAINING_DATA_ROOT", root.resolve())
    return tmp_path


@pytest.fixture
def injected_settings(isolated_data_root):
    """注入含 tmp input_dir/training_data_dir 的 settings（参考音频白名单校验用）。
    注：训练数据目录校验经 sec_module patch 已指向 tmp，此处使白名单根一致。"""
    tmp = isolated_data_root
    return load_config(
        {
            "sovits_svc": {
                "input_dir": str(tmp / "data" / "input"),
                "training_data_dir": str(tmp / "data" / "training" / "sovits_svc"),
            }
        }
    )


class _FakeVoxCPMClient:
    """mock VoxCPM 客户端：直接写伪 WAV 字节（回归用）"""

    def __init__(self):
        self.calls: list[tuple] = []

    async def design(self, text, control, output_path, **kwargs):
        self.calls.append(("design", text, control))
        Path(output_path).write_bytes(b"RIFF-vox-" + text.encode("utf-8"))
        return Path(output_path)


class _FakeRuntimeClient:
    """mock RuntimeTTSClient：记录调用并写伪 WAV；可注入异常模拟运行时不可达"""

    def __init__(self, fail_on_text: set[str] | None = None):
        self.voicedesign_calls: list[tuple] = []
        self.cosyvoice_calls: list[tuple] = []
        self.fail_on_text = fail_on_text or set()

    def _write(self, text: str, output_path) -> Path:
        out = Path(output_path)
        if text in self.fail_on_text:
            raise RuntimeTTSError(
                "运行时不可达: connection refused", base_url="http://127.0.0.1:18094"
            )
        out.write_bytes(b"RIFF-rt-" + text.encode("utf-8"))
        return out

    async def synthesize_voicedesign(self, text, voice_description, output_path):
        self.voicedesign_calls.append((text, voice_description))
        return self._write(text, output_path)

    async def synthesize_cosyvoice_zero(self, text, ref_audio_path, ref_text, output_path):
        self.cosyvoice_calls.append((text, str(ref_audio_path), ref_text))
        return self._write(text, output_path)


@pytest.fixture
def engine_builder(isolated_data_root, injected_settings):
    """返回构造器：_make(vox_fake, rt_fake) → DatasetBuilderService（全 mock 注入）"""

    def _make(vox_fake=None, rt_fake=None, settings=None):
        vox_fake = vox_fake if vox_fake is not None else _FakeVoxCPMClient()
        rt_fake = rt_fake if rt_fake is not None else _FakeRuntimeClient()
        service = DatasetBuilderService(
            settings=settings if settings is not None else injected_settings,
            client_factory=lambda: vox_fake,
            runtime_client_factory=lambda engine: rt_fake,
        )
        return service, vox_fake, rt_fake

    return _make


@pytest_asyncio.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def patched_builder(engine_builder, monkeypatch):
    """API 测试：模块级单例替换为全 mock 注入实例（含 tmp 白名单 settings）"""
    service, vox_fake, rt_fake = engine_builder()
    monkeypatch.setattr(db_module, "_builder_instance", service)
    return service, vox_fake, rt_fake


@pytest.fixture
def patched_tts_runtime_settings(monkeypatch):
    """模拟 Task 1 已落地的 tts_runtime 配置段（API 层防御检查通过）。

    包装真实 settings：仅补充 tts_runtime 段，其余属性委托真实 get_settings()，
    保证 create_app 等真实消费方不受影响。
    """
    runtime_ns = SimpleNamespace(
        voicedesign_base_url="http://127.0.0.1:18091",
        voicedesign_model="test-voicedesign-model",
        cosyvoice_base_url="http://127.0.0.1:18094",
        cosyvoice_model="test-cosyvoice-model",
        timeout_seconds=5.0,
        sample_rate=24000,
    )
    real_get = config_module.get_settings

    class _SettingsWithRuntime:
        tts_runtime = runtime_ns

        def __getattr__(self, name):
            return getattr(real_get(), name)

    monkeypatch.setattr(config_module, "get_settings", lambda: _SettingsWithRuntime())
    return runtime_ns


async def _wait_task(service: DatasetBuilderService, task_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        task = service.get_task(task_id)
        assert task is not None
        if task["status"] in ("completed", "failed"):
            return task
        if time.monotonic() > deadline:
            raise TimeoutError(f"task {task_id} did not finish: {task}")
        await asyncio.sleep(0.01)


async def _wait_api_task(client: AsyncClient, task_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        resp = await client.get(f"/api/datasets/batch-generate/{task_id}")
        assert resp.status_code == 200
        task = resp.json()
        if task["status"] in ("completed", "failed"):
            return task
        if time.monotonic() > deadline:
            raise TimeoutError(f"api task {task_id} did not finish: {task}")
        await asyncio.sleep(0.01)


async def _wait_api_task_status(
    client: AsyncClient,
    task_id: str,
    statuses: tuple[str, ...],
    timeout: float = 10.0,
) -> dict:
    """轮询直到任务进入给定状态（试听闸门为 awaiting_review，非终态）。"""
    deadline = time.monotonic() + timeout
    while True:
        resp = await client.get(f"/api/datasets/batch-generate/{task_id}")
        assert resp.status_code == 200
        task = resp.json()
        if task["status"] in statuses:
            return task
        if time.monotonic() > deadline:
            raise TimeoutError(f"api task {task_id} did not reach {statuses}: {task}")
        await asyncio.sleep(0.01)


def _make_ref_wav(input_dir: Path, name: str = "ref.wav") -> str:
    """在白名单 input_dir 下构造参考音频文件，返回路径字符串。"""
    ref = input_dir / name
    ref.write_bytes(b"RIFF-fake-ref-audio")
    return str(ref)


# ---------------------------------------------------------------------------
# manifest v1→v2 迁移
# ---------------------------------------------------------------------------


def test_migrate_manifest_v1_to_v2_idempotent():
    """v1→v2 迁移：缺 text 补 None、缺 engine 补 voxcpm；已有字段保留；二次调用幂等"""
    manifest = {
        "version": 1,
        "entries": [
            {"fingerprint": "a", "file": "0001.wav", "md5": "x",
             "text": "已有文本", "engine": "voxcpm", "mode": "design", "control": ""},
            {"fingerprint": "b", "file": "0002.wav", "md5": "y", "mode": "design"},
        ],
    }
    migrated = migrate_manifest_to_v2(manifest)
    assert migrated["version"] == 2
    assert migrated["entries"][0]["text"] == "已有文本"  # 已有字段保留
    assert migrated["entries"][1]["text"] is None  # 缺失补 None
    assert migrated["entries"][1]["engine"] == ENGINE_VOXCPM  # 缺失补 voxcpm

    # 幂等：二次调用原样返回（version>=2 短路，不重复迁移）
    again = migrate_manifest_to_v2(migrated)
    assert again is migrated
    assert again["entries"][1]["text"] is None
    assert list(again["entries"][1].keys()).count("text") == 1  # 未重复插入


def test_migrate_manifest_edge_cases():
    """非法 version / 损坏结构 / 非 dict 输入 → 安全兜底"""
    bad = migrate_manifest_to_v2({"version": "abc", "entries": [{"file": "x.wav"}]})
    assert bad["version"] == 2 and bad["entries"][0]["text"] is None

    broken = migrate_manifest_to_v2({"version": 1, "entries": "oops"})
    assert broken["version"] == 2 and broken["entries"] == []

    non_dict = migrate_manifest_to_v2("not-a-dict")  # type: ignore[arg-type]
    assert non_dict == {"version": 2, "entries": []}

    assert migrate_manifest_to_v2({"version": 2, "entries": [{"f": 1}]})["entries"][0] == {"f": 1}


def test_load_manifest_migrates_v1_on_disk(engine_builder, isolated_data_root):
    """磁盘上的 v1 manifest 经 _load_manifest 读侧迁移（内存生效，不主动落盘）"""
    service, _vox, _rt = engine_builder()
    ds_dir = isolated_data_root / "data" / "training" / "sovits_svc" / "raw" / "legacy"
    ds_dir.mkdir(parents=True)
    (ds_dir / "0001.wav").write_bytes(b"RIFF-old")
    (ds_dir / "manifest.json").write_text(
        json.dumps(
            {"version": 1, "entries": [{"fingerprint": "z", "file": "0001.wav", "md5": "m"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = service._load_manifest(ds_dir)
    assert manifest["version"] == 2
    assert manifest["entries"][0]["text"] is None
    assert manifest["entries"][0]["engine"] == ENGINE_VOXCPM
    # 磁盘文件保持 v1 原样（读侧迁移不落盘，下次生成任务写入时持久化 v2）
    on_disk = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["version"] == 1


# ---------------------------------------------------------------------------
# 三引擎分发（服务层）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_cosyvoice3_zero_dispatch(engine_builder, isolated_data_root):
    """cosyvoice3_zero：调 RuntimeTTSClient（白名单归一化路径 + ref_text），manifest 条目 v2"""
    service, _vox, rt = engine_builder()
    ref_path = _make_ref_wav(isolated_data_root / "data" / "input")

    task = await _wait_task(
        service,
        await service.submit(
            "cvspk",
            [{"text": "第一句"}, {"text": "第二句"}],
            engine=ENGINE_COSYVOICE3_ZERO,
            engine_params={"ref_audio_path": ref_path, "ref_text": "参考转写"},
        ),
    )
    assert task["status"] == "completed"
    assert task["engine"] == ENGINE_COSYVOICE3_ZERO
    assert task["done"] == 2
    assert len(rt.cosyvoice_calls) == 2
    # 调用参数：归一化绝对路径 + ref_text
    assert Path(rt.cosyvoice_calls[0][1]).is_absolute()
    assert Path(rt.cosyvoice_calls[0][1]) == Path(ref_path).resolve()
    assert rt.cosyvoice_calls[0][2] == "参考转写"

    ds_dir = isolated_data_root / "data" / "training" / "sovits_svc" / "raw" / "cvspk"
    assert len(list(ds_dir.glob("*.wav"))) == 2
    manifest = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 2
    for entry in manifest["entries"]:
        assert entry["engine"] == ENGINE_COSYVOICE3_ZERO
        assert entry["text"] in {"第一句", "第二句"}
        assert entry["mode"] == ""  # 运行时引擎不使用 voxcpm mode
        actual = hashlib.md5((ds_dir / entry["file"]).read_bytes()).hexdigest()
        assert entry["md5"] == actual


@pytest.mark.asyncio
async def test_engine_qwen3_voicedesign_dispatch(engine_builder, isolated_data_root):
    """qwen3_voicedesign：调 RuntimeTTSClient（音色描述透传），manifest 条目 v2"""
    service, _vox, rt = engine_builder()

    task = await _wait_task(
        service,
        await service.submit(
            "vdspk",
            [{"text": "设计一句"}],
            engine=ENGINE_QWEN3_VOICEDESIGN,
            engine_params={"voice_description": "低沉磁性的男声"},
        ),
    )
    assert task["status"] == "completed"
    assert task["engine"] == ENGINE_QWEN3_VOICEDESIGN
    assert rt.voicedesign_calls == [("设计一句", "低沉磁性的男声")]

    ds_dir = isolated_data_root / "data" / "training" / "sovits_svc" / "raw" / "vdspk"
    manifest = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["entries"][0]
    assert entry["engine"] == ENGINE_QWEN3_VOICEDESIGN
    assert entry["text"] == "设计一句"
    assert entry["mode"] == ""


@pytest.mark.asyncio
async def test_voxcpm_engine_unchanged(engine_builder, isolated_data_root):
    """回归：voxcpm 引擎仍走 VoxCPMClient 子进程链路（不触运行时工厂）"""
    service, vox, rt = engine_builder()
    task = await _wait_task(
        service, await service.submit("vcpm", [{"text": "x"}], engine=ENGINE_VOXCPM)
    )
    assert task["status"] == "completed"
    assert task["engine"] == ENGINE_VOXCPM
    assert task["mode"] == "design"  # voxcpm 保留 mode
    assert len(vox.calls) == 1
    assert len(rt.voicedesign_calls) == 0 and len(rt.cosyvoice_calls) == 0
    ds_dir = isolated_data_root / "data" / "training" / "sovits_svc" / "raw" / "vcpm"
    manifest = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entries"][0]["engine"] == ENGINE_VOXCPM
    assert manifest["entries"][0]["mode"] == "design"


@pytest.mark.asyncio
async def test_runtime_engine_dedup_on_resubmit(engine_builder):
    """运行时引擎同文本+同参数重复提交：指纹命中跳过，不重复生成"""
    service, _vox, rt = engine_builder()
    params = {"voice_description": "同一音色描述"}
    texts = [{"text": "甲"}, {"text": "乙"}]

    first = await _wait_task(
        service, await service.submit("ddspk", texts, engine=ENGINE_QWEN3_VOICEDESIGN,
                                      engine_params=params)
    )
    assert first["done"] == 2
    second = await _wait_task(
        service, await service.submit("ddspk", texts, engine=ENGINE_QWEN3_VOICEDESIGN,
                                      engine_params=params)
    )
    assert second["status"] == "completed"
    assert second["skipped"] == 2 and second["done"] == 0
    assert len(rt.voicedesign_calls) == 2  # 未触发生成

    # 参数变更（音色描述不同）→ 同文本重新生成
    third = await _wait_task(
        service, await service.submit("ddspk", [{"text": "甲"}],
                                      engine=ENGINE_QWEN3_VOICEDESIGN,
                                      engine_params={"voice_description": "另一种音色"})
    )
    assert third["done"] == 1 and third["skipped"] == 0
    assert len(rt.voicedesign_calls) == 3


@pytest.mark.asyncio
async def test_runtime_unreachable_task_failed_with_base_url(engine_builder):
    """运行时不可达：逐条失败，任务 failed，错误含 base_url 与修复指引"""
    service, _vox, _rt = engine_builder(
        rt_fake=_FakeRuntimeClient(fail_on_text={"断流句"})
    )
    task = await _wait_task(
        service,
        await service.submit(
            "failspk",
            [{"text": "正常句"}, {"text": "断流句"}],
            engine=ENGINE_QWEN3_VOICEDESIGN,
            engine_params={"voice_description": "描述"},
        ),
    )
    assert task["status"] == "failed"
    assert task["done"] == 1 and task["failed"] == 1
    assert "http://127.0.0.1:18094" in task["error"]
    assert "检查 vLLM 运行时是否启动" in task["error"]
    assert task["failures"][0]["text"] == "断流句"


# ---------------------------------------------------------------------------
# 引擎参数校验
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_engine_rejected(engine_builder):
    """engine 不在支持列表 → ValueError（allowed 列表含三引擎）"""
    service, _vox, _rt = engine_builder()
    with pytest.raises(ValueError, match="Unsupported engine"):
        await service.submit("x", [{"text": "t"}], engine="cosyvoice")
    with pytest.raises(ValueError, match="Unsupported engine"):
        await service.submit("x", [{"text": "t"}], engine="cosyvoice3")


@pytest.mark.asyncio
async def test_cosyvoice_missing_ref_audio_path(engine_builder):
    """cosyvoice3_zero 缺 ref_audio_path → ValueError"""
    service, _vox, _rt = engine_builder()
    with pytest.raises(ValueError, match="ref_audio_path is required"):
        await service.submit("x", [{"text": "t"}], engine=ENGINE_COSYVOICE3_ZERO, engine_params={})
    with pytest.raises(ValueError, match="ref_audio_path is required"):
        await service.submit(
            "x", [{"text": "t"}], engine=ENGINE_COSYVOICE3_ZERO,
            engine_params={"ref_audio_path": "  "},
        )


@pytest.mark.asyncio
async def test_cosyvoice_ref_outside_whitelist(engine_builder, isolated_data_root):
    """参考音频在白名单根（training_data_dir ∪ input_dir）之外 → ValueError"""
    service, _vox, _rt = engine_builder()
    outside = isolated_data_root / "outside.wav"
    outside.write_bytes(b"RIFF-evil")
    with pytest.raises(ValueError, match="must be located under"):
        await service.submit(
            "x", [{"text": "t"}], engine=ENGINE_COSYVOICE3_ZERO,
            engine_params={"ref_audio_path": str(outside)},
        )


@pytest.mark.asyncio
async def test_voicedesign_missing_voice_description(engine_builder):
    """qwen3_voicedesign 缺 voice_description → ValueError"""
    service, _vox, _rt = engine_builder()
    with pytest.raises(ValueError, match="voice_description is required"):
        await service.submit("x", [{"text": "t"}], engine=ENGINE_QWEN3_VOICEDESIGN)
    with pytest.raises(ValueError, match="voice_description is required"):
        await service.submit(
            "x", [{"text": "t"}], engine=ENGINE_QWEN3_VOICEDESIGN,
            engine_params={"voice_description": ""},
        )


# ---------------------------------------------------------------------------
# text 完整率统计
# ---------------------------------------------------------------------------


def test_manifest_stats_and_list_datasets(isolated_data_root):
    """统计：v2 数据集报告版本与 text 完整率；v1 数据集如实报告（未迁移、无 text）"""
    raw = isolated_data_root / "data" / "training" / "sovits_svc" / "raw"

    v2_dir = raw / "withtext"
    v2_dir.mkdir(parents=True)
    (v2_dir / "a.wav").write_bytes(b"RIFF-a")
    (v2_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "entries": [
                    {"file": "0001.wav", "text": "第一句", "engine": "voxcpm"},
                    {"file": "0002.wav", "text": "第二句", "engine": "cosyvoice3_zero"},
                    {"file": "0003.wav", "text": None, "engine": "voxcpm"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    v1_dir = raw / "legacy"
    v1_dir.mkdir(parents=True)
    (v1_dir / "b.wav").write_bytes(b"RIFF-b")
    (v1_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "entries": [{"file": "0001.wav"}]}), encoding="utf-8"
    )
    no_manifest = raw / "empty"
    no_manifest.mkdir(parents=True)
    (no_manifest / "c.wav").write_bytes(b"RIFF-c")

    stats_v2 = get_manifest_stats(v2_dir)
    assert stats_v2 == {
        "manifest_version": 2,
        "entry_count": 3,
        "text_count": 2,
        "text_ratio": round(2 / 3, 4),
    }
    stats_v1 = get_manifest_stats(v1_dir)
    assert stats_v1["manifest_version"] == 1
    assert stats_v1["entry_count"] == 1 and stats_v1["text_count"] == 0
    assert stats_v1["text_ratio"] == 0.0
    stats_none = get_manifest_stats(no_manifest)
    assert stats_none["manifest_version"] is None and stats_none["text_ratio"] is None

    datasets = {d["name"]: d for d in db_module.list_datasets()}
    assert datasets["withtext"]["text_ratio"] == round(2 / 3, 4)
    assert datasets["withtext"]["file_count"] == 1  # 音频文件计数不受 manifest 影响
    assert datasets["legacy"]["manifest_version"] == 1
    assert datasets["empty"]["has_manifest"] is False
    assert datasets["empty"]["manifest_version"] is None


# ---------------------------------------------------------------------------
# API 层（新端点 / 旧路由移除 / tts_runtime 防御）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_cosyvoice3_zero_flow(
    client, patched_builder, patched_tts_runtime_settings, isolated_data_root
):
    """POST /api/datasets/batch-generate（engine=cosyvoice3_zero）→ task_id → completed"""
    ref_path = _make_ref_wav(isolated_data_root / "data" / "input")
    payload = {
        "speaker_name": "apicv",
        "texts": [{"text": "你好"}, {"text": "世界"}],
        "engine": "cosyvoice3_zero",
        "engine_params": {"ref_audio_path": ref_path, "ref_text": "参考文本"},
    }
    resp = await client.post("/api/datasets/batch-generate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success" and body["total"] == 2

    task = await _wait_api_task(client, body["task_id"])
    assert task["status"] == "completed"
    assert task["engine"] == "cosyvoice3_zero"
    assert task["mode"] == ""

    ds_dir = isolated_data_root / "data" / "training" / "sovits_svc" / "raw" / "apicv"
    manifest = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    assert all(e["engine"] == "cosyvoice3_zero" for e in manifest["entries"])
    assert all(e["text"] in {"你好", "世界"} for e in manifest["entries"])


@pytest.mark.asyncio
async def test_api_qwen3_voicedesign_flow(
    client, patched_builder, patched_tts_runtime_settings
):
    """POST /api/datasets/batch-generate（engine=qwen3_voicedesign）→ completed"""
    payload = {
        "speaker_name": "apivd",
        "texts": [{"text": "设计文本"}],
        "engine": "qwen3_voicedesign",
        "engine_params": {"voice_description": "清亮的少年音"},
    }
    resp = await client.post("/api/datasets/batch-generate", json=payload)
    assert resp.status_code == 200
    task = await _wait_api_task(client, resp.json()["task_id"])
    assert task["status"] == "completed"
    assert task["engine"] == "qwen3_voicedesign"
    _service, _vox, rt = patched_builder
    assert rt.voicedesign_calls == [("设计文本", "清亮的少年音")]


@pytest.mark.asyncio
async def test_api_voxcpm_engine_rejected_422(client, patched_builder):
    """voxcpm 已停用：engine=voxcpm → 422（批量语料统一走一套流程 /api/datasets/generate）"""
    resp = await client.post(
        "/api/datasets/batch-generate",
        json={"speaker_name": "apidef", "texts": [{"text": "默认引擎"}], "engine": "voxcpm"},
    )
    assert resp.status_code == 422
    _service, vox, rt = patched_builder
    assert len(vox.calls) == 0 and len(rt.voicedesign_calls) == 0


@pytest.mark.asyncio
async def test_api_pipeline_end_to_end(
    client, patched_builder, isolated_data_root
):
    """统一流程 POST /api/datasets/generate：参考音频 → cosyvoice 批量 → 编号 DS-001"""
    payload = {
        "voice_description": "温柔的青年女声",
        "ref_text": "这是参考文本",
        "texts": [{"text": "第一句"}, {"text": "第二句"}],
    }
    resp = await client.post("/api/datasets/generate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["dataset_id"] == "DS-001"
    assert body["name"] == "DS-001"
    assert body["total"] == 3

    task = await _wait_api_task(client, body["task_id"])
    assert task["status"] == "completed"
    assert task["engine"] == "pipeline"
    assert task["dataset_id"] == "DS-001"
    assert task["done"] == 3

    _service, _vox, rt = patched_builder
    # 阶段1：参考音频以 qwen3 生成（内容=ref_text）
    assert rt.voicedesign_calls == [("这是参考文本", "温柔的青年女声")]
    # 阶段2：cosyvoice 以生成的参考音频克隆
    assert len(rt.cosyvoice_calls) == 2
    ref_used = rt.cosyvoice_calls[0][1]
    assert ref_used.startswith(str(isolated_data_root / "data" / "training" / "sovits_svc" / "raw" / "DS-001"))

    ds_dir = isolated_data_root / "data" / "training" / "sovits_svc" / "raw" / "DS-001"
    assert len(list(ds_dir.glob("*.wav"))) == 3
    manifest = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entries"][0]["role"] == "reference"
    assert manifest["entries"][0]["text"] == "这是参考文本"

    # 注册表：DS-001 已登记且 ref_audio 回填
    reg = json.loads(
        (isolated_data_root / "data" / "training" / "dataset_registry.json")
        .read_text(encoding="utf-8")
    )
    assert reg["datasets"]["DS-001"]["ref_audio"] == manifest["entries"][0]["file"]

    # 列表附带编号
    resp_list = await client.get("/api/sovits-svc/datasets")
    assert resp_list.status_code == 200
    match = [d for d in resp_list.json()["datasets"] if d["name"] == "DS-001"]
    assert match and match[0]["dataset_id"] == "DS-001"


@pytest.mark.asyncio
async def test_api_preview_then_continue_flow(
    client, patched_builder, patched_tts_runtime_settings, isolated_data_root
):
    """试听闸门：/preview 只生成参考音频并停 awaiting_review；确认后 /continue 续跑语料"""
    resp = await client.post(
        "/api/datasets/preview",
        json={"voice_description": "温柔的青年女声", "ref_text": "这是参考文本"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == "DS-001" and body["total"] == 1

    task = await _wait_api_task_status(
        client, body["task_id"], ("awaiting_review", "failed")
    )
    assert task["status"] == "awaiting_review"
    assert task["phase"] == "reference" and task["done"] == 1
    assert task["ref_file"]
    # 试听阶段不产生任何语料克隆调用
    _service, _vox, rt = patched_builder
    assert rt.voicedesign_calls == [("这是参考文本", "温柔的青年女声")]
    assert len(rt.cosyvoice_calls) == 0

    ds_dir = isolated_data_root / "data" / "training" / "sovits_svc" / "raw" / "DS-001"
    assert (ds_dir / task["ref_file"]).is_file()

    # 用户试听确认后继续：以该参考音频克隆其余语料（逐句推进，done 递增）
    resp2 = await client.post(
        f"/api/datasets/{body['dataset_id']}/continue",
        json={"texts": [{"text": "第一句"}, {"text": "第二句"}]},
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["dataset_id"] == "DS-001" and body2["total"] == 3

    task2 = await _wait_api_task(client, body2["task_id"])
    assert task2["status"] == "completed"
    assert task2["phase"] == "corpus" and task2["done"] == 3
    assert len(rt.cosyvoice_calls) == 2
    assert rt.cosyvoice_calls[0][1].endswith(task["ref_file"])


@pytest.mark.asyncio
async def test_api_preview_regenerate_reference_keeps_single_entry(
    client, patched_builder, patched_tts_runtime_settings, isolated_data_root
):
    """试听不满意改描述重试：复用 dataset_id，只保留最新一条 reference 条目"""
    first = (
        await client.post(
            "/api/datasets/preview",
            json={"voice_description": "第一个描述", "ref_text": "参考文本一"},
        )
    ).json()
    task1 = await _wait_api_task_status(
        client, first["task_id"], ("awaiting_review", "failed")
    )
    assert task1["status"] == "awaiting_review"

    second = (
        await client.post(
            "/api/datasets/preview",
            json={
                "voice_description": "第二个描述",
                "ref_text": "参考文本二",
                "dataset_id": first["dataset_id"],
            },
        )
    ).json()
    assert second["dataset_id"] == first["dataset_id"]
    task2 = await _wait_api_task_status(
        client, second["task_id"], ("awaiting_review", "failed")
    )
    assert task2["status"] == "awaiting_review"
    assert task2["ref_file"] != task1["ref_file"]

    _service, _vox, rt = patched_builder
    assert len(rt.voicedesign_calls) == 2

    ds_dir = isolated_data_root / "data" / "training" / "sovits_svc" / "raw" / "DS-001"
    manifest = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    refs = [e for e in manifest["entries"] if e.get("role") == "reference"]
    assert len(refs) == 1
    assert refs[0]["file"] == task2["ref_file"] and refs[0]["text"] == "参考文本二"
    assert not (ds_dir / task1["ref_file"]).exists()


@pytest.mark.asyncio
async def test_api_pipeline_missing_voice_description_400(client, patched_builder):
    """统一流程缺 voice_description → 422（pydantic min_length）"""
    resp = await client.post(
        "/api/datasets/generate",
        json={"texts": [{"text": "hi"}]},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_api_invalid_engine_422(client, patched_builder):
    """非法 engine → 422（Literal 校验，可读错误）"""
    resp = await client.post(
        "/api/datasets/batch-generate",
        json={"speaker_name": "x", "texts": [{"text": "hi"}], "engine": "cosyvoice"},
    )
    assert resp.status_code == 422
    assert "qwen3_voicedesign" in json.dumps(resp.json(), ensure_ascii=False)


@pytest.mark.asyncio
async def test_api_engine_params_missing_400(
    client, patched_builder, patched_tts_runtime_settings
):
    """运行时引擎参数缺失 → 400：cosyvoice 缺 ref_audio_path、voicedesign 缺 voice_description"""
    resp = await client.post(
        "/api/datasets/batch-generate",
        json={"speaker_name": "x", "texts": [{"text": "hi"}], "engine": "cosyvoice3_zero"},
    )
    assert resp.status_code == 400
    assert "ref_audio_path" in resp.json()["detail"]

    resp = await client.post(
        "/api/datasets/batch-generate",
        json={
            "speaker_name": "x", "texts": [{"text": "hi"}],
            "engine": "qwen3_voicedesign", "engine_params": {},
        },
    )
    assert resp.status_code == 400
    assert "voice_description" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_api_cosyvoice_ref_outside_whitelist_400(
    client, patched_builder, patched_tts_runtime_settings, isolated_data_root
):
    """参考音频在白名单外 → 400"""
    outside = isolated_data_root / "evil.wav"
    outside.write_bytes(b"RIFF-evil")
    resp = await client.post(
        "/api/datasets/batch-generate",
        json={
            "speaker_name": "x", "texts": [{"text": "hi"}],
            "engine": "cosyvoice3_zero",
            "engine_params": {"ref_audio_path": str(outside)},
        },
    )
    assert resp.status_code == 400
    assert "must be located under" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_api_tts_runtime_not_ready_503(client, patched_builder, monkeypatch):
    """并行时序防御：tts_runtime 配置段缺失 → 503 含「tts_runtime 配置段未就绪」提示。

    并行分支 Task 1 的 tts_runtime 段已落地（真实配置含该段，防御分支在合跑后
    不可达）；此处经 monkeypatch 模拟段缺失场景验证防御逻辑本身。
    """
    monkeypatch.setattr(config_module, "get_settings", lambda: SimpleNamespace())

    resp = await client.post(
        "/api/datasets/batch-generate",
        json={
            "speaker_name": "x", "texts": [{"text": "hi"}],
            "engine": "cosyvoice3_zero",
            "engine_params": {"ref_audio_path": "data/input/a.wav"},
        },
    )
    assert resp.status_code == 503
    assert "tts_runtime 配置段未就绪" in resp.json()["detail"]

    # voxcpm 已停用 → 422（pydantic Literal 校验，与 tts_runtime 无关）
    resp = await client.post(
        "/api/datasets/batch-generate",
        json={"speaker_name": "y", "texts": [{"text": "hi"}], "engine": "voxcpm"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_api_old_voxcpm_route_removed(client):
    """旧路由 /api/voxcpm/batch-dataset 已移除（BREAKING 变更守卫）。

    断言路由表中不存在旧路径（HTTP 层断言受 SPA catch-all 干扰：
    POST 命中 GET-only fallback → 405，故以路由表为准 + 状态码非 2xx 辅助断言）。
    """
    from modelstation.main import create_app

    app = create_app()
    api_paths = [getattr(r, "path", "") for r in app.routes]
    assert "/api/voxcpm/batch-dataset" not in api_paths
    assert "/api/voxcpm/batch-dataset/{task_id}" not in api_paths
    assert "/api/datasets/batch-generate" in api_paths
    assert "/api/datasets/batch-generate/{task_id}" in api_paths

    resp = await client.post(
        "/api/voxcpm/batch-dataset",
        json={"speaker_name": "x", "texts": [{"text": "hi"}]},
    )
    # 不再是 API 成功路径（SPA fallback 下 POST 命中 GET-only 路由 → 405）
    assert resp.status_code in (404, 405)
    assert "task_id" not in resp.json()


@pytest.mark.asyncio
async def test_api_datasets_list_has_stats(client, patched_builder, isolated_data_root):
    """数据集列表响应附 manifest 版本与 text 完整率"""
    ds = isolated_data_root / "data" / "training" / "sovits_svc" / "raw" / "statdspk"
    ds.mkdir(parents=True)
    (ds / "a.wav").write_bytes(b"RIFF-a")
    (ds / "manifest.json").write_text(
        json.dumps(
            {"version": 2, "entries": [{"file": "a.wav", "text": "文本", "engine": "voxcpm"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    resp = await client.get("/api/sovits-svc/datasets")
    assert resp.status_code == 200
    match = [d for d in resp.json()["datasets"] if d["name"] == "statdspk"]
    assert len(match) == 1
    entry = match[0]
    assert entry["manifest_version"] == 2
    assert entry["entry_count"] == 1
    assert entry["text_count"] == 1
    assert entry["text_ratio"] == 1.0
