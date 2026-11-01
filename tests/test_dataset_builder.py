"""
VoxCPM 批量数据集与数据集管理 API 测试

自 CX-O-VoiceWorkStation/tests/test_dataset_builder.py 迁移
（change-id: split-audio-workstation-cxfc-modelstation）。
所有外部依赖（VoxCPM 模型子进程）均 mock，不触发真实推理；
数据集目录通过 patch security_utils._MS_ROOT/_TRAINING_DATA_ROOT 隔离到 tmp_path，
不污染仓库真实 data/training 目录。
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import modelstation.services.dataset_builder as db_module
import modelstation.services.security_utils as sec_module
from modelstation.main import create_app
from modelstation.services.dataset_builder import DatasetBuilderService


# ---------------------------------------------------------------------------
# 测试基建
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_data_root(monkeypatch, tmp_path):
    """隔离训练数据根目录到 tmp_path：patch security_utils._MS_ROOT 与
    _TRAINING_DATA_ROOT（导入期常量）指向 tmp 下的根。

    validate_training_data_dir 的相对路径分支锚定 _MS_ROOT 解析
    （不依赖进程 CWD），因此这里 patch _MS_ROOT 使相对路径落入 tmp；
    保留 chdir 兼容其他可能按 CWD 解析的相对路径。
    """
    root = tmp_path / "data" / "training"
    root.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sec_module, "_MS_ROOT", tmp_path)
    monkeypatch.setattr(sec_module, "_TRAINING_DATA_ROOT", root.resolve())
    return root


class _FakeVoxCPMClient:
    """mock VoxCPM 客户端：直接写伪 WAV 字节，可按文本集合注入失败"""

    def __init__(self, fail_on: set[str] | None = None):
        self.calls: list[tuple] = []
        self.fail_on = fail_on or set()

    def _write(self, text: str, output_path: str) -> Path:
        if text in self.fail_on:
            raise RuntimeError(f"mock voxcpm failure: {text}")
        out = Path(output_path)
        out.write_bytes(b"RIFF-fake-" + text.encode("utf-8"))
        return out

    async def design(self, text, control, output_path, **kwargs):
        self.calls.append(("design", text, control, kwargs))
        return self._write(text, output_path)

    async def controllable_clone(self, text, control, reference_audio, output_path, **kwargs):
        self.calls.append(("controllable_clone", text, control, kwargs))
        return self._write(text, output_path)

    async def ultimate_clone(self, text, prompt_audio, prompt_text, output_path, **kwargs):
        self.calls.append(("ultimate_clone", text, kwargs))
        return self._write(text, output_path)


class _FakeRuntimeClient:
    """mock RuntimeTTSClient：记录调用并写伪 WAV（API 流程测试用）"""

    def __init__(self):
        self.voicedesign_calls: list[tuple] = []
        self.cosyvoice_calls: list[tuple] = []

    async def synthesize_voicedesign(self, text, voice_description, output_path):
        self.voicedesign_calls.append((text, voice_description))
        out = Path(output_path)
        out.write_bytes(b"RIFF-rt-" + text.encode("utf-8"))
        return out

    async def synthesize_cosyvoice_zero(self, text, ref_audio_path, ref_text, output_path):
        self.cosyvoice_calls.append((text, str(ref_audio_path), ref_text))
        out = Path(output_path)
        out.write_bytes(b"RIFF-rt-" + text.encode("utf-8"))
        return out


@pytest.fixture
def builder_factory(isolated_data_root):
    """返回 (service, fake_client) 工厂，各测试独立实例互不污染"""

    def _make(fail_on: set[str] | None = None):
        fake = _FakeVoxCPMClient(fail_on=fail_on)
        service = DatasetBuilderService(client_factory=lambda: fake)
        return service, fake

    return _make


@pytest_asyncio.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def patched_builder_singleton(monkeypatch, isolated_data_root):
    """API 测试：将模块级单例替换为挂 fake client 的实例（voxcpm + 运行时双 mock）"""
    fake = _FakeVoxCPMClient()
    rt_fake = _FakeRuntimeClient()
    builder = DatasetBuilderService(
        client_factory=lambda: fake,
        runtime_client_factory=lambda engine: rt_fake,
    )
    monkeypatch.setattr(db_module, "_builder_instance", builder)
    return builder, fake, rt_fake


async def _wait_task(service: DatasetBuilderService, task_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        task = service.get_task(task_id)
        assert task is not None
        if task["status"] in ("completed", "failed"):
            return task
        if time.monotonic() > deadline:
            raise TimeoutError(f"task {task_id} did not finish within {timeout}s: {task}")
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
            raise TimeoutError(f"task {task_id} did not finish within {timeout}s: {task}")
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# 服务层：批量生成端到端 / 去重 / 部分失败 / 参数校验
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_dataset_end_to_end(builder_factory, isolated_data_root):
    """提交 → 完成 → speaker 目录含全部 WAV 且 manifest.json 正确（MD5 与文件一致）"""
    service, fake = builder_factory()
    texts = [{"text": "第一句"}, {"text": "第二句", "control": "温柔"}, {"text": "第三句"}]

    task_id = await service.submit("alice", texts, mode="design", control="默认控制")
    task = await _wait_task(service, task_id)

    assert task["status"] == "completed"
    assert task["total"] == 3
    assert task["done"] == 3
    assert task["skipped"] == 0
    assert task["failed"] == 0
    assert task["speaker_name"] == "alice"
    assert len(fake.calls) == 3
    # 条目级 control 覆盖任务级
    assert fake.calls[0][2] == "默认控制"
    assert fake.calls[1][2] == "温柔"

    dataset_dir = isolated_data_root / "sovits_svc" / "raw" / "alice"
    wavs = sorted(dataset_dir.glob("*.wav"))
    assert len(wavs) == 3

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 2  # manifest v2（条目含 text/engine，见 dataset_builder._MANIFEST_VERSION）
    entries = manifest["entries"]
    assert len(entries) == 3
    for entry in entries:
        # MD5 与文件实际内容一致
        actual = hashlib.md5((dataset_dir / entry["file"]).read_bytes()).hexdigest()
        assert entry["md5"] == actual
        assert entry["fingerprint"]
        assert entry["text"] in {"第一句", "第二句", "第三句"}
        assert entry["mode"] == "design"


@pytest.mark.asyncio
async def test_batch_dataset_dedup_on_resubmit(builder_factory, isolated_data_root):
    """重复提交相同文本+参数：全部跳过不重复生成；变更一条则仅生成该条"""
    service, fake = builder_factory()
    texts = [{"text": "甲"}, {"text": "乙"}]

    first = await _wait_task(service, await service.submit("bob", texts))
    assert first["done"] == 2
    assert len(fake.calls) == 2

    # 原样重复提交 → 全部命中指纹跳过
    second = await _wait_task(service, await service.submit("bob", texts))
    assert second["status"] == "completed"
    assert second["done"] == 0
    assert second["skipped"] == 2
    assert len(fake.calls) == 2  # 未触发生成

    dataset_dir = isolated_data_root / "sovits_svc" / "raw" / "bob"
    assert len(list(dataset_dir.glob("*.wav"))) == 2

    # 变更一条 → 仅新生成该条，其余跳过
    third = await _wait_task(service, await service.submit("bob", [{"text": "甲"}, {"text": "丙"}]))
    assert third["done"] == 1
    assert third["skipped"] == 1
    assert len(fake.calls) == 3
    assert len(list(dataset_dir.glob("*.wav"))) == 3

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 3


@pytest.mark.asyncio
async def test_batch_dataset_partial_failure(builder_factory, isolated_data_root):
    """单条生成失败不中断整批：failed 计数 + failures 明细，任务状态 failed"""
    service, fake = builder_factory(fail_on={"会失败"})

    task = await _wait_task(
        service, await service.submit("carol", [{"text": "正常一"}, {"text": "会失败"}, {"text": "正常二"}])
    )

    assert task["status"] == "failed"
    assert task["done"] == 2
    assert task["failed"] == 1
    assert task["skipped"] == 0
    assert len(task["failures"]) == 1
    assert task["failures"][0]["index"] == 1
    assert task["failures"][0]["text"] == "会失败"
    assert "1/3" in task["error"]

    dataset_dir = isolated_data_root / "sovits_svc" / "raw" / "carol"
    assert len(list(dataset_dir.glob("*.wav"))) == 2
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 2


@pytest.mark.asyncio
async def test_batch_submit_validation(builder_factory):
    """空文本清单 / 空文本 / 模式专属参数缺失 → ValueError"""
    service, _ = builder_factory()

    with pytest.raises(ValueError, match="texts must not be empty"):
        await service.submit("dave", [])

    with pytest.raises(ValueError, match=r"texts\[0\]\.text"):
        await service.submit("dave", [{"text": "  "}])

    with pytest.raises(ValueError, match="reference_audio_path is required"):
        await service.submit("dave", [{"text": "hi"}], mode="controllable_clone")

    with pytest.raises(ValueError, match="prompt_audio_path is required"):
        await service.submit("dave", [{"text": "hi"}], mode="ultimate_clone")

    with pytest.raises(ValueError, match="prompt_text is required"):
        await service.submit(
            "dave", [{"text": "hi"}], mode="ultimate_clone", prompt_audio_path="data/input/a.wav"
        )


@pytest.mark.asyncio
async def test_speaker_name_sanitized(builder_factory, isolated_data_root):
    """speaker_name 经清洗（路径分隔符等非法字符 → 下划线），目录仍落在 raw/ 下"""
    service, _ = builder_factory()
    task = await _wait_task(service, await service.submit("../evil中文", [{"text": "x"}]))
    assert task["status"] == "completed"
    # 清洗后名称不含路径分隔符，且目录位于数据集根之内
    name = task["speaker_name"]
    assert "/" not in name and "\\" not in name and ".." not in name
    assert (isolated_data_root / "sovits_svc" / "raw" / name).is_dir()


@pytest.mark.asyncio
async def test_validate_training_data_dir_invoked(monkeypatch, builder_factory):
    """批量提交必须经过 security_utils.validate_training_data_dir 集中校验"""
    calls: list[str] = []
    real = sec_module.validate_training_data_dir

    def spy(path: str):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(db_module, "validate_training_data_dir", spy)
    service, _ = builder_factory()
    task = await _wait_task(service, await service.submit("erin", [{"text": "y"}]))
    assert task["status"] == "completed"
    # 路径基准随白名单根解析（打包态=用户数据根），故按规范化后的后缀匹配而非字面量
    assert any(
        Path(c).as_posix().endswith("data/training/sovits_svc/raw/erin") for c in calls
    )


@pytest.mark.asyncio
async def test_validate_training_data_dir_rejects(monkeypatch, builder_factory):
    """集中校验拒绝时（非法目录），提交直接抛 ValueError"""
    monkeypatch.setattr(
        db_module,
        "validate_training_data_dir",
        lambda path: (_ for _ in ()).throw(ValueError("mock reject")),
    )
    service, _ = builder_factory()
    with pytest.raises(ValueError, match="mock reject"):
        await service.submit("frank", [{"text": "z"}])


# ---------------------------------------------------------------------------
# API 层：批量任务 / 数据集列表 / 导入 / 删除
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_batch_dataset_flow(client, patched_builder_singleton, isolated_data_root):
    """POST /api/datasets/batch-generate → task_id；GET 查询至 completed；重复提交去重"""
    builder, _fake, rt = patched_builder_singleton
    payload = {
        "speaker_name": "apispk",
        "texts": [{"text": "你好"}, {"text": "世界"}],
        "engine": "qwen3_voicedesign",
        "engine_params": {"voice_description": "清亮女声"},
    }

    resp = await client.post("/api/datasets/batch-generate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["total"] == 2
    task_id = body["task_id"]

    task = await _wait_api_task(client, task_id)
    assert task["status"] == "completed"
    assert task["done"] == 2
    assert task["failed"] == 0

    dataset_dir = isolated_data_root / "sovits_svc" / "raw" / "apispk"
    assert len(list(dataset_dir.glob("*.wav"))) == 2
    assert (dataset_dir / "manifest.json").is_file()

    # 重复提交相同内容 → 全部跳过
    resp2 = await client.post("/api/datasets/batch-generate", json=payload)
    assert resp2.status_code == 200
    task2 = await _wait_api_task(client, resp2.json()["task_id"])
    assert task2["skipped"] == 2
    assert task2["done"] == 0
    assert len(rt.voicedesign_calls) == 2


@pytest.mark.asyncio
async def test_api_batch_task_not_found(client, patched_builder_singleton):
    """查询不存在的 task_id → 404"""
    resp = await client.get("/api/datasets/batch-generate/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_api_batch_rejects_invalid_payload(client, patched_builder_singleton):
    """空文本清单 422；controllable_clone 缺 reference_audio_path → 400"""
    resp = await client.post(
        "/api/datasets/batch-generate", json={"speaker_name": "x", "texts": []}
    )
    assert resp.status_code == 422

    resp = await client.post(
        "/api/datasets/batch-generate",
        json={"speaker_name": "x", "texts": [{"text": "hi"}], "mode": "controllable_clone"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_api_datasets_list(client, isolated_data_root):
    """GET /api/sovits-svc/datasets 返回名称、文件数、总大小、创建时间"""
    ds = isolated_data_root / "sovits_svc" / "raw" / "listspk"
    ds.mkdir(parents=True)
    (ds / "a.wav").write_bytes(b"RIFF-a")
    (ds / "b.flac").write_bytes(b"FLAC-b")
    (ds / "manifest.json").write_text("{}", encoding="utf-8")

    resp = await client.get("/api/sovits-svc/datasets")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    match = [d for d in body["datasets"] if d["name"] == "listspk"]
    assert len(match) == 1
    entry = match[0]
    assert entry["file_count"] == 2
    assert entry["total_size_bytes"] == len(b"RIFF-a") + len(b"FLAC-b")
    assert entry["created_at"]
    assert entry["has_manifest"] is True


@pytest.mark.asyncio
async def test_api_import_multifile(client, isolated_data_root):
    """POST /api/sovits-svc/datasets/import：multipart 多文件上传成功"""
    resp = await client.post(
        "/api/sovits-svc/datasets/import",
        data={"speaker_name": "imp1"},
        files=[
            ("files", ("a.wav", b"RIFF-a", "audio/wav")),
            ("files", ("b.flac", b"FLAC-b", "audio/flac")),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["name"] == "imp1"
    assert body["imported"] == 2
    assert sorted(body["files"]) == ["a.wav", "b.flac"]

    ds = isolated_data_root / "sovits_svc" / "raw" / "imp1"
    assert (ds / "a.wav").read_bytes() == b"RIFF-a"
    assert (ds / "b.flac").read_bytes() == b"FLAC-b"


@pytest.mark.asyncio
async def test_api_import_zip(client, isolated_data_root):
    """zip 包导入：音频成员解出，非音频成员跳过"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("voice/a.wav", b"RIFF-a")
        z.writestr("b.wav", b"RIFF-b")
        z.writestr("notes.txt", b"nope")

    resp = await client.post(
        "/api/sovits-svc/datasets/import",
        data={"speaker_name": "impzip"},
        files=[("files", ("pack.zip", buf.getvalue(), "application/zip"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["imported"] == 2
    assert sorted(body["files"]) == ["a.wav", "b.wav"]
    assert body["skipped"] == ["notes.txt"]

    ds = isolated_data_root / "sovits_svc" / "raw" / "impzip"
    assert (ds / "a.wav").read_bytes() == b"RIFF-a"


@pytest.mark.asyncio
async def test_api_import_zip_strips_traversal(client, isolated_data_root):
    """zip 内带 ../ 前缀的成员被剥离目录（防 zip-slip），仅落入数据集目录"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../evil.wav", b"evil")

    resp = await client.post(
        "/api/sovits-svc/datasets/import",
        data={"speaker_name": "impevil"},
        files=[("files", ("evil.zip", buf.getvalue(), "application/zip"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["files"] == ["evil.wav"]

    ds = isolated_data_root / "sovits_svc" / "raw" / "impevil"
    assert (ds / "evil.wav").read_bytes() == b"evil"
    # 不得逸出到数据集目录之外
    assert not (isolated_data_root / "evil.wav").exists()
    assert not (isolated_data_root / "sovits_svc" / "evil.wav").exists()


@pytest.mark.asyncio
async def test_api_import_rejects_bad_extension(client, isolated_data_root):
    """非音频扩展名 → 400"""
    resp = await client.post(
        "/api/sovits-svc/datasets/import",
        data={"speaker_name": "imp2"},
        files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))],
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_api_import_rejects_traversal_speaker_name(client, isolated_data_root):
    """speaker_name 含路径穿越/绝对路径 → 400"""
    for bad in ("../evil", "..", "a/b", "C:\\\\evil", "a b"):
        resp = await client.post(
            "/api/sovits-svc/datasets/import",
            data={"speaker_name": bad},
            files=[("files", ("a.wav", b"RIFF", "audio/wav"))],
        )
        assert resp.status_code == 400, f"expected 400 for speaker_name={bad!r}"


@pytest.mark.asyncio
async def test_api_delete_dataset(client, isolated_data_root):
    """DELETE /api/sovits-svc/datasets/{name}：存在 → 200 删除；再删 → 404"""
    ds = isolated_data_root / "sovits_svc" / "raw" / "del1"
    ds.mkdir(parents=True)
    (ds / "a.wav").write_bytes(b"RIFF")

    resp = await client.delete("/api/sovits-svc/datasets/del1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert not ds.exists()

    resp = await client.delete("/api/sovits-svc/datasets/del1")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_api_delete_rejects_traversal(client, isolated_data_root):
    """DELETE 路径穿越尝试（%2e%2e 编码绕过 httpx 归一化）→ 400，目录不受影响"""
    ds = isolated_data_root / "sovits_svc" / "raw" / "keep"
    ds.mkdir(parents=True)
    (ds / "a.wav").write_bytes(b"RIFF")

    resp = await client.delete("/api/sovits-svc/datasets/%2e%2e")
    assert resp.status_code == 400
    assert ds.is_dir()


# ---------------------------------------------------------------------------
# 引擎来源（单一 voxcpm，SVC 训练数据默认引擎）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_voxcpm_engine_explicit_default(builder_factory, isolated_data_root):
    """engine=voxcpm 显式传值：行为与不传等价，调用 VoxCPMClient.design"""
    service, vox = builder_factory()
    await _wait_task(service, await service.submit("vcpmspk", [{"text": "x"}], engine="voxcpm"))

    assert len(vox.calls) == 1
    assert vox.calls[0][0] == "design"


@pytest.mark.asyncio
async def test_engine_dedup_within_same_engine(builder_factory, isolated_data_root):
    """同引擎重复提交相同文本+参数：指纹命中跳过，不重复生成"""
    service, vox = builder_factory()
    texts = [{"text": "甲"}]

    first = await _wait_task(service, await service.submit("dpspk", texts, engine="voxcpm"))
    assert first["done"] == 1
    assert len(vox.calls) == 1

    second = await _wait_task(service, await service.submit("dpspk", texts, engine="voxcpm"))
    assert second["done"] == 0
    assert second["skipped"] == 1
    assert len(vox.calls) == 1  # 未触发生成


@pytest.mark.asyncio
async def test_engine_validation_unsupported(builder_factory):
    """engine 不在支持列表 → ValueError"""
    service, _vox = builder_factory()
    with pytest.raises(ValueError, match="Unsupported engine"):
        await service.submit("vspk", [{"text": "x"}], engine="cosyvoice")


@pytest.mark.asyncio
async def test_dataset_output_compatible_with_svc_preprocess(builder_factory, isolated_data_root):
    """voxcpm 产出目录结构与 manifest 兼容 /api/sovits-svc/preprocess 的输入期望：
    data/training/sovits_svc/raw/<speaker>/*.wav + manifest.json（version+entries）"""
    service, _vox = builder_factory()
    await _wait_task(service, await service.submit("svcspk", [{"text": "v"}], engine="voxcpm"))

    # preprocess 期望的 training_data_dir = data/training/sovits_svc
    training_root = isolated_data_root  # = data/training
    raw_dir = training_root / "sovits_svc" / "raw" / "svcspk"
    assert raw_dir.is_dir()
    wavs = list(raw_dir.glob("*.wav"))
    assert len(wavs) == 1
    # 全部为 WAV 扩展名（SVC preprocess 只接受音频文件）
    assert all(w.suffix.lower() == ".wav" for w in wavs)

    manifest_path = raw_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 2
    assert isinstance(manifest["entries"], list)
    assert len(manifest["entries"]) == 1
    # 每条 entry 的 md5 与落盘文件一致（MD5 防重可校验）
    for entry in manifest["entries"]:
        assert entry["file"].endswith(".wav")
        assert (raw_dir / entry["file"]).is_file()
        actual = hashlib.md5((raw_dir / entry["file"]).read_bytes()).hexdigest()
        assert entry["md5"] == actual


@pytest.mark.asyncio
async def test_api_batch_dataset_rejects_invalid_engine(client, monkeypatch, isolated_data_root):
    """engine 不合法 → 422（pattern 校验）"""
    fake = _FakeVoxCPMClient()
    builder = DatasetBuilderService(client_factory=lambda: fake)
    monkeypatch.setattr(db_module, "_builder_instance", builder)

    resp = await client.post(
        "/api/datasets/batch-generate",
        json={"speaker_name": "x", "texts": [{"text": "hi"}], "engine": "cosyvoice"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_api_batch_dataset_engine_defaults_to_cosyvoice(client, monkeypatch, isolated_data_root):
    """不传 engine → 默认 cosyvoice3_zero（voxcpm 已停用）；缺 ref_audio_path → 400"""
    fake = _FakeVoxCPMClient()
    builder = DatasetBuilderService(
        client_factory=lambda: fake,
        runtime_client_factory=lambda engine: _FakeRuntimeClient(),
    )
    monkeypatch.setattr(db_module, "_builder_instance", builder)

    resp = await client.post(
        "/api/datasets/batch-generate",
        json={"speaker_name": "defspk", "texts": [{"text": "hi"}]},
    )
    assert resp.status_code == 400
    assert "ref_audio_path" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 语料成片组装（assemble_sentence_clips：长音频按句子边界 5~15s 切分）
# ---------------------------------------------------------------------------


def _write_sentence_wav(path: Path, seconds: float, sr: int = 44100) -> None:
    import numpy as np
    import soundfile as sf

    n = int(seconds * sr)
    sf.write(str(path), np.linspace(-0.1, 0.1, n).astype("float32"), sr, subtype="PCM_16")


def _make_sentence_dataset(tmp_path: Path, durations: list[float], *, corrupt_idx: int = -1) -> Path:
    """构造逐句合成数据集：reference 条目 + N 条语料（真实 wav + text）。"""
    dataset_dir = tmp_path / "DS-900"
    dataset_dir.mkdir(parents=True)
    entries = [{
        "fingerprint": "ref", "file": "0001_ref.wav", "md5": "",
        "text": "参考音频。", "engine": "qwen3_voicedesign", "role": "reference",
    }]
    _write_sentence_wav(dataset_dir / "0001_ref.wav", 0.2)
    for i, seconds in enumerate(durations, start=2):
        name = f"{i:04d}_s.wav"
        if i == corrupt_idx:
            (dataset_dir / name).write_bytes(b"not-a-wav")
        else:
            _write_sentence_wav(dataset_dir / name, seconds)
        entries.append({
            "fingerprint": f"fp{i}", "file": name, "md5": "",
            "text": f"第{i}句语料内容。", "engine": "cosyvoice3_zero",
        })
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"version": 2, "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return dataset_dir


def test_assemble_sentence_clips_grouping(tmp_path):
    """短句贪心拼片：达到 min_seconds 收片，文本顺序拼接，原句文件删除"""
    import soundfile as sf

    dataset_dir = _make_sentence_dataset(tmp_path, [3.0, 3.0, 3.0])
    clips = db_module.assemble_sentence_clips(dataset_dir)
    assert clips == 2  # [3s+3s → 6.3s 达标收片] + [3s 尾片]
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    body = [e for e in manifest["entries"] if e.get("role") != "reference"]
    assert len(manifest["entries"]) == 3  # reference + 2 成片
    assert [e["file"] for e in body] == ["clip_0001.wav", "clip_0002.wav"]
    assert body[0]["text"] == "第2句语料内容。第3句语料内容。"
    assert body[0]["role"] == "corpus_clip"
    assert body[1]["text"] == "第4句语料内容。"
    # 音频：成片 44.1kHz 单声道，时长含句间 gap
    info = sf.info(str(dataset_dir / "clip_0001.wav"))
    assert info.samplerate == 44100 and info.channels == 1
    assert abs(info.frames / info.samplerate - 6.3) < 0.05
    # 原句文件已删除
    assert not (dataset_dir / "0002_s.wav").exists()


def test_assemble_sentence_clips_max_cap_and_oversize(tmp_path):
    """超上限提前收片；单句超长独立成片"""
    dataset_dir = _make_sentence_dataset(tmp_path, [10.0, 10.0, 16.0])
    clips = db_module.assemble_sentence_clips(dataset_dir)
    assert clips == 3  # [10s] + [10s]（再拼超 15s）+ [16s 超长独立]
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    body = [e for e in manifest["entries"] if e.get("role") != "reference"]
    assert [e["file"] for e in body] == ["clip_0001.wav", "clip_0002.wav", "clip_0003.wav"]


def test_assemble_sentence_clips_decode_failure_kept(tmp_path):
    """单条解码失败：该句独立保留原条目（作分片屏障），其余正常拼片"""
    dataset_dir = _make_sentence_dataset(tmp_path, [3.0, 3.0, 3.0], corrupt_idx=3)
    clips = db_module.assemble_sentence_clips(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    body = [e for e in manifest["entries"] if e.get("role") != "reference"]
    # 句3 解码失败 → 屏障：句2 独立成片、句3 保留原文件、句4 独立成片
    assert clips == 3
    files = {e["file"] for e in body}
    assert files == {"clip_0001.wav", "0003_s.wav", "clip_0003.wav"}
    texts = {e["text"] for e in body}
    assert texts == {"第2句语料内容。", "第3句语料内容。", "第4句语料内容。"}
    assert (dataset_dir / "0003_s.wav").exists()


def test_assemble_sentence_clips_noop(tmp_path):
    """无 manifest / 条目不足 / 全部解码失败 → 不改动返回 0"""
    assert db_module.assemble_sentence_clips(tmp_path / "nope") == 0
    single = _make_sentence_dataset(tmp_path / "s1", [3.0])
    assert db_module.assemble_sentence_clips(single) == 0
