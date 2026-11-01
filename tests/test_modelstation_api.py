"""
ModelStation API 冒烟测试

覆盖（spec「CXO-ModelStation 后端服务」验收相关冒烟）：
- /health 健康检查；
- audio-files：未知 category 404、路径穿越防护 403、非法扩展名 404；
- infer：非法路径（白名单外）→ 400、白名单内不存在文件 → 404；
- datasets CRUD（隔离到 tmp 目录，不污染真实 data/training）。
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import modelstation.services.security_utils as sec_module
from modelstation.main import create_app


@pytest.fixture
def isolated_data_root(monkeypatch, tmp_path):
    """隔离训练数据根目录到 tmp_path（datasets CRUD 不触碰真实 data/training）"""
    root = tmp_path / "data" / "training"
    root.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sec_module, "_MS_ROOT", tmp_path)
    monkeypatch.setattr(sec_module, "_TRAINING_DATA_ROOT", root.resolve())
    return root


@pytest_asyncio.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health(client):
    """GET /health → healthy + 服务名"""
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["service"] == "CXO-ModelStation"
    assert body["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# audio-files：category 白名单 / 路径穿越 / 扩展名白名单
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_files_unknown_category_404(client):
    """未知 category → 404"""
    resp = await client.get("/api/audio-files/nonexistent/file.wav")
    assert resp.status_code == 404
    assert "Unknown audio category" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_audio_files_category_whitelist_only_audition_datasets(client):
    """白名单仅含 audition/datasets：原 svc-results/songs/voxcpm category 全部 404"""
    for old in ("svc-results", "songs", "voxcpm"):
        resp = await client.get(f"/api/audio-files/{old}/file.wav")
        assert resp.status_code == 404, f"category={old} 应为 404"


@pytest.mark.asyncio
async def test_audio_files_path_traversal_blocked(client, isolated_data_root):
    """路径穿越（%2e%2e%2f 绕过 httpx 归一化）→ 403，不泄露目录外文件"""
    resp = await client.get("/api/audio-files/audition/%2e%2e%2f%2e%2e%2fsecret.wav")
    assert resp.status_code == 403

    resp = await client.get("/api/audio-files/datasets/%2e%2e%2fsecret.wav")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_audio_files_unsupported_extension_404(client, isolated_data_root):
    """非音频扩展名 → 404（不暴露任意文件类型）"""
    resp = await client.get("/api/audio-files/audition/config.json")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_audio_files_missing_file_404(client, isolated_data_root):
    """白名单 category + 合法扩展名 + 文件不存在 → 404"""
    resp = await client.get("/api/audio-files/audition/no_such_file.wav")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# infer 参数校验（非法路径 4xx，不触发子进程）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_infer_rejects_path_outside_whitelist(client, isolated_data_root):
    """白名单外的 audio_path → 400（ValueError），不触发推理子进程"""
    resp = await client.post(
        "/api/sovits-svc/infer",
        json={"audio_path": "C:\\Windows\\System32\\evil.wav", "model_path": "whatever"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_infer_rejects_traversal_audio_path(client, isolated_data_root):
    """穿越尝试的 audio_path → 400"""
    resp = await client.post(
        "/api/sovits-svc/infer",
        json={"audio_path": "../../etc/passwd.wav", "model_path": "whatever"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_infer_whitelisted_but_missing_audio_404(client):
    """白名单根（配置的 training_data_dir）内但文件不存在 → 404（FileNotFoundError）"""
    from pathlib import Path

    from modelstation.config import get_settings

    settings = get_settings()
    audio = Path(settings.sovits_svc.training_data_dir) / "raw" / "ghost" / "missing.wav"
    resp = await client.post(
        "/api/sovits-svc/infer",
        json={"audio_path": str(audio), "model_path": "whatever"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# datasets CRUD（tmp 隔离目录）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_datasets_crud_flow(client, isolated_data_root):
    """导入 → 列表 → 删除 → 再删 404 的完整 CRUD 冒烟"""
    # 1. 导入
    resp = await client.post(
        "/api/sovits-svc/datasets/import",
        data={"speaker_name": "smoke1"},
        files=[("files", ("a.wav", b"RIFF-a", "audio/wav"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["imported"] == 1
    assert body["files"] == ["a.wav"]

    ds_dir = isolated_data_root / "sovits_svc" / "raw" / "smoke1"
    assert (ds_dir / "a.wav").read_bytes() == b"RIFF-a"

    # 2. 列表
    resp = await client.get("/api/sovits-svc/datasets")
    assert resp.status_code == 200
    names = [d["name"] for d in resp.json()["datasets"]]
    assert "smoke1" in names

    # 3. 删除
    resp = await client.delete("/api/sovits-svc/datasets/smoke1")
    assert resp.status_code == 200
    assert not ds_dir.exists()

    # 4. 再删 → 404
    resp = await client.delete("/api/sovits-svc/datasets/smoke1")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_datasets_import_invalid_name_400(client, isolated_data_root):
    """非法 speaker_name（穿越/非法字符）→ 400；空串被 Form 校验拦截 → 422"""
    for bad in ("../evil", "a/b", "a b"):
        resp = await client.post(
            "/api/sovits-svc/datasets/import",
            data={"speaker_name": bad},
            files=[("files", ("a.wav", b"RIFF", "audio/wav"))],
        )
        assert resp.status_code == 400, f"speaker_name={bad!r} 应 400"

    resp = await client.post(
        "/api/sovits-svc/datasets/import",
        data={"speaker_name": ""},
        files=[("files", ("a.wav", b"RIFF", "audio/wav"))],
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# workflow / 训练状态端点冒烟（只读，不触发子进程）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_status_shape(client):
    """GET /api/workflow/status → 三步初始 pending"""
    resp = await client.get("/api/workflow/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_step"] == 0
    assert [s["id"] for s in body["steps"]] == ["train_prep", "training", "inference"]
    assert all(s["status"] == "pending" for s in body["steps"])


@pytest.mark.asyncio
async def test_training_status_idle(client):
    """GET /api/sovits-svc/status → idle 初始状态（models 降级为列表）"""
    resp = await client.get("/api/sovits-svc/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "idle"
    assert isinstance(body["models"], list)


@pytest.mark.asyncio
async def test_workflow_unknown_step_output_404(client):
    """查询不存在的 step output → 404"""
    resp = await client.get("/api/workflow/step/nonexistent/output")
    assert resp.status_code == 404
