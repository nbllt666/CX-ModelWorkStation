"""
MeloTTS 试听推理测试（POST /api/melotts/infer）

覆盖：
- 成功链路：子进程桩在临时输出目录产出 wav → 搬到 audition 目录并返回 audio_url；
- 指定检查点 / 缺省取最新检查点；
- 参数校验：空文本 400、非 models_dir 路径 400、checkpoint 路径穿越 400；
- 资源缺失：模型目录无 G_*.pth 404。

子进程全部 mock（asyncio.create_subprocess_exec），不触碰真实引擎与 GPU。
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from modelstation.config import ModelStationSettings
from modelstation.main import create_app
from modelstation.services import melotts_infer as mi


class _FakeProc:
    pid = 4321
    returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", b""


def _fake_asyncio(exec_impl):
    """用带桩 create_subprocess_exec 的 asyncio 门面替换模块引用。"""
    return SimpleNamespace(
        create_subprocess_exec=exec_impl,
        wait_for=asyncio.wait_for,
        TimeoutError=asyncio.TimeoutError,
        subprocess=asyncio.subprocess,
    )


def _make_exec(produce: bool = True):
    async def fake_exec(*args, **kwargs):
        out_dir = Path(args[args.index("--output_dir") + 1])
        if produce:
            (out_dir / "DS-001").mkdir(parents=True, exist_ok=True)
            (out_dir / "DS-001" / "output.wav").write_bytes(b"RIFF-fake")
        return _FakeProc()

    return fake_exec


@pytest.fixture
def infer_env(monkeypatch, tmp_path):
    """隔离引擎/模型/试听目录，返回 paths dict。"""
    base = tmp_path / "env"
    engine_dir = base / "engines" / "MeloTTS"
    (engine_dir / "melo").mkdir(parents=True)
    (engine_dir / "melo" / "infer.py").write_text("# fake infer", encoding="utf-8")

    models_dir = base / "data" / "models" / "melotts"
    model_dir = models_dir / "auto_DS-001_melotts"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"data": {"spk2id": {"DS-001": 0}}}), encoding="utf-8"
    )
    (model_dir / "G_252.pth").write_bytes(b"g252")
    (model_dir / "G_1800.pth").write_bytes(b"g1800")
    # 显式 mtime 保证「最新检查点」判定可重复
    os.utime(model_dir / "G_252.pth", (1_000, 1_000))
    os.utime(model_dir / "G_1800.pth", (2_000, 2_000))

    audition_dir = base / "data" / "audition"
    audition_dir.mkdir(parents=True)

    def _fake_settings():
        return ModelStationSettings.model_validate(
            {
                "melotts": {
                    "engine_dir": str(engine_dir),
                    "models_dir": str(models_dir),
                },
                "sovits_svc": {"audition_dir": str(audition_dir)},
            }
        )

    monkeypatch.setattr(mi, "get_settings", _fake_settings)
    return {
        "base": base,
        "engine_dir": engine_dir,
        "models_dir": models_dir,
        "model_dir": model_dir,
        "audition_dir": audition_dir,
    }


@pytest_asyncio.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_infer_success_returns_audition_url(monkeypatch, infer_env, client):
    monkeypatch.setattr(mi, "asyncio", _fake_asyncio(_make_exec()))

    resp = await client.post(
        "/api/melotts/infer",
        json={
            "model_path": str(infer_env["model_dir"]),
            "checkpoint": "G_1800.pth",
            "text": "今天天气真好，我们一起出去玩吧。",
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "success"
    assert body["checkpoint"] == "G_1800.pth"
    assert body["audio_url"] == f"/api/audio-files/audition/{body['output_filename']}"
    assert body["output_filename"].startswith("melotts_G_1800_")
    produced = infer_env["audition_dir"] / body["output_filename"]
    assert produced.is_file()
    # 临时目录已清理，audition 目录仅剩最终产物
    assert [p.name for p in infer_env["audition_dir"].iterdir()] == [body["output_filename"]]


@pytest.mark.asyncio
async def test_infer_defaults_to_latest_checkpoint(monkeypatch, infer_env, client):
    monkeypatch.setattr(mi, "asyncio", _fake_asyncio(_make_exec()))

    resp = await client.post(
        "/api/melotts/infer",
        json={"model_path": str(infer_env["model_dir"]), "text": "你好呀"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["checkpoint"] == "G_1800.pth"


@pytest.mark.asyncio
async def test_infer_empty_text_400(infer_env, client):
    resp = await client.post(
        "/api/melotts/infer",
        json={"model_path": str(infer_env["model_dir"]), "text": "   "},
    )
    assert resp.status_code == 400
    assert "text 不能为空" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_infer_model_path_outside_models_dir_400(infer_env, client):
    resp = await client.post(
        "/api/melotts/infer",
        json={"model_path": str(infer_env["base"] / "other"), "text": "你好"},
    )
    assert resp.status_code == 400
    assert "model_path must be located under" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_infer_checkpoint_traversal_400(monkeypatch, infer_env, client):
    monkeypatch.setattr(mi, "asyncio", _fake_asyncio(_make_exec()))
    resp = await client.post(
        "/api/melotts/infer",
        json={
            "model_path": str(infer_env["model_dir"]),
            "checkpoint": "../G_1.pth",
            "text": "你好",
        },
    )
    assert resp.status_code == 400
    assert "只接受 G_*.pth 文件名" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_infer_no_checkpoint_404(infer_env, client):
    empty_model = infer_env["models_dir"] / "empty_model"
    empty_model.mkdir()
    (empty_model / "config.json").write_text("{}", encoding="utf-8")

    resp = await client.post(
        "/api/melotts/infer",
        json={"model_path": str(empty_model), "text": "你好"},
    )
    assert resp.status_code == 404
    assert "无 G_*.pth" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_infer_subprocess_without_audio_500(monkeypatch, infer_env, client):
    monkeypatch.setattr(mi, "asyncio", _fake_asyncio(_make_exec(produce=False)))

    resp = await client.post(
        "/api/melotts/infer",
        json={"model_path": str(infer_env["model_dir"]), "text": "你好"},
    )
    assert resp.status_code == 500
    assert "未产出音频" in resp.json()["detail"]
