"""
MeloTTS API 冒烟测试（5 端点 + 未就绪报错路径 + 跨类型 409 语义）

覆盖（Task 3.4）：
- POST /api/melotts/preprocess（成功统计 / 非法路径 400）；
- POST /api/melotts/train（未就绪明确报错含 setup 指引 / 跨类型互斥 409 含类型与
  task_id / mock 全链路成功启动）；
- POST /api/melotts/stop、GET /api/melotts/status、GET /api/melotts/models 冒烟；
- 反向：melotts 占用时 POST /api/sovits-svc/train → 409（sovits 侧接线语义）。

全部子进程链路 mock（_run_subprocess/_spawn_train/_probe_melo_import），
不触碰真实引擎与 GPU。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import modelstation.config as config_module
import modelstation.services.security_utils as sec_module
from modelstation.config import ModelStationSettings
from modelstation.main import create_app
from modelstation.services import melotts_trainer as mt
from modelstation.services.melotts_trainer import MeloTTSTrainer
from modelstation.services.training_mutex import (
    TRAINING_MELOTTS,
    TRAINING_SOVITS_SVC,
    current_training,
    reset_registry,
    try_begin_training,
)


# ---------------------------------------------------------------------------
# 桩（与 test_melotts_trainer 相同的 FakeProc/FakeStream 模式）
# ---------------------------------------------------------------------------


class FakeStream:
    def __init__(self, lines: list[str]):
        self._lines = [line.encode("utf-8") + b"\n" for line in lines]

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class FakeProc:
    def __init__(self, exit_code: int = 0, delay: float = 0.05):
        self.pid = 4242
        self._exit_code = exit_code
        self._delay = delay
        self.returncode: int | None = None
        self.stdout = FakeStream([])
        self.stderr = FakeStream([])
        self.terminated = False

    async def wait(self) -> int:
        await asyncio.sleep(self._delay)
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 1

    def kill(self) -> None:
        self.terminated = True
        self.returncode = 1


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state():
    reset_registry()
    mt.reset_train_status()
    mt.reset_trainer()
    yield
    reset_registry()
    mt.reset_train_status()
    mt.reset_trainer()


@pytest.fixture
def melotts_env(monkeypatch, tmp_path):
    """隔离数据根 + 补丁 settings.melotts 指向 tmp mock 引擎环境。

    返回 paths dict（engine_dir/training_data_dir/models_dir/dataset_root）。
    """
    base = tmp_path / "env"
    engine_dir = base / "engines" / "MeloTTS"
    melo_dir = engine_dir / "melo"
    (melo_dir / "configs").mkdir(parents=True)
    (melo_dir / "train.py").write_text("# fake", encoding="utf-8")
    (melo_dir / "preprocess_text.py").write_text("# fake", encoding="utf-8")
    (melo_dir / "configs" / "config.json").write_text("{}", encoding="utf-8")
    training_data_dir = base / "data" / "training" / "melotts"
    models_dir = base / "data" / "models" / "melotts"
    dataset_root = base / "data" / "training" / "sovits_svc" / "raw"
    for d in (training_data_dir, models_dir, dataset_root):
        d.mkdir(parents=True, exist_ok=True)
    # 预训练底模缓存就位（真实链路会下载+转换；API 单测不触网）
    pretrained_dir = models_dir / "_pretrained"
    pretrained_dir.mkdir(parents=True)
    (pretrained_dir / "melotts_zh_base.pth").write_bytes(b"stub-base")

    monkeypatch.setattr(sec_module, "_MS_ROOT", base)
    monkeypatch.setattr(
        sec_module, "_TRAINING_DATA_ROOT", (base / "data" / "training").resolve()
    )

    def _fake_settings():
        return ModelStationSettings.model_validate(
            {
                "melotts": {
                    "engine_dir": str(engine_dir),
                    "training_data_dir": str(training_data_dir),
                    "models_dir": str(models_dir),
                }
            }
        )

    monkeypatch.setattr(config_module, "get_settings", _fake_settings)
    return {
        "engine_dir": engine_dir,
        "melo_dir": melo_dir,
        "training_data_dir": training_data_dir,
        "models_dir": models_dir,
        "dataset_root": dataset_root,
        "base": base,
    }


def _build_dataset(paths: dict, speaker: str, total: int = 10) -> Path:
    """构建含 v2 manifest 的 speaker 数据集（wav 桩文件在盘）"""
    ds_dir = paths["dataset_root"] / speaker
    ds_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, total + 1):
        (ds_dir / f"{i:04d}.wav").write_bytes(b"RIFF-fake")
    manifest = {
        "version": 2,
        "entries": [
            {"fingerprint": f"fp-{i}", "file": f"{i:04d}.wav", "text": f"第{i}句",
             "engine": "voxcpm"}
            for i in range(1, total + 1)
        ],
    }
    (ds_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return ds_dir


def _patch_probe_ok(monkeypatch) -> None:
    """对 api 单例 trainer 注入就绪试探成功桩"""
    from modelstation.services.melotts_trainer import get_trainer

    trainer = get_trainer()

    async def _ok() -> None:
        return None

    monkeypatch.setattr(trainer, "_probe_melo_import", _ok)


def _patch_full_fake_chain(monkeypatch, paths: dict, proc: FakeProc) -> None:
    """注入官方清洗 + train.py 启动桩（成功链路）"""
    from modelstation.services.melotts_trainer import get_trainer

    trainer: MeloTTSTrainer = get_trainer()

    async def _ok() -> None:
        return None

    async def fake_run(args: list[str], cwd: Path, timeout: float):
        # 训练后的声码器恢复子进程：本桩直接视为成功，不触碰文件
        if "--metadata" not in args:
            return 0, "", ""
        metadata = Path(args[args.index("--metadata") + 1])
        train_out = Path(args[args.index("--train-path") + 1])
        config_out = metadata.parent / "config.json"
        lines = [l for l in metadata.read_text(encoding="utf-8").splitlines() if l.strip()]
        train_out.write_text(
            "\n".join(l + "|ph|1|1" for l in lines) + "\n", encoding="utf-8"
        )
        config_out.write_text(
            json.dumps(
                {"train": {}, "data": {"training_files": "", "validation_files": ""}}
            ),
            encoding="utf-8",
        )
        return 0, "", ""

    async def fake_spawn(args: list[str], cwd: Path, env: dict) -> FakeProc:
        output_name = args[args.index("--model") + 1]
        logs_dir = Path(cwd) / "logs" / output_name
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "G_100.pth").write_bytes(b"fake-g")
        return proc

    monkeypatch.setattr(trainer, "_probe_melo_import", _ok)
    monkeypatch.setattr(trainer, "_run_subprocess", fake_run)
    monkeypatch.setattr(trainer, "_spawn_train", fake_spawn)


@pytest_asyncio.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _wait_status_via_api(client: AsyncClient, status: str, timeout: float = 5.0) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        resp = await client.get("/api/melotts/status")
        body = resp.json()
        if body["status"] == status:
            return body
        await asyncio.sleep(0.02)
    raise AssertionError(f"状态未在 {timeout}s 内到达 {status!r}: {body}")


# ---------------------------------------------------------------------------
# 冒烟：status / models / stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_idle_shape(client, melotts_env):
    resp = await client.get("/api/melotts/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "idle"
    assert body["task_id"] is None
    assert isinstance(body["models"], list)


@pytest.mark.asyncio
async def test_models_empty_list(client, melotts_env):
    resp = await client.get("/api/melotts/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["models"] == []


@pytest.mark.asyncio
async def test_export_model_copies_to_dest(client, melotts_env, tmp_path):
    """POST /export：整个模型目录复制到 <dest>/<model>；同名目标 409、相对路径 400"""
    model_dir = melotts_env["models_dir"] / "demo_model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "G_100.pth").write_bytes(b"g" * 1024)
    (model_dir / "D_100.pth").write_bytes(b"d" * 512)

    dest = tmp_path / "exports"
    resp = await client.post(
        "/api/melotts/export",
        json={"model_name": "demo_model", "dest_dir": str(dest)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["export_path"] == str(dest / "demo_model")
    assert body["file_count"] == 3
    assert body["total_bytes"] == 1024 + 512 + 2
    assert (dest / "demo_model" / "G_100.pth").is_file()
    assert (dest / "demo_model" / "config.json").is_file()

    # 仅导出指定 G 检查点 + 配置（不含 D）
    dest_ckpt = tmp_path / "exports_ckpt"
    single = await client.post(
        "/api/melotts/export",
        json={
            "model_name": "demo_model",
            "dest_dir": str(dest_ckpt),
            "checkpoint": "G_100.pth",
        },
    )
    assert single.status_code == 200
    single_body = single.json()
    assert single_body["file_count"] == 2
    assert single_body["total_bytes"] == 1024 + 2
    assert (dest_ckpt / "demo_model" / "G_100.pth").is_file()
    assert not (dest_ckpt / "demo_model" / "D_100.pth").exists()

    # 非 G 检查点 / 不存在的检查点：400
    bad_ckpt = await client.post(
        "/api/melotts/export",
        json={
            "model_name": "demo_model",
            "dest_dir": str(tmp_path / "exports_bad"),
            "checkpoint": "D_100.pth",
        },
    )
    assert bad_ckpt.status_code == 400
    missing_ckpt = await client.post(
        "/api/melotts/export",
        json={
            "model_name": "demo_model",
            "dest_dir": str(tmp_path / "exports_missing"),
            "checkpoint": "G_999.pth",
        },
    )
    assert missing_ckpt.status_code == 400

    # 目标已存在：拒绝覆盖
    again = await client.post(
        "/api/melotts/export",
        json={"model_name": "demo_model", "dest_dir": str(dest)},
    )
    assert again.status_code == 409

    # 相对路径 / 非法模型名：400
    rel = await client.post(
        "/api/melotts/export",
        json={"model_name": "demo_model", "dest_dir": "relative/dir"},
    )
    assert rel.status_code == 400
    bad_name = await client.post(
        "/api/melotts/export",
        json={"model_name": "../demo_model", "dest_dir": str(tmp_path / "x")},
    )
    assert bad_name.status_code == 400


@pytest.mark.asyncio
async def test_train_log_tail(client, melotts_env):
    """GET /log：无训练记录时 available=false；有日志时返回尾部并钳制行数"""
    resp = await client.get("/api/melotts/log")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert body["log_path"] is None
    assert body["lines"] == []

    from modelstation.services.melotts_trainer import get_trainer

    trainer = get_trainer()
    log_dir = melotts_env["engine_dir"] / "melo" / "logs" / "demo"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "train.log"
    log_path.write_text(
        "\n".join(f"line-{i}" for i in range(50)) + "\n", encoding="utf-8"
    )
    trainer._last_log_path = log_path

    resp = await client.get("/api/melotts/log?lines=20")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["log_path"] == str(log_path)
    assert len(body["lines"]) == 20
    assert body["lines"][-1] == "line-49"


@pytest.mark.asyncio
async def test_stop_when_idle(client, melotts_env):
    resp = await client.post("/api/melotts/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preprocess_success_stats(client, melotts_env):
    ds_dir = _build_dataset(melotts_env, "api_spk", total=10)
    resp = await client.post(
        "/api/melotts/preprocess", json={"dataset_dir": str(ds_dir), "speaker_name": "api_spk"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    stats = body["stats"]
    assert stats["total"] == 10
    assert stats["used"] == 10
    assert stats["train_count"] + stats["val_count"] == 10
    assert Path(stats["metadata_file"]).is_file()
    # 产出位于 melotts training_data_dir 下
    assert Path(stats["output_dir"]).parent == melotts_env["training_data_dir"]


@pytest.mark.asyncio
async def test_preprocess_invalid_dir_400(client, melotts_env):
    resp = await client.post(
        "/api/melotts/preprocess", json={"dataset_dir": "C:/Windows/System32/evil"}
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_preprocess_missing_manifest_400(client, melotts_env):
    empty = melotts_env["dataset_root"] / "no_manifest"
    empty.mkdir(parents=True, exist_ok=True)
    resp = await client.post("/api/melotts/preprocess", json={"dataset_dir": str(empty)})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# train：未就绪 / 409 跨类型 / 成功链路
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_train_engine_not_ready_guidance(client, monkeypatch, tmp_path):
    """引擎目录缺失 → 500 明确报错含 tools/setup_engines.py --clone-melotts 指引"""
    base = tmp_path / "noenv"

    def _fake_settings():
        return ModelStationSettings.model_validate(
            {"melotts": {"engine_dir": str(base / "missing_engine")}}
        )

    monkeypatch.setattr(config_module, "get_settings", _fake_settings)
    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://test"
    ) as c:
        resp = await c.post("/api/melotts/train", json={"epochs": 5})
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "--clone-melotts" in detail
    assert "setup_engines" in detail
    assert current_training() is None


@pytest.mark.asyncio
async def test_train_409_when_sovits_occupies(client, melotts_env, monkeypatch):
    """sovits 占用 → melotts train 409，错误含当前类型与 task_id（spec 冻结场景）"""
    _patch_probe_ok(monkeypatch)
    ok, _ = try_begin_training(TRAINING_SOVITS_SVC, "tid-sovits-77")
    assert ok
    resp = await client.post("/api/melotts/train", json={"epochs": 5})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["current_training"]["owner_type"] == TRAINING_SOVITS_SVC
    assert detail["current_training"]["task_id"] == "tid-sovits-77"


@pytest.mark.asyncio
async def test_train_409_when_melotts_occupies(client, melotts_env, monkeypatch):
    """melotts 运行中再次 train → 409（本地/互斥双重语义一致）"""
    _patch_full_fake_chain(monkeypatch, melotts_env, FakeProc(exit_code=0, delay=2.0))
    ds_dir = _build_dataset(melotts_env, "busy_spk")
    prep = await client.post("/api/melotts/preprocess", json={"dataset_dir": str(ds_dir)})
    assert prep.status_code == 200

    first = await client.post("/api/melotts/train", json={"epochs": 5})
    assert first.status_code == 200
    second = await client.post("/api/melotts/train", json={"epochs": 5})
    assert second.status_code == 409
    detail = second.json()["detail"]
    assert detail["current_training"]["owner_type"] == TRAINING_MELOTTS
    assert detail["current_training"]["task_id"] == first.json()["task_id"]

    # 清理：停止训练释放互斥
    stop = await client.post("/api/melotts/stop")
    assert stop.status_code == 200
    assert current_training() is None


@pytest.mark.asyncio
async def test_train_full_chain_success_smoke(client, melotts_env, monkeypatch):
    """mock 全链路：prep → train → status running→completed → models 列出产物"""
    _patch_full_fake_chain(monkeypatch, melotts_env, FakeProc(exit_code=0, delay=0.05))
    ds_dir = _build_dataset(melotts_env, "smoke_spk")
    prep = await client.post("/api/melotts/preprocess", json={"dataset_dir": str(ds_dir)})
    assert prep.status_code == 200

    resp = await client.post(
        "/api/melotts/train",
        json={"epochs": 7, "batch_size": 2, "learning_rate": 0.001, "output_name": "smoke_m"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    task_id = body["task_id"]

    final = await _wait_status_via_api(client, "completed")
    assert final["task_id"] == task_id
    assert final["total_epochs"] == 7
    assert final["progress"] == 1.0

    models_resp = await client.get("/api/melotts/models")
    assert models_resp.status_code == 200
    models = models_resp.json()["models"]
    assert any(m["name"] == "smoke_m" for m in models)
    assert models[0]["g_model"]  # G checkpoint 已收集

    # 完成出口：互斥已释放
    assert current_training() is None


@pytest.mark.asyncio
async def test_train_missing_prep_400(client, melotts_env, monkeypatch):
    """无 prep 产物 → 400（含 preprocess 指引）"""
    _patch_probe_ok(monkeypatch)
    resp = await client.post("/api/melotts/train", json={"epochs": 5})
    assert resp.status_code == 400
    assert "preprocess" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 反向互斥：melotts 占用时 sovits train → 409（sovits 侧接线语义）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sovits_train_409_when_melotts_occupies(client):
    """sovits train 入口消费共享互斥：melotts 占用时 409 且错误含类型与 task_id"""
    ok, _ = try_begin_training(TRAINING_MELOTTS, "tid-melotts-88")
    assert ok
    resp = await client.post("/api/sovits-svc/train", json={"epochs": 10})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["current_training"]["owner_type"] == TRAINING_MELOTTS
    assert detail["current_training"]["task_id"] == "tid-melotts-88"
    # sovits 本地状态已复位（未被占用残留阻塞后续请求）
    status_resp = await client.get("/api/sovits-svc/status")
    assert status_resp.json()["status"] == "idle"


@pytest.mark.asyncio
async def test_sovits_train_local_conflict_unchanged(client):
    """sovits 独占场景行为不变：本地 running 状态时 409 文本语义保持现状"""
    from modelstation.api import sovits_svc as sovits_api

    sovits_api._train_status["status"] = "running"
    try:
        resp = await client.post("/api/sovits-svc/train", json={"epochs": 10})
        assert resp.status_code == 409
        assert resp.json()["detail"] == "训练任务正在进行中"
    finally:
        sovits_api._train_status["status"] = "idle"
