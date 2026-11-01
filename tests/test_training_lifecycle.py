"""Training shutdown/deadline regressions without starting engines or processes."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from modelstation import main
from modelstation.api import sovits_svc as api
from modelstation.services import melotts_trainer as mt
from modelstation.services import sovits_svc_trainer as svc
from modelstation.services.training_mutex import (
    TRAINING_MELOTTS, current_training, reset_registry, try_begin_training,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_fails", [False, True])
async def test_lifespan_awaits_both_trainers(monkeypatch, first_fails):
    first = AsyncMock(side_effect=RuntimeError("failed") if first_fails else None)
    second = AsyncMock()
    monkeypatch.setattr(api, "_trainer_instance", SimpleNamespace(stop_training=first))
    monkeypatch.setattr(mt, "_trainer_instance", SimpleNamespace(stop_training=second))
    async with main.lifespan(None):
        first.assert_not_awaited()
    first.assert_awaited_once()
    second.assert_awaited_once()


class HangingStream:
    def __init__(self, fail=False):
        self.fail = fail
        self.readers = 0

    async def readline(self):
        self.readers += 1
        try:
            if self.fail:
                raise OSError("broken log pipe")
            await asyncio.Event().wait()
        finally:
            self.readers -= 1


class HangingProcess:
    def __init__(self, fail=False):
        self.pid = 999999
        self.stdout = HangingStream(fail)
        self.stderr = HangingStream()
        self.returncode = None
        self.terminated = False
        self.exited = asyncio.Event()
        self.waiters = 0

    async def wait(self):
        self.waiters += 1
        try:
            await self.exited.wait()
            return self.returncode
        finally:
            self.waiters -= 1

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.exited.set()

    kill = terminate


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["svc", "melo"])
@pytest.mark.parametrize("failure", ["timeout", "stream_error", "cancel"])
async def test_monitor_bounds_pipe_reads_and_cleans_tasks(monkeypatch, tmp_path, engine, failure):
    module = svc if engine == "svc" else mt
    monkeypatch.setattr(module, "_TRAIN_MONITOR_TIMEOUT", 0.02)
    monkeypatch.setattr(module, "_TRAIN_STOP_WAIT_TIMEOUT", 0.02)
    proc = HangingProcess(fail=failure == "stream_error")
    updates = []
    reset_registry()
    mt.reset_train_status()
    if engine == "svc":
        trainer = svc.SoVITSSVCTrainer(output_dir=str(tmp_path))
        monitor = trainer._monitor_training(1, lambda **kw: updates.append(kw), proc=proc)
    else:
        trainer = mt.MeloTTSTrainer(str(tmp_path), str(tmp_path), str(tmp_path))
        assert try_begin_training(TRAINING_MELOTTS, "test")[0]
        monitor = trainer._monitor_training("test", "output", 1, proc)
    task = asyncio.create_task(monitor)
    try:
        if failure == "cancel":
            await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await asyncio.wait_for(task, 1.0)
            final = updates[-1] if engine == "svc" else mt.get_train_status()
            assert final["status"] == "failed"
            assert ("超时" if failure == "timeout" else "broken log pipe") in final["message"]
        assert proc.terminated
        assert proc.stdout.readers == proc.stderr.readers == proc.waiters == 0
        assert current_training() is None
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        reset_registry()
        mt.reset_train_status()


@pytest.mark.asyncio
async def test_sovits_epoch_parsed_from_log_file(monkeypatch, tmp_path):
    """引擎 epoch 日志只落 train.log 文件（get_logger 仅 FileHandler）→ 监控从文件尾部解析并回调真实 epoch"""
    log_dir = tmp_path / "logs" / "mymodel"
    log_dir.mkdir(parents=True)
    (log_dir / "train.log").write_text(
        "2026-10-18\tINFO\t====> Epoch: 7, cost 1.00 s\n", encoding="utf-8"
    )
    monkeypatch.setattr(svc, "_EPOCH_FILE_POLL_SECONDS", 0.02)
    # HangingProcess 流永不 EOF：监控 gather 依赖超时收尾，缩短以匹配测试节奏
    monkeypatch.setattr(svc, "_TRAIN_MONITOR_TIMEOUT", 0.3)
    proc = HangingProcess()
    updates: list[dict] = []
    trainer = svc.SoVITSSVCTrainer(output_dir=str(tmp_path), so_vits_svc_dir=str(tmp_path))
    monitor = trainer._monitor_training(
        10, lambda **kw: updates.append(kw), proc=proc, output_name="mymodel"
    )
    task = asyncio.create_task(monitor)
    try:
        await asyncio.sleep(0.1)
        # stdout/stderr 未输出任何 epoch 行，epoch 只能来自 train.log 文件解析
        assert any(u.get("epoch") == 7 for u in updates), f"updates={updates}"
        proc.terminate()  # returncode=-15 → 监控按异常退出收尾并回填终态
        await asyncio.wait_for(task, 1.0)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        reset_registry()
        mt.reset_train_status()
    final = updates[-1]
    assert final["status"] == "failed"
    assert final["epoch"] == 7  # 终态回填的 epoch 取文件解析值，而非 0


# ---------------------------------------------------------------------------
# So-VITS-SVC 暂停 / 重启续训
# ---------------------------------------------------------------------------


class ExitingProcess:
    """按固定延时自然退出（exit=0）的训练进程桩；可被 terminate。"""

    def __init__(self, delay: float = 1.2):
        self.pid = 4242
        self.stdout = None
        self.stderr = None
        self.returncode: int | None = None
        self.delay = delay
        self.terminated = False

    async def wait(self) -> int:
        await asyncio.sleep(self.delay)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    kill = terminate


def _make_sovits_env(tmp_path: Path, output_name: str) -> tuple[svc.SoVITSSVCTrainer, Path]:
    """构造最小 sovits 引擎环境（configs/config.json + 预置 logs/检查点）。"""
    engine_dir = tmp_path / "engine"
    (engine_dir / "configs").mkdir(parents=True)
    (engine_dir / "configs" / "config.json").write_text(
        json.dumps(
            {
                "train": {"epochs": 1, "batch_size": 1, "learning_rate": 0.0001},
                "data": {},
            }
        ),
        encoding="utf-8",
    )
    models_dir = tmp_path / "models"
    trainer = svc.SoVITSSVCTrainer(
        output_dir=str(models_dir),
        training_data_dir=str(tmp_path / "data"),
        so_vits_svc_dir=str(engine_dir),
        python_path="python",
    )
    trainer._preprocessed.add("spk")
    logs_dir = engine_dir / "logs" / output_name
    logs_dir.mkdir(parents=True)
    (logs_dir / "G_1000.pth").write_bytes(b"fake-g")
    (logs_dir / "D_1000.pth").write_bytes(b"fake-d")
    (logs_dir / "train.log").write_text("====> Epoch: 4, cost 1.00 s\n", encoding="utf-8")
    (models_dir / output_name).mkdir(parents=True, exist_ok=True)
    return trainer, logs_dir


def _patch_sovits_spawn(trainer: svc.SoVITSSVCTrainer, delay: float = 1.2):
    """桩化 _spawn_train：登记启动参数并返回自然退出进程。"""
    calls: list[list[str]] = []

    async def fake_spawn(args: list[str]):
        calls.append(list(args))
        return ExitingProcess(delay=delay)

    trainer._spawn_train = fake_spawn  # type: ignore[method-assign]
    return calls


@pytest.mark.asyncio
async def test_sovits_pause_persists_and_restores_after_restart(tmp_path):
    """暂停落盘 train_state.json → 新 trainer 恢复 paused → 续训重新拉起 train.py"""
    trainer, logs_dir = _make_sovits_env(tmp_path, "resume_case")
    calls = _patch_sovits_spawn(trainer, delay=1.2)
    updates: list[dict] = []

    task_id = await trainer.start_training(
        epochs=100,
        batch_size=2,
        output_name="resume_case",
        speaker_name="spk",
        progress_callback=lambda **kw: updates.append(kw),
    )
    snapshot = await trainer.pause_training(timeout=5.0)

    assert snapshot["status"] == "paused"
    assert snapshot["epoch"] == 4
    assert trainer._active_ctx is not None
    assert trainer._process is None
    assert trainer.can_resume()
    state_file = logs_dir / "train_state.json"
    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert persisted["status"] == "paused"
    assert persisted["task_id"] == task_id
    assert Path(persisted["config_path"]).is_file()
    assert not (logs_dir / ".stop_requested").exists()

    # 模拟应用重启：进程内状态全部丢失，新实例从磁盘恢复
    fresh = svc.SoVITSSVCTrainer(
        output_dir=str(tmp_path / "models"),
        training_data_dir=str(tmp_path / "data"),
        so_vits_svc_dir=str(tmp_path / "engine"),
        python_path="python",
    )
    restored = fresh.restore_resumable_training()
    assert restored is not None
    assert restored["status"] == "paused"
    assert restored["epoch"] == 4
    assert fresh.can_resume()

    resumed_calls = _patch_sovits_spawn(fresh, delay=0.05)
    await fresh.resume_training(progress_callback=lambda **kw: updates.append(kw))
    assert len(resumed_calls) == 1
    assert resumed_calls[0][resumed_calls[0].index("-m") + 1] == "resume_case"
    assert any(u.get("status") == "running" for u in updates)
    await fresh.stop_training()


@pytest.mark.asyncio
async def test_sovits_interrupted_running_state_restores_as_paused(tmp_path):
    """应用被强杀（状态停在 running）→ 重启恢复为 paused，epoch 取日志最新值"""
    engine_dir = tmp_path / "engine"
    logs_dir = engine_dir / "logs" / "killed_case"
    logs_dir.mkdir(parents=True)
    (logs_dir / "G_2000.pth").write_bytes(b"fake-g")
    (logs_dir / "train.log").write_text("====> Epoch: 7\n", encoding="utf-8")
    config_path = tmp_path / "models" / "killed_case"
    config_path.mkdir(parents=True)
    (config_path / "config.json").write_text("{}", encoding="utf-8")
    (logs_dir / "train_state.json").write_text(
        json.dumps(
            {
                "task_id": "tid-killed",
                "output_name": "killed_case",
                "epochs": 100,
                "batch_size": 2,
                "config_path": str(config_path / "config.json"),
                "log_dir": str(logs_dir),
                "status": "running",
                "epoch": 0,
            }
        ),
        encoding="utf-8",
    )
    trainer = svc.SoVITSSVCTrainer(
        output_dir=str(tmp_path / "models"),
        so_vits_svc_dir=str(engine_dir),
        python_path="python",
    )
    restored = trainer.restore_resumable_training()
    assert restored is not None
    assert restored["status"] == "paused"
    assert restored["epoch"] == 7
    assert restored["progress"] == pytest.approx(0.07)
    assert trainer.can_resume()


@pytest.mark.asyncio
async def test_sovits_pause_resume_api_wiring(monkeypatch):
    """API /pause、/resume 接线：暂停释放互斥、继续重新占用"""
    fake = SimpleNamespace(
        pause_training=AsyncMock(
            return_value={
                "status": "paused",
                "task_id": "tid-api",
                "progress": 0.3,
                "epoch": 30,
                "total_epochs": 100,
                "message": "训练已暂停（epoch 30），检查点已保存",
            }
        ),
        resume_training=AsyncMock(return_value="tid-api"),
    )
    monkeypatch.setattr(api, "_trainer_instance", fake)
    reset_registry()
    api._train_status.update(
        task_id="tid-api", status="running", progress=0, epoch=0,
        total_epochs=100, message="",
    )
    try:
        assert try_begin_training("sovits_svc", "tid-api")[0]
        paused = await api.pause_training()
        assert paused["status"] == "success"
        assert paused["message"].startswith("训练已暂停")
        assert api._train_status["status"] == "paused"
        assert current_training() is None  # 暂停出口释放互斥

        resumed = await api.resume_training()
        assert resumed["task_id"] == "tid-api"
        assert api._train_status["status"] == "running"
        assert current_training() is not None  # 继续重新占用互斥
    finally:
        monkeypatch.setattr(api, "_trainer_instance", None)
        api._train_status.update(
            task_id=None, status="idle", progress=0, epoch=0,
            total_epochs=0, message="",
        )
        reset_registry()
