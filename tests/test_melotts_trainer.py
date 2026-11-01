"""
MeloTTS 训练器（melotts_trainer）单元测试（mock 子进程/asyncio）

覆盖（Task 3.4）：
- 状态机：busy_starting → running → completed / failed（mock 子进程）；
- 就绪校验失败明确报错（含 tools/setup_engines.py --clone-melotts 指引）；
- 跨类型训练互斥双向（注册表层面）；
- 所有出口 end_training（监控收尾 / stop / start 异常路径，无泄漏）。
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from modelstation.config import ModelStationSettings
import modelstation.config as config_module
import modelstation.services.security_utils as sec_module
from modelstation.services import melotts_trainer as mt
from modelstation.services.melotts_dataset_prep import prepare_filelists
from modelstation.services.melotts_trainer import (
    MeloTTSTrainer,
    MelottsEngineNotReadyError,
    TrainingInProgressError,
    get_train_status,
)
from modelstation.services.training_mutex import (
    TRAINING_MELOTTS,
    TRAINING_SOVITS_SVC,
    current_training,
    end_training,
    reset_registry,
    try_begin_training,
)


# ---------------------------------------------------------------------------
# 测试桩
# ---------------------------------------------------------------------------


class FakeStream:
    """readline() 逐行返回预置行（UTF-8），耗尽后返回 b""（模拟 EOF）。"""

    def __init__(self, lines: list[str]):
        self._lines = [line.encode("utf-8") + b"\n" for line in lines]

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class FakeProc:
    """asyncio.subprocess.Process 最小桩：wait/terminate/kill + 可选输出流。"""

    def __init__(
        self,
        exit_code: int = 0,
        delay: float = 0.05,
        stdout_lines: list[str] | None = None,
        stderr_lines: list[str] | None = None,
    ):
        self.pid = 4242
        self._exit_code = exit_code
        self._delay = delay
        self.returncode: int | None = None
        self.stdout = FakeStream(stdout_lines or []) if stdout_lines is not None else None
        self.stderr = FakeStream(stderr_lines or []) if stderr_lines is not None else None
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        await asyncio.sleep(self._delay)
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 1  # 模拟被终止后的非零退出码

    def kill(self) -> None:
        self.killed = True
        self.returncode = 1


@pytest.fixture(autouse=True)
def _clean_mutex_and_status():
    """每测清空互斥注册表与训练状态（防跨测试泄漏）"""
    reset_registry()
    mt.reset_train_status()
    mt.reset_trainer()
    yield
    reset_registry()
    mt.reset_train_status()
    mt.reset_trainer()


@pytest.fixture
def trainer_env(monkeypatch, tmp_path):
    """构建 mock 引擎环境 + 已注入桩的 trainer。

    返回 (trainer, paths_dict)。_probe_melo_import 默认桩为成功；
    _run_subprocess / _spawn_train 由各测试按需覆盖。
    """
    base = tmp_path / "env"
    engine_dir = base / "engines" / "MeloTTS"
    melo_dir = engine_dir / "melo"
    (melo_dir / "configs").mkdir(parents=True)
    (melo_dir / "train.py").write_text("# fake train entry", encoding="utf-8")
    (melo_dir / "preprocess_text.py").write_text("# fake prep entry", encoding="utf-8")
    (melo_dir / "configs" / "config.json").write_text(
        json.dumps({"train": {}, "data": {}}), encoding="utf-8"
    )
    training_data_dir = base / "data" / "training" / "melotts"
    models_dir = base / "data" / "models" / "melotts"
    training_data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    # 预训练底模缓存就位（真实链路会下载+转换；单测只验证训练编排，不触网）
    pretrained_dir = models_dir / "_pretrained"
    pretrained_dir.mkdir(parents=True)
    (pretrained_dir / "melotts_zh_base.pth").write_bytes(b"stub-base")

    # prep 缺省 output_dir 依赖 get_settings → 补丁到 tmp
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

    # 数据集路径校验根补丁（prep 内部使用）
    ms_root = tmp_path / "env"
    monkeypatch.setattr(sec_module, "_MS_ROOT", ms_root)
    monkeypatch.setattr(sec_module, "_TRAINING_DATA_ROOT", (base / "data" / "training").resolve())

    trainer = MeloTTSTrainer(
        engine_dir=str(engine_dir),
        training_data_dir=str(training_data_dir),
        models_dir=str(models_dir),
        python_path="python",
        language="ZH",
    )

    async def _ok_probe() -> None:
        return None

    monkeypatch.setattr(trainer, "_probe_melo_import", _ok_probe)
    paths = {
        "engine_dir": engine_dir,
        "melo_dir": melo_dir,
        "training_data_dir": training_data_dir,
        "models_dir": models_dir,
        "base": base,
    }
    return trainer, paths


def _make_prep(trainer: MeloTTSTrainer, paths: dict, speaker: str = "spk", total: int = 20) -> dict:
    """经真实 prep 模块产出训练 filelist（metadata.list + manifest_prep.json）。"""
    dataset_root = paths["base"] / "data" / "training" / "sovits_svc" / "raw" / speaker
    dataset_root.mkdir(parents=True, exist_ok=True)
    for i in range(1, total + 1):
        (dataset_root / f"{i:04d}.wav").write_bytes(b"RIFF-fake")
    manifest = {
        "version": 2,
        "entries": [
            {"fingerprint": f"fp-{i}", "file": f"{i:04d}.wav", "text": f"第{i}句",
             "engine": "voxcpm"}
            for i in range(1, total + 1)
        ],
    }
    (dataset_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    stats = prepare_filelists(str(dataset_root), speaker_name=speaker)
    assert stats["train_count"] + stats["val_count"] == total
    return stats


def _patch_fake_run_subprocess(trainer: MeloTTSTrainer, *, fail: bool = False, empty: bool = False):
    """桩化官方 preprocess_text.py 子进程：产出 config.json + official_train.list。"""

    async def fake_run(args: list[str], cwd: Path, timeout: float) -> tuple[int, str, str]:
        if fail:
            return 1, "", "simulated preprocess_text failure"
        # 训练后的声码器恢复子进程：本桩直接视为成功，不触碰文件
        if "--metadata" not in args:
            return 0, "", ""
        metadata = Path(args[args.index("--metadata") + 1])
        train_out = Path(args[args.index("--train-path") + 1])
        config_out = metadata.parent / "config.json"
        lines = [l for l in metadata.read_text(encoding="utf-8").splitlines() if l.strip()]
        payload = [] if empty else [l + "|ph one two|1 1 1|2 1 1" for l in lines]
        train_out.write_text("\n".join(payload) + "\n", encoding="utf-8")
        config = {
            "train": {"epochs": 1, "batch_size": 1, "learning_rate": 0.001},
            "data": {"training_files": "", "validation_files": "", "spk2id": {}},
        }
        config_out.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        return 0, "", ""

    trainer._run_subprocess = fake_run  # type: ignore[method-assign]


def _patch_fake_spawn(trainer: MeloTTSTrainer, paths: dict, proc: FakeProc):
    """桩化 train.py 子进程启动：登记调用参数，并在引擎 logs 目录预置产物。"""
    spawned: list[dict] = []

    async def fake_spawn(args: list[str], cwd: Path, env: dict) -> FakeProc:
        spawned.append({"args": args, "cwd": cwd, "env": env})
        # 模拟 train.py 产物落盘 ./logs/<model>（相对 melo CWD，实码硬编码）
        output_name = args[args.index("--model") + 1]
        logs_dir = Path(cwd) / "logs" / output_name
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "G_1000.pth").write_bytes(b"fake-g-ckpt")
        (logs_dir / "D_1000.pth").write_bytes(b"fake-d-ckpt")
        return proc

    trainer._spawn_train = fake_spawn  # type: ignore[method-assign]
    return spawned


async def _wait_for_status(status: str, timeout: float = 5.0) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        st = get_train_status()
        if st["status"] == status:
            return st
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"训练状态未在 {timeout}s 内到达 {status!r}，当前: {get_train_status()}"
    )


async def _wait_for_mutex_release(timeout: float = 2.0) -> None:
    """等待监控收尾释放训练互斥（终态回填先于 finally 的 end_training）。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if current_training() is None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"训练互斥未在 {timeout}s 内释放，当前: {current_training()}"
    )


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_machine_idle_to_completed(trainer_env):
    """start → running → 子进程正常退出 → completed + 产物收集 + 互斥释放"""
    trainer, paths = trainer_env
    prep = _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)
    proc = FakeProc(exit_code=0, delay=0.05)
    spawned = _patch_fake_spawn(trainer, paths, proc)

    task_id = await trainer.start_training(epochs=100, batch_size=2, learning_rate=1e-3)
    assert task_id
    st = get_train_status()
    assert st["task_id"] == task_id
    assert st["status"] in ("busy_starting", "running")
    assert st["total_epochs"] == 100

    # 占用注册表：owner=melotts 且 task_id 一致
    current = current_training()
    assert current is not None and current["owner_type"] == TRAINING_MELOTTS
    assert current["task_id"] == task_id

    final = await _wait_for_status("completed")
    assert final["progress"] == 1.0
    # 产物已从引擎 logs 拷贝至 models_dir/<output_name>
    output_name = spawned[0]["args"][spawned[0]["args"].index("--model") + 1]
    out_dir = paths["models_dir"] / output_name
    assert (out_dir / "G_1000.pth").is_file()
    assert (out_dir / "D_1000.pth").is_file()
    # 训练 config 注入校验（超参 + filelist 绝对路径）
    config_path = Path(prep["output_dir"]) / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["train"]["epochs"] == 100
    assert cfg["train"]["batch_size"] == 2
    assert Path(cfg["data"]["training_files"]).is_absolute()
    assert Path(cfg["data"]["training_files"]).is_file()
    # 完成出口：互斥已释放（终态回填先于监控 finally，稍等其收尾）
    await _wait_for_mutex_release()
    assert current_training() is None


@pytest.mark.asyncio
async def test_epoch_log_parsing_updates_progress(trainer_env):
    """Train Epoch / ====> Epoch 日志行解析 → epoch/progress 更新"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)
    proc = FakeProc(
        exit_code=0,
        delay=0.15,
        stdout_lines=["Train Epoch: 3 [50%]", "====> Epoch: 5"],
        stderr_lines=[],
    )
    _patch_fake_spawn(trainer, paths, proc)

    await trainer.start_training(epochs=10)
    # stdout/stderr 桩瞬时吐完 → 轮询等待终态前，epoch 应已被解析
    final = await _wait_for_status("completed")
    assert final["epoch"] == 5
    assert final["progress"] == 1.0


@pytest.mark.asyncio
async def test_epoch_parsed_from_train_log_file(trainer_env, monkeypatch):
    """引擎 epoch 日志只落 train.log 文件（get_logger 仅 FileHandler）→ 监控从文件尾部解析真实进度"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)

    async def fake_spawn(args: list[str], cwd: Path, env: dict) -> FakeProc:
        # 模拟 train.py：产物与 train.log 落 ./logs/<model>，epoch 行只进文件不进流
        output_name = args[args.index("--model") + 1]
        logs_dir = Path(cwd) / "logs" / output_name
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "G_1000.pth").write_bytes(b"fake-g-ckpt")
        (logs_dir / "D_1000.pth").write_bytes(b"fake-d-ckpt")
        (logs_dir / "train.log").write_text(
            "2026-10-18\tINFO\t====> Epoch: 7\n", encoding="utf-8"
        )
        return FakeProc(exit_code=0, delay=0.3)

    trainer._spawn_train = fake_spawn  # type: ignore[method-assign]
    # 缩短文件轮询间隔，使 0.3s 的 mock 进程寿命内可观测到文件解析的 epoch
    monkeypatch.setattr(mt, "_EPOCH_FILE_POLL_SECONDS", 0.02)

    await trainer.start_training(epochs=10, output_name="logtail")
    final = await _wait_for_status("completed")
    # epoch 来自 train.log 文件解析（stdout/stderr 未输出任何 epoch 行）
    assert final["epoch"] == 7
    assert final["progress"] == 1.0


@pytest.mark.asyncio
async def test_nonzero_exit_marks_failed(trainer_env):
    """子进程非零退出 → failed；互斥释放（trainer 失败出口）"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)
    _patch_fake_spawn(trainer, paths, FakeProc(exit_code=2, delay=0.05))

    await trainer.start_training()
    final = await _wait_for_status("failed")
    assert "exit=2" in final["message"]
    await _wait_for_mutex_release()
    assert current_training() is None


@pytest.mark.asyncio
async def test_time_based_estimate_when_no_epoch_logs(trainer_env, monkeypatch):
    """完全解析不到 epoch 日志 → 按已运行时长估算进度且 message 注明估算"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)
    # 长 delay + 无 epoch 行 + 极小总 epochs → 估算很快达上限 0.95
    _patch_fake_spawn(trainer, paths, FakeProc(exit_code=0, delay=0.4))
    # 缩短估算 tick 间隔，使 0.4s 的 mock 进程寿命内可观测到估算 message
    monkeypatch.setattr(mt, "_ESTIMATE_INTERVAL_SECONDS", 0.03)

    await trainer.start_training(epochs=1)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 3.0
    saw_estimate = False
    while loop.time() < deadline:
        st = get_train_status()
        if "估算" in (st.get("message") or ""):
            saw_estimate = True
            break
        if st["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.02)
    assert saw_estimate, f"未观察到估算 message: {get_train_status()}"
    final = await _wait_for_status("completed")
    assert final["progress"] == 1.0


# ---------------------------------------------------------------------------
# 就绪校验
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ready_check_missing_engine_guidance(tmp_path):
    """引擎目录缺失 → MelottsEngineNotReadyError 含 --clone-melotts 指引；无互斥泄漏"""
    reset_registry()
    trainer = MeloTTSTrainer(
        engine_dir=str(tmp_path / "no_such_engine"),
        training_data_dir=str(tmp_path / "d"),
        models_dir=str(tmp_path / "m"),
    )
    with pytest.raises(MelottsEngineNotReadyError) as excinfo:
        await trainer.start_training()
    msg = str(excinfo.value)
    assert "--clone-melotts" in msg
    assert "setup_engines" in msg
    assert current_training() is None
    assert get_train_status()["status"] == "idle"


@pytest.mark.asyncio
async def test_ready_check_import_probe_failure(trainer_env):
    """melo 包导入试探失败 → 明确报错含依赖安装/克隆指引"""
    trainer, paths = trainer_env

    async def _bad_probe() -> None:
        raise MelottsEngineNotReadyError(
            "MeloTTS 训练管线不可导入: ModuleNotFoundError。"
            "修复指引: tools/setup_engines.py --clone-melotts 或 pip install -e ."
        )

    trainer._probe_melo_import = _bad_probe  # type: ignore[method-assign]
    _make_prep(trainer, paths)
    with pytest.raises(MelottsEngineNotReadyError) as excinfo:
        await trainer.start_training()
    assert "clone-melotts" in str(excinfo.value)
    assert current_training() is None


# ---------------------------------------------------------------------------
# 跨类型训练互斥（注册表层，双向）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mutex_conflict_sovits_occupies(trainer_env):
    """sovits 占用注册表时 melotts start_training → TrainingInProgressError
    （错误携带当前类型与 task_id），且不误清 sovits 的占用"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    ok, _ = try_begin_training(TRAINING_SOVITS_SVC, "tid-sovits-1")
    assert ok
    with pytest.raises(TrainingInProgressError) as excinfo:
        await trainer.start_training()
    current = excinfo.value.current
    assert current is not None
    assert current["owner_type"] == TRAINING_SOVITS_SVC
    assert current["task_id"] == "tid-sovits-1"
    # sovits 占用未被误清
    assert current_training()["task_id"] == "tid-sovits-1"


@pytest.mark.asyncio
async def test_mutex_owner_melotts_registered_with_task_id(trainer_env):
    """melotts 启动期间注册表持有 owner=melotts + 真实 task_id"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)
    proc = FakeProc(exit_code=0, delay=1.5)
    _patch_fake_spawn(trainer, paths, proc)
    task_id = await trainer.start_training()
    current = current_training()
    assert current["owner_type"] == TRAINING_MELOTTS
    assert current["task_id"] == task_id
    # 收尾
    await trainer.stop_training()
    assert current_training() is None


@pytest.mark.asyncio
async def test_local_running_conflict(trainer_env):
    """trainer 本地已有运行中子进程 → TrainingInProgressError（不触碰注册表）"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    ok, _ = try_begin_training(TRAINING_MELOTTS, "tid-holder")
    assert ok
    trainer._process = FakeProc(exit_code=0, delay=10)  # returncode=None → 运行中
    with pytest.raises(TrainingInProgressError):
        await trainer.start_training()
    assert current_training()["task_id"] == "tid-holder"


# ---------------------------------------------------------------------------
# 启动异常路径（互斥无泄漏）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_prep_artifact_raises(trainer_env):
    """无 prep 产物 → FileNotFoundError（含 preprocess 指引）；互斥无泄漏"""
    trainer, paths = trainer_env
    with pytest.raises(FileNotFoundError) as excinfo:
        await trainer.start_training()
    assert "preprocess" in str(excinfo.value)
    assert current_training() is None
    assert get_train_status()["status"] == "idle"


@pytest.mark.asyncio
async def test_preprocess_step_failure_releases_mutex(trainer_env):
    """官方清洗子进程失败（exit=1）→ RuntimeError；互斥释放 + 状态复位 idle"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer, fail=True)
    with pytest.raises(RuntimeError) as excinfo:
        await trainer.start_training()
    assert "preprocess_text.py" in str(excinfo.value)
    assert current_training() is None
    assert get_train_status()["status"] == "idle"


@pytest.mark.asyncio
async def test_resplit_empty_result_releases_mutex(trainer_env):
    """清洗结果为空（官方清洗全跳过）→ RuntimeError；互斥释放"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer, empty=True)
    with pytest.raises(RuntimeError) as excinfo:
        await trainer.start_training()
    assert "清洗结果为空" in str(excinfo.value)
    assert current_training() is None


@pytest.mark.asyncio
async def test_invalid_output_name_rejected_before_mutex(trainer_env):
    """非法 output_name（白名单外字符）→ ValueError 且未占用互斥
    （basename 归一已剥离路径分隔符，与 sovits trainer 清洗语义一致）"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    with pytest.raises(ValueError):
        await trainer.start_training(output_name="bad name!")
    assert current_training() is None


# ---------------------------------------------------------------------------
# stop 路径
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_training_terminate_and_release(trainer_env):
    """stop：terminate 生效 + 状态 stopped + 互斥释放（无泄漏）"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)
    proc = FakeProc(exit_code=0, delay=5.0)
    _patch_fake_spawn(trainer, paths, proc)

    await trainer.start_training()
    await trainer.stop_training()
    assert proc.terminated
    assert get_train_status()["status"] == "stopped"
    assert current_training() is None
    # trainer 进程句柄复位（CWD 免疫相关路径不依赖实例残留）
    assert trainer._process is None
    assert trainer._monitor_task is None


@pytest.mark.asyncio
async def test_stop_when_idle_is_noop(trainer_env):
    """无训练时 stop → 幂等安全，状态 stopped，互斥保持空"""
    trainer, _ = trainer_env
    await trainer.stop_training()
    assert get_train_status()["status"] == "stopped"
    assert current_training() is None


# ---------------------------------------------------------------------------
# 暂停 / 继续
# ---------------------------------------------------------------------------


def _patch_pausing_spawn(trainer: MeloTTSTrainer, *, delay: float = 1.2):
    """桩化 train.py：落盘检查点与 train.log，并记录每次拉起的参数。"""
    calls: list[list[str]] = []

    async def fake_spawn(args: list[str], cwd: Path, env: dict) -> FakeProc:
        calls.append(list(args))
        output_name = args[args.index("--model") + 1]
        logs_dir = Path(cwd) / "logs" / output_name
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "G_1000.pth").write_bytes(b"fake-g-ckpt")
        (logs_dir / "D_1000.pth").write_bytes(b"fake-d-ckpt")
        (logs_dir / "train.log").write_text(
            "2026-10-18\tINFO\t====> Epoch: 4\n", encoding="utf-8"
        )
        return FakeProc(exit_code=0, delay=delay)

    trainer._spawn_train = fake_spawn  # type: ignore[method-assign]
    return calls


@pytest.mark.asyncio
async def test_pause_saves_checkpoint_and_keeps_context(trainer_env):
    """暂停：.pause_requested → 引擎存盘退出 → paused，保留继续所需上下文"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)
    calls = _patch_pausing_spawn(trainer, delay=1.2)

    await trainer.start_training(epochs=100, batch_size=2, output_name="pause_case")
    st = await trainer.pause_training(timeout=5.0)
    await _wait_for_mutex_release()

    assert st["status"] == "paused"
    assert st["epoch"] == 4
    assert "已暂停" in st["message"]
    assert len(calls) == 1
    # 继续所需上下文保留；进程句柄清空；互斥已释放
    assert trainer._active_ctx is not None
    assert trainer._process is None
    assert current_training() is None
    assert trainer.can_resume()
    # 哨兵已由监控清理，检查点仍在引擎 logs 目录
    sentinel = paths["melo_dir"] / "logs" / "pause_case" / ".pause_requested"
    assert not sentinel.exists()
    assert (paths["melo_dir"] / "logs" / "pause_case" / "G_1000.pth").is_file()


@pytest.mark.asyncio
async def test_resume_restarts_from_saved_context(trainer_env):
    """继续：清理哨兵 → 重新拉起 train.py（同 config/model/预训练底模）→ running"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)
    calls = _patch_pausing_spawn(trainer, delay=2.0)

    await trainer.start_training(epochs=100, batch_size=2, output_name="resume_case")
    await trainer.pause_training(timeout=5.0)
    await _wait_for_mutex_release()
    assert len(calls) == 1

    task_id = await trainer.resume_training()

    assert task_id
    assert len(calls) == 2
    resumed_args = calls[1]
    assert resumed_args[resumed_args.index("--model") + 1] == "resume_case"
    assert "--c" in resumed_args
    # 底模来自缓存（fixture 预置），继续时沿用同一预训练权重
    assert "--pretrain_G" in resumed_args
    assert get_train_status()["status"] == "running"
    assert trainer._process is not None
    sentinel = paths["melo_dir"] / "logs" / "resume_case" / ".pause_requested"
    assert not sentinel.exists()

    await trainer.stop_training()
    assert current_training() is None
    assert trainer._active_ctx is None


@pytest.mark.asyncio
async def test_pause_without_running_raises(trainer_env):
    """非运行态调用暂停 → ValueError（API 层映射 400）"""
    trainer, _ = trainer_env
    with pytest.raises(ValueError):
        await trainer.pause_training(timeout=1.0)


@pytest.mark.asyncio
async def test_resume_without_paused_raises(trainer_env):
    """非暂停态调用继续 → ValueError"""
    trainer, _ = trainer_env
    assert not trainer.can_resume()
    with pytest.raises(ValueError):
        await trainer.resume_training()


@pytest.mark.asyncio
async def test_resume_conflicts_with_other_training(trainer_env):
    """暂停后其他训练占到互斥槽位 → resume 抛 TrainingInProgressError"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)
    _patch_pausing_spawn(trainer, delay=1.2)

    await trainer.start_training(epochs=100, batch_size=2, output_name="resume_conflict")
    await trainer.pause_training(timeout=5.0)
    await _wait_for_mutex_release()
    ok, _ = try_begin_training(TRAINING_SOVITS_SVC, "tid-sovits-2")
    assert ok
    with pytest.raises(TrainingInProgressError):
        await trainer.resume_training()
    # 冲突不污染状态：仍是 paused，可稍后重试
    assert get_train_status()["status"] == "paused"
    assert trainer.can_resume()


# ---------------------------------------------------------------------------
# 应用重启后从检查点续训（train_state.json 落盘 + 启动恢复）
# ---------------------------------------------------------------------------


def _new_trainer(paths: dict) -> MeloTTSTrainer:
    """按同一引擎/数据路径构造新 trainer 实例（模拟应用重启后的新进程）。"""
    return MeloTTSTrainer(
        engine_dir=str(paths["engine_dir"]),
        training_data_dir=str(paths["training_data_dir"]),
        models_dir=str(paths["models_dir"]),
        python_path="python",
        language="ZH",
    )


@pytest.mark.asyncio
async def test_paused_training_survives_app_restart(trainer_env):
    """暂停后关闭应用再打开：新实例从 train_state.json 恢复 paused，可继续训练"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)
    calls = _patch_pausing_spawn(trainer, delay=1.2)

    await trainer.start_training(epochs=100, batch_size=2, output_name="restart_case")
    await trainer.pause_training(timeout=5.0)
    await _wait_for_mutex_release()

    state_file = paths["melo_dir"] / "logs" / "restart_case" / "train_state.json"
    assert state_file.is_file()
    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert persisted["status"] == "paused"
    assert persisted["epoch"] == 4
    assert Path(persisted["config_path"]).is_file()

    # 模拟应用重启：进程内状态全部丢失，新 trainer 从磁盘恢复
    mt.reset_train_status()
    fresh = _new_trainer(paths)
    assert fresh.restore_resumable_training()
    st = get_train_status()
    assert st["status"] == "paused"
    assert st["epoch"] == 4
    assert st["total_epochs"] == 100
    assert "检测到上次未完成的训练" in st["message"]
    assert fresh.can_resume()

    resumed_calls = _patch_pausing_spawn(fresh, delay=0.05)
    task_id = await fresh.resume_training()
    assert task_id == persisted["task_id"]
    assert len(resumed_calls) == 1
    assert resumed_calls[0][resumed_calls[0].index("--model") + 1] == "restart_case"
    await fresh.stop_training()


@pytest.mark.asyncio
async def test_get_trainer_restores_paused_state_after_restart(trainer_env):
    """单例工厂重建时（应用启动首个请求）自动恢复 paused 态"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)
    _patch_pausing_spawn(trainer, delay=1.2)

    await trainer.start_training(epochs=100, batch_size=2, output_name="factory_case")
    await trainer.pause_training(timeout=5.0)
    await _wait_for_mutex_release()

    mt.reset_train_status()
    mt.reset_trainer()
    fresh = mt.get_trainer()
    assert get_train_status()["status"] == "paused"
    assert fresh.can_resume()


@pytest.mark.asyncio
async def test_interrupted_running_state_restored_as_paused(trainer_env):
    """应用被强杀（状态停在 running）→ 重启恢复为 paused，epoch 取日志最新值"""
    _, paths = trainer_env
    log_dir = paths["melo_dir"] / "logs" / "killed_case"
    log_dir.mkdir(parents=True)
    (log_dir / "G_2000.pth").write_bytes(b"fake-g-ckpt")
    (log_dir / "train.log").write_text("====> Epoch: 7\n", encoding="utf-8")
    config_path = paths["base"] / "run_config.json"
    config_path.write_text("{}", encoding="utf-8")
    state = {
        "task_id": "tid-killed",
        "output_name": "killed_case",
        "epochs": 100,
        "batch_size": 2,
        "config_path": str(config_path),
        "base_ckpt": "",
        "log_dir": str(log_dir),
        "status": "running",
        "epoch": 0,
        "updated_at": time.time(),
    }
    (log_dir / "train_state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
    )

    fresh = _new_trainer(paths)
    assert fresh.restore_resumable_training()
    st = get_train_status()
    assert st["status"] == "paused"
    assert st["epoch"] == 7
    assert st["progress"] == pytest.approx(0.07)
    assert fresh.can_resume()


@pytest.mark.asyncio
async def test_completed_or_stopped_training_not_restored(trainer_env):
    """最近一次训练已完成/主动停止 → 重启不恢复恢复态（不回翻旧检查点）"""
    trainer, paths = trainer_env
    _make_prep(trainer, paths)
    _patch_fake_run_subprocess(trainer)
    proc = FakeProc(exit_code=0, delay=0.05)
    _patch_fake_spawn(trainer, paths, proc)

    await trainer.start_training(epochs=10, batch_size=2, output_name="done_case")
    await _wait_for_status("completed")
    await _wait_for_mutex_release()
    state_file = paths["melo_dir"] / "logs" / "done_case" / "train_state.json"
    assert json.loads(state_file.read_text(encoding="utf-8"))["status"] == "completed"

    mt.reset_train_status()
    fresh = _new_trainer(paths)
    assert not fresh.restore_resumable_training()
    assert get_train_status()["status"] == "idle"
    assert not fresh.can_resume()

    # 主动停止（stopped）同样不恢复：最近一次会话是用户明确结束的
    proc2 = FakeProc(exit_code=0, delay=5.0)
    _patch_fake_spawn(trainer, paths, proc2)
    await trainer.start_training(epochs=10, batch_size=2, output_name="stop_case")
    await trainer.stop_training()
    mt.reset_train_status()
    fresh2 = _new_trainer(paths)
    assert not fresh2.restore_resumable_training()
