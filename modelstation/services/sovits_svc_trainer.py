"""
So-VITS-SVC 训练服务

自 CX-O-VoiceWorkStation/workstation/services/sovits_svc_trainer.py 迁移
（change-id: split-audio-workstation-cxfc-modelstation），
import 路径 workstation.* → modelstation.* 全量改写，逻辑不变。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Callable, Optional

from modelstation.services.gpu_batch import is_oom_signature, resolve_auto_batch_size
from modelstation.services.security_utils import validate_training_data_dir

logger = logging.getLogger(__name__)

_OUTPUT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_SPEAKER_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]")

# 自适应 batch 参数（本机实测标定：8 样本 + 模型常驻约 6.7GB ≈ 0.5GB/样本）
_PER_SAMPLE_GB = 0.5
_VRAM_RESERVE_GB = 2.0
_MAX_AUTO_BATCH = 36
_DEFAULT_BATCH = 4

# 训练 / 预处理子步骤默认超时（秒）
_TRAIN_SUBPROCESS_TIMEOUT = 3600.0  # 1 小时
# 训练监控超时（秒）：So-VITS-SVC 训练通常耗时数小时甚至数天，需远大于预处理超时
_TRAIN_MONITOR_TIMEOUT = 7 * 24 * 3600.0  # 7 days for training monitor
_TRAIN_STOP_WAIT_TIMEOUT = 10.0

# 注意：引擎 get_logger 只挂 FileHandler（engines/so-vits-svc-4.1-Stable/utils.py
# get_logger），epoch 日志仅落 logs/<名>/train.log 文件、不进子进程 stdout/stderr 流。
# 监控除解析流（兼容未来引擎改动）外，另按固定间隔读日志文件尾部提取最近 epoch，
# 作为真实进度经 progress_callback 回填（终态回填的 epoch 亦取该值）。
_EPOCH_FILE_POLL_SECONDS = 5.0
_EPOCH_LOG_TAIL_BYTES = 65536
_EPOCH_LOG_PATTERN = re.compile(r"(?:Train Epoch|====> Epoch):\s*(\d+)")

# 可续训状态落盘：应用关闭后进程内上下文丢失，需在引擎 logs 目录保留
# train_state.json；重启时据此恢复 paused 态（继续训练入口）。
_TRAIN_STATE_FILENAME = "train_state.json"
_RESUMABLE_STATE_STATUSES = frozenset({"paused", "running", "busy_starting"})
# 引擎现有优雅退出哨兵（train.py 在 epoch 边界存盘后退出）；暂停复用该哨兵
_STOP_SENTINEL_NAME = ".stop_requested"


def _parse_last_epoch_from_file(log_path: Optional[Path]) -> int:
    """从引擎 train.log 尾部解析最近一次 epoch 标记；文件缺失/不可读返回 0。"""
    if log_path is None:
        return 0
    try:
        with open(log_path, "rb") as fp:
            fp.seek(0, os.SEEK_END)
            size = fp.tell()
            fp.seek(max(0, size - _EPOCH_LOG_TAIL_BYTES))
            text = fp.read(_EPOCH_LOG_TAIL_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return 0
    last = 0
    for match in _EPOCH_LOG_PATTERN.finditer(text):
        last = int(match.group(1))
    return last


def _engine_env() -> dict:
    """引擎子进程环境：附带 torch≥2.6 的 weights_only 兼容开关。

    引擎（fairseq ContentVec / So-VITS 训练与推理）用 torch.load 读取包内预训练权重与
    用户本地训练产物，这些检查点含非张量对象（fairseq Dictionary、optimizer 状态等），
    而 torch 2.6 起 torch.load 默认 weights_only=True 会直接 UnpicklingError
    （实机复现：安装版 preprocess_hubert_f0 必失败，rc=1）。按 PyTorch 官方兼容开关
    （TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD）对"可信本地权重"放开，避免改动第三方库代码。
    """
    return {**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}


async def _wait_for_subprocess_exit(process: asyncio.subprocess.Process, timeout: float) -> bool:
    """等待子进程退出；超时则先 terminate 再 kill，返回是否主动 kill。"""
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        return False
    except asyncio.TimeoutError:
        logger.warning(f"Subprocess (pid={process.pid}) did not exit within {timeout}s; terminating")
        try:
            process.terminate()
        except ProcessLookupError:
            return True
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error(f"Subprocess (pid={process.pid}) did not exit after terminate; killing")
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        return True


def _sanitize_output_name(name: str) -> str:
    """校验 output_name 仅由字母/数字/下划线/连字符组成，防止目录穿越。"""
    base = os.path.basename(name or "")
    if not base or not _OUTPUT_NAME_PATTERN.match(base):
        raise ValueError(
            f"Invalid output_name: {name!r}. "
            "Only letters, digits, underscore and hyphen are allowed and path separators are forbidden."
        )
    return base


def _sanitize_speaker_name(name: str) -> str:
    """清洗 speaker_name：把不在白名单内的字符替换为下划线。"""
    if not name:
        return "speaker"
    cleaned = _SPEAKER_NAME_PATTERN.sub("_", name)
    cleaned = cleaned.strip("_") or "speaker"
    return cleaned


class SoVITSSVCTrainer:
    def __init__(
        self,
        output_dir: str = "data/models/sovits_svc",
        training_data_dir: str = "data/training/sovits_svc",
        so_vits_svc_dir: str = "",
        python_path: str = "python",
    ):
        self._output_dir = Path(output_dir)
        self._training_data_dir = Path(training_data_dir)
        self._so_vits_svc_dir = Path(so_vits_svc_dir) if so_vits_svc_dir else Path("so-vits-svc-4.1-Stable")
        self._python_path = python_path
        self._process: Optional[asyncio.subprocess.Process] = None
        self._task_id: Optional[str] = None
        self._preprocessed: set[str] = set()
        self._monitor_task: Optional[asyncio.Task] = None
        # 暂停/继续上下文：进程优雅退出后保留拉起训练所需的全部参数
        self._active_ctx: Optional[dict] = None
        # 用户主动暂停标记：区分 .stop_requested（用户暂停 vs 判停停止）
        self._pause_requested = False

    @property
    def training_data_dir(self) -> Path:
        """训练数据目录（公开访问接口）。"""
        return self._training_data_dir

    async def _run_subprocess(self, args: list[str]) -> tuple[int, str, str]:
        logger.info(f"Running subprocess: {' '.join(args)} (cwd={self._so_vits_svc_dir})")
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._so_vits_svc_dir),
            env=_engine_env(),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=_TRAIN_SUBPROCESS_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Subprocess timeout after {_TRAIN_SUBPROCESS_TIMEOUT}s: {' '.join(args)}"
            )
            await _wait_for_subprocess_exit(process, _TRAIN_STOP_WAIT_TIMEOUT)
            raise RuntimeError(
                f"Subprocess timed out after {_TRAIN_SUBPROCESS_TIMEOUT}s: {' '.join(args)}"
            )
        return process.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")

    async def preprocess(self, training_data_dir: str, speaker_name: str = "speaker") -> dict:
        # 清洗 speaker_name，移除所有非白名单字符避免路径穿越
        speaker_name = _sanitize_speaker_name(speaker_name)
        # 校验 training_data_dir 必须位于允许的根目录之下，防止目录穿越
        training_data_dir = validate_training_data_dir(training_data_dir)

        results = {}

        # 按上游 argparse 实码重写三步预处理参数：
        #   resample.py:               --sr2 <int> --in_dir <含 speaker 子目录的 raw 根> [--out_dir2]
        #   preprocess_flist_config.py: --train_list/--val_list/--source_dir（默认 ./filelists/*.txt、./dataset/44k）
        #   preprocess_hubert_f0.py:   -d/--device --in_dir --f0_predictor --num_processes
        # 三步 CWD 均为上游仓库根（_run_subprocess 固定 cwd），默认相对路径与上游标准
        # 工作流一致，故仅显式传必要参数。hubert 步骤读 configs/config.json
        # （flist 步骤产出，顺序已保证），--f0_predictor 显式取 pm 与推理默认对齐且免下载
        # rmvpe 模型。
        raw_root = training_data_dir / "raw"
        raw_dir = raw_root / speaker_name
        raw_dir.mkdir(parents=True, exist_ok=True)

        returncode, stdout, stderr = await self._run_subprocess(
            [self._python_path, "resample.py", "--sr2", "44100", "--in_dir", str(raw_root)]
        )
        results["resample"] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "success": returncode == 0,
        }
        if returncode != 0:
            logger.error(f"Resample failed: {stderr}")
            return results

        returncode, stdout, stderr = await self._run_subprocess(
            [self._python_path, "preprocess_flist_config.py"]
        )
        results["preprocess_flist_config"] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "success": returncode == 0,
        }
        if returncode != 0:
            logger.error(f"Preprocess flist config failed: {stderr}")
            return results

        returncode, stdout, stderr = await self._run_subprocess(
            [self._python_path, "preprocess_hubert_f0.py", "--f0_predictor", "pm"]
        )
        results["preprocess_hubert_f0"] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "success": returncode == 0,
        }
        if returncode != 0:
            logger.error(f"Preprocess hubert f0 failed: {stderr}")
            return results

        self._preprocessed.add(speaker_name)
        logger.info(f"Preprocessing completed successfully for speaker: {speaker_name}")
        return results

    @staticmethod
    def _write_runtime_config(
        source_config_path: Path,
        target_path: Path,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        eval_interval: Optional[int] = None,
    ) -> None:
        """读取上游 config.json，改写 train 段超参后写入 target_path。

        字段名以上游 configs_template/config_template.json 与 utils.get_hparams 实码为准：
        train.epochs / train.batch_size / train.learning_rate。
        eval_interval（可选）：一键成模判停保障（change-id:
        add-one-click-voice-model-pipeline）——判停终止训练子进程后引擎不再存盘，
        eval_interval 必须足够小以留存检查点；缺省 None 保持上游配置不变。
        """
        if not source_config_path.exists():
            raise RuntimeError(
                f"上游 config.json 不存在: {source_config_path}（请先完成 preprocess 再训练）"
            )
        with open(source_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        train_section = config.get("train") if isinstance(config, dict) else None
        if not isinstance(train_section, dict):
            raise RuntimeError(f"上游 config.json 缺少 train 段: {source_config_path}")
        train_section["epochs"] = int(epochs)
        train_section["batch_size"] = int(batch_size)
        train_section["learning_rate"] = float(learning_rate)
        if eval_interval is not None:
            train_section["eval_interval"] = int(eval_interval)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    async def _spawn_train(self, args: list[str]) -> asyncio.subprocess.Process:
        """启动训练子进程（独立方法便于测试注入 mock 进程）。"""
        return await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._so_vits_svc_dir),
            env=_engine_env(),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

    def _train_state_path(self, output_name: str) -> Path:
        """可续训状态文件路径（与引擎检查点同目录：logs/<output_name>/）。"""
        return self._so_vits_svc_dir / "logs" / output_name / _TRAIN_STATE_FILENAME

    def _persist_train_state(
        self,
        status: str,
        *,
        ctx: Optional[dict] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """把续训所需上下文原子落盘；失败仅告警，不影响训练主链路。"""
        context = dict(ctx or self._active_ctx or {})
        output_name = str(context.get("output_name") or "")
        if not output_name:
            return
        if epoch is not None:
            default_epoch = int(epoch)
        else:
            log_dir_value = str(context.get("log_dir") or "")
            default_epoch = (
                _parse_last_epoch_from_file(Path(log_dir_value) / "train.log")
                if log_dir_value
                else 0
            )
        payload = {
            "task_id": context.get("task_id"),
            "output_name": output_name,
            "epochs": int(context.get("epochs") or 0),
            "batch_size": int(context.get("batch_size") or 0),
            "learning_rate": float(context.get("learning_rate") or 0.0),
            "eval_interval": context.get("eval_interval"),
            "config_path": str(context.get("config_path") or ""),
            "log_dir": str(context.get("log_dir") or ""),
            "status": status,
            "epoch": default_epoch,
            "updated_at": time.time(),
        }
        path = self._train_state_path(output_name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("写入续训状态失败（忽略）: %s (%s)", path, exc)

    def restore_resumable_training(self) -> Optional[dict]:
        """应用重启后恢复最近一次未完成训练为 paused，返回状态快照（无可恢复时 None）。

        进程内上下文随应用关闭丢失，这里只读 logs/<output_name>/train_state.json；
        仅看最近一次训练会话（updated_at 最大），避免把历史暂停任务翻出来。
        """
        if self._active_ctx is not None:
            return None
        logs_root = self._so_vits_svc_dir / "logs"
        if not logs_root.is_dir():
            return None
        latest: Optional[tuple[float, Path, dict]] = None
        for state_path in logs_root.glob(f"*/{_TRAIN_STATE_FILENAME}"):
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                updated = float(data.get("updated_at") or 0.0)
            except (OSError, ValueError, TypeError):
                continue
            if latest is None or updated > latest[0]:
                latest = (updated, state_path, data)
        if latest is None:
            return None
        _, state_path, data = latest
        output_name = state_path.parent.name
        try:
            _sanitize_output_name(output_name)
        except ValueError:
            return None
        if str(data.get("status") or "") not in _RESUMABLE_STATE_STATUSES:
            return None
        try:
            epochs = int(data.get("epochs") or 0)
            batch_size = int(data.get("batch_size") or 0)
            stored_epoch = int(data.get("epoch") or 0)
        except (TypeError, ValueError):
            return None
        config_path = Path(str(data.get("config_path") or ""))
        log_dir = state_path.parent
        if epochs < 1 or not config_path.is_file():
            return None
        if not any(log_dir.glob("G_*.pth")):
            return None

        epoch = max(_parse_last_epoch_from_file(log_dir / "train.log"), stored_epoch)
        progress = min(epoch / epochs, 1.0) if epochs > 0 else 0.0
        self._active_ctx = {
            "task_id": data.get("task_id") or str(uuid.uuid4()),
            "output_name": output_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": float(data.get("learning_rate") or 0.0),
            "eval_interval": data.get("eval_interval"),
            "config_path": str(config_path),
            "log_dir": str(log_dir),
        }
        self._task_id = self._active_ctx["task_id"]
        logger.info(
            "已恢复可续训的 So-VITS-SVC 训练: output=%s epoch=%d state=%s",
            output_name, epoch, state_path,
        )
        return {
            "task_id": self._active_ctx["task_id"],
            "status": "paused",
            "progress": progress,
            "epoch": epoch,
            "total_epochs": epochs,
            "message": (
                f"检测到上次未完成的训练 {output_name}（epoch {epoch}），"
                "检查点已就绪，可点击「继续训练」从该检查点恢复"
            ),
        }

    def can_resume(self) -> bool:
        """当前是否处于可继续的暂停态（进程已退出且上下文仍在）。"""
        return self._active_ctx is not None and self._process is None

    async def start_training(
        self,
        epochs: int = 10000,
        batch_size: Optional[int] = None,
        learning_rate: float = 1e-4,
        output_name: Optional[str] = None,
        speaker_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        eval_interval: Optional[int] = None,
    ) -> str:
        if self._process and self._process.returncode is None:
            raise RuntimeError("训练已在进行，请先停止当前训练")

        # batch_size 缺省（None）= 按显存自适应；显式给定则以其为准（不强改用户意图）
        if batch_size is None:
            batch_size, batch_reason = await resolve_auto_batch_size(
                self._python_path,
                per_sample_gb=_PER_SAMPLE_GB,
                reserve_gb=_VRAM_RESERVE_GB,
                max_batch=_MAX_AUTO_BATCH,
                fallback=_DEFAULT_BATCH,
                env=_engine_env(),
            )
        else:
            batch_reason = f"请求显式指定 batch={int(batch_size)}"
        batch_size = int(batch_size)
        logger.info(f"So-VITS-SVC batch_size={batch_size}（{batch_reason}）")

        target_speaker = _sanitize_speaker_name(speaker_name or "speaker")
        if target_speaker not in self._preprocessed:
            raise RuntimeError("Preprocessing must be completed before training. Call preprocess() first.")

        self._task_id = str(uuid.uuid4())
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if output_name:
            output_name = _sanitize_output_name(output_name)
        else:
            output_name = f"sovits_svc_{self._task_id[:8]}"
        output_path = self._output_dir / output_name
        output_path.mkdir(parents=True, exist_ok=True)

        source_config_path = self._so_vits_svc_dir / "configs" / "config.json"
        model_name = output_name

        # 上游 train.py 仅通过 -c 接收超参（utils.get_hparams），请求中的
        # epochs/batch_size/learning_rate 按请求参数改写上游 config.json 的 train 段
        # 并落盘独立副本到本训练 output_path 下（多训练互不覆盖）。
        # batch 只影响该配置文件，故 OOM 降批重试可复用同一入口重新拉起进程。
        runtime_config_path = output_path / "config.json"

        args = [
            self._python_path,
            "train.py",
            "-c", str(runtime_config_path),
            "-m", model_name,
        ]

        async def _spawn_with_batch(batch: int) -> asyncio.subprocess.Process:
            """按给定 batch 重写运行配置并拉起训练子进程。"""
            self._write_runtime_config(
                source_config_path,
                runtime_config_path,
                epochs=epochs,
                batch_size=batch,
                learning_rate=learning_rate,
                eval_interval=eval_interval,
            )
            if self._active_ctx is not None:
                self._active_ctx["batch_size"] = int(batch)
                # OOM 降批后同步落盘，重启续训沿用新的 batch
                self._persist_train_state("running")
            return await self._spawn_train(args)

        logger.info(f"Starting So-VITS-SVC training: {self._task_id}")
        logger.info(f"  Training data: {self._training_data_dir}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        logger.info(f"  Runtime config: {runtime_config_path}")

        # 暂停/继续上下文：应用重启后按磁盘状态恢复（train_state.json）
        self._active_ctx = {
            "task_id": self._task_id,
            "output_name": model_name,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "eval_interval": eval_interval,
            "config_path": str(runtime_config_path.resolve()),
            "log_dir": str((self._so_vits_svc_dir / "logs" / model_name).resolve()),
        }
        self._pause_requested = False
        # 先落盘再拉起进程：应用被强杀（Electron taskkill /T）时也能恢复
        self._persist_train_state("busy_starting", epoch=0)
        try:
            proc = await _spawn_with_batch(batch_size)
        except BaseException:
            self._persist_train_state("failed")
            self._active_ctx = None
            raise
        self._process = proc
        self._persist_train_state("running", epoch=0)

        self._monitor_task = asyncio.create_task(
            self._monitor_training(
                epochs, progress_callback, proc, model_name,
                respawn=_spawn_with_batch, batch_size=batch_size,
            )
        )

        return self._task_id

    async def _read_stream(self, stream, callback):
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            callback(line.decode("utf-8", errors="replace").strip())

    def _collect_outputs(self, output_name: str) -> Path:
        """训练完成后收集产物：engine logs/<name> → models_dir/<name>。

        上游 train.py 将产物硬编码落盘 ``./logs/<model>``（相对引擎仓库根，
        实码见 engines/so-vits-svc-4.1-Stable/utils.py get_hparams 的 model_dir），
        本服务约定训练产物落盘 models_dir/<output_name>/，此处搬运；
        与 melotts_trainer._collect_outputs 同构（同一条链路两种引擎行为一致）。
        """
        src = self._so_vits_svc_dir / "logs" / output_name
        dst = self._output_dir / output_name
        dst.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            raise RuntimeError(
                f"训练子进程 exit=0 但引擎日志目录不存在: {src}（无产物可收集）"
            )
        for pattern in ("G_*.pth", "D_*.pth", "config.json"):
            for f in src.glob(pattern):
                shutil.copy2(f, dst / f.name)
        if not list(dst.glob("G_*.pth")):
            raise RuntimeError(
                f"训练完成但未在 {src} 找到 G_*.pth 产物；请检查训练日志"
            )
        return dst

    async def _monitor_training(
        self,
        total_epochs: int,
        progress_callback: Optional[Callable] = None,
        proc: Optional[asyncio.subprocess.Process] = None,
        output_name: Optional[str] = None,
        respawn: Optional[Callable] = None,
        batch_size: int = 0,
    ):
        epoch_pattern = re.compile(r"epoch:\s*(\d+)", re.IGNORECASE)
        current_epoch = 0
        last_output_line = ""
        # 近若干行输出：OOM 特征可能不在最后一行，需跨行上下文判定
        recent_output: "deque[str]" = deque(maxlen=40)
        process = proc or self._process
        if process is None:
            logger.warning("Monitor training called without a process")
            return

        # 引擎 epoch 日志只落文件（见模块常量注释），按日志尾部解析真实进度
        epoch_log_path = (
            self._so_vits_svc_dir / "logs" / output_name / "train.log" if output_name else None
        )

        async def _epoch_file_poller() -> None:
            """定期读 train.log 尾部提取最近 epoch，经 progress_callback 回填真实进度。"""
            nonlocal current_epoch
            while True:
                await asyncio.sleep(_EPOCH_FILE_POLL_SECONDS)
                if process.returncode is not None:
                    return
                epoch = _parse_last_epoch_from_file(epoch_log_path)
                if epoch > current_epoch:
                    current_epoch = epoch
                    progress = min(epoch / total_epochs, 1.0) if total_epochs > 0 else 0.0
                    if progress_callback:
                        try:
                            progress_callback(
                                progress=progress,
                                epoch=epoch,
                                total_epochs=total_epochs,
                                message=f"训练进行中（epoch {epoch}/{total_epochs}）",
                            )
                        except Exception as e:
                            logger.warning(f"Progress callback error: {e}")

        epoch_file_task = asyncio.create_task(_epoch_file_poller())

        def _process_line(line_str: str):
            nonlocal current_epoch, last_output_line
            if not line_str:
                return
            # 保留最后一行输出：失败终态时并入 message（子进程 traceback 落在
            # stdout/stderr 流里，不保留则排障时拿不到真实原因）
            last_output_line = line_str[:300]
            recent_output.append(line_str[:300])
            match = epoch_pattern.search(line_str)
            if match:
                current_epoch = int(match.group(1))
                progress = min(current_epoch / total_epochs, 1.0) if total_epochs > 0 else 0.0
                if progress_callback:
                    try:
                        progress_callback(
                            progress=progress,
                            epoch=current_epoch,
                            total_epochs=total_epochs,
                            message=line_str,
                        )
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

        retried_on_oom = False
        while True:
            stdout_task = asyncio.create_task(self._read_stream(process.stdout, _process_line))
            stderr_task = asyncio.create_task(self._read_stream(process.stderr, _process_line))
            wait_task = asyncio.create_task(process.wait())
            tasks = (stdout_task, stderr_task, wait_task)
            monitor_error = ""
            # 管道读取与进程等待共用期限；日志不再输出/不 EOF 时也必须能超时。
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=_TRAIN_MONITOR_TIMEOUT)
            except asyncio.TimeoutError:
                monitor_error = f"训练监控超时（{_TRAIN_MONITOR_TIMEOUT}s）"
                logger.error(monitor_error)
            except asyncio.CancelledError:
                epoch_file_task.cancel()
                await asyncio.gather(epoch_file_task, return_exceptions=True)
                if process.returncode is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    await _wait_for_subprocess_exit(process, _TRAIN_STOP_WAIT_TIMEOUT)
                raise
            except Exception as exc:
                monitor_error = f"训练日志读取失败：{exc}"
                logger.exception("Training monitor failed")
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            if monitor_error and process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                await _wait_for_subprocess_exit(process, _TRAIN_STOP_WAIT_TIMEOUT)

            returncode = -1 if monitor_error else process.returncode
            if returncode is None:
                returncode = -1
            if monitor_error:
                message = monitor_error
                break
            if returncode == 0:
                # 正常退出：暂停与完成在收尾阶段区分（暂停看 .stop_requested 标记）
                message = ""
                break
            # OOM 降批重试（仅一次）：自适应估算偏乐观时兜底，避免整轮训练白跑
            if (
                not retried_on_oom
                and respawn is not None
                and batch_size > 1
                and is_oom_signature("\n".join(recent_output))
            ):
                retried_on_oom = True
                new_batch = max(1, batch_size // 2)
                logger.warning(
                    f"So-VITS-SVC 显存不足（batch={batch_size}），降批到 {new_batch} 后重试一次"
                )
                if progress_callback:
                    try:
                        progress_callback(message=f"显存不足，batch {batch_size} → {new_batch} 后重试")
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
                recent_output.clear()
                process = await respawn(new_batch)
                self._process = process
                batch_size = new_batch
                continue

            tail = f"：{last_output_line}" if last_output_line else ""
            message = f"训练子进程异常退出（exit={returncode}）{tail}"
            logger.error(
                "So-VITS-SVC training failed: task_id=%s exit=%s tail=%s",
                self._task_id, returncode, last_output_line,
            )
            break

        # 收尾：停掉日志文件轮询（终态回填前，epoch 不再变化）
        epoch_file_task.cancel()
        await asyncio.gather(epoch_file_task, return_exceptions=True)

        # 终态回填前补一次文件解析：最后一次 epoch 行可能落在轮询间隔内
        final_epoch = max(current_epoch, _parse_last_epoch_from_file(epoch_log_path))
        log_dir = epoch_log_path.parent if epoch_log_path is not None else None
        paused = (
            returncode == 0
            and self._pause_requested
            and (total_epochs <= 0 or final_epoch < total_epochs)
        )
        if paused:
            # 用户主动暂停：清理哨兵；保留检查点与上下文供继续训练，不收集产物
            if log_dir is not None:
                try:
                    (log_dir / _STOP_SENTINEL_NAME).unlink()
                except FileNotFoundError:
                    pass
            message = (
                f"训练已暂停（epoch {final_epoch}），检查点已保存，可稍后继续训练"
            )
            self._pause_requested = False
            self._persist_train_state("paused", epoch=final_epoch)
        elif returncode == 0:
            # 产物收集：引擎 logs/<name> → models_dir/<name>（收集失败按训练失败上报，
            # 避免"训练成功但模型列表为空"的静默不一致）
            try:
                dst = self._collect_outputs(output_name or "")
                message = f"训练完成，产物已收集至 {dst}"
                logger.info(
                    "So-VITS-SVC training completed: task_id=%s -> %s", self._task_id, dst
                )
            except Exception as exc:
                returncode = -1
                message = f"训练子进程正常退出但产物收集失败: {exc}"
                logger.error("So-VITS-SVC output collection failed: %s", exc)
            self._pause_requested = False
            self._persist_train_state(
                "completed" if returncode == 0 else "failed", epoch=final_epoch
            )
            self._active_ctx = None
        else:
            self._pause_requested = False
            if not message:
                message = f"训练子进程异常退出（exit={returncode}）"
            self._persist_train_state("failed", epoch=final_epoch)
            self._active_ctx = None

        if progress_callback:
            try:
                progress_callback(
                    progress=(
                        min(final_epoch / total_epochs, 1.0)
                        if paused
                        else (1.0 if returncode == 0 else 0.0)
                    ),
                    epoch=final_epoch,
                    total_epochs=total_epochs,
                    status=(
                        "paused"
                        if paused
                        else ("completed" if returncode == 0 else "failed")
                    ),
                    message=message,
                )
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

        # 终态复位进程句柄（暂停态保留上下文供 resume 复用）
        self._process = None
        if self._monitor_task is asyncio.current_task():
            self._monitor_task = None

    async def pause_training(self, timeout: float = 120.0) -> dict:
        """暂停训练：写 .stop_requested 哨兵，等引擎在 epoch 边界存盘退出。

        返回暂停后的状态快照（status=paused）。训练进程退出但保留检查点，
        可随后调用 resume_training 从最近检查点继续。
        """
        process = self._process
        if (
            self._active_ctx is None
            or process is None
            or process.returncode is not None
        ):
            raise ValueError("当前没有正在运行的 So-VITS-SVC 训练任务")
        ctx = dict(self._active_ctx)
        log_dir = Path(ctx["log_dir"])
        sentinel = log_dir / _STOP_SENTINEL_NAME
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        self._pause_requested = True
        sentinel.write_text("", encoding="utf-8")
        logger.info("已写入暂停哨兵: %s", sentinel)

        monitor = self._monitor_task
        try:
            if monitor is not None:
                # shield：超时仅结束等待，不取消监控（由下方强制停止兜底）
                await asyncio.wait_for(asyncio.shield(monitor), timeout=max(1.0, float(timeout)))
        except asyncio.TimeoutError:
            logger.warning("暂停哨兵未在 %.0fs 内生效，退回强制停止（保留检查点）", timeout)
            await self.stop_training(preserve_resume=True)
            raise RuntimeError("暂停超时，已强制停止训练（产物取最近检查点）")

        if self._active_ctx is None:
            # 引擎在暂停生效前已自然到顶/失败：状态由监控回填，清理残留哨兵
            self._pause_requested = False
            try:
                sentinel.unlink()
            except FileNotFoundError:
                pass
            raise RuntimeError("训练已在暂停前结束，无可继续的检查点")

        epoch = _parse_last_epoch_from_file(log_dir / "train.log")
        epochs = int(ctx["epochs"])
        progress = min(epoch / epochs, 1.0) if epochs > 0 else 0.0
        return {
            "status": "paused",
            "task_id": ctx["task_id"],
            "progress": progress,
            "epoch": epoch,
            "total_epochs": epochs,
            "message": f"训练已暂停（epoch {epoch}），检查点已保存，可稍后继续训练",
        }

    async def resume_training(
        self, progress_callback: Optional[Callable] = None
    ) -> str:
        """从暂停时的检查点继续训练（同一 task_id / 模型目录）。

        上游 train.py 启动时自动加载 logs/<model>/ 下最新 G/D 检查点，
        因此只需按原运行配置重新拉起进程即可续训（无需重新预处理）。
        """
        if not self.can_resume():
            raise ValueError("当前没有可继续的 So-VITS-SVC 训练任务")
        ctx = dict(self._active_ctx or {})
        log_dir = Path(str(ctx.get("log_dir") or ""))
        config_path = Path(str(ctx.get("config_path") or ""))
        if not config_path.is_file():
            raise RuntimeError(f"训练配置不存在: {config_path}（无法续训）")
        if not any(log_dir.glob("G_*.pth")):
            raise RuntimeError(f"未找到检查点（{log_dir}/G_*.pth），无法续训")

        # 清理可能残留的停止哨兵，避免引擎刚启动就再次退出
        for name in (_STOP_SENTINEL_NAME, ".pause_requested"):
            try:
                (log_dir / name).unlink()
            except FileNotFoundError:
                pass

        self._task_id = str(ctx.get("task_id") or uuid.uuid4())
        ctx["task_id"] = self._task_id
        args = [
            self._python_path,
            "train.py",
            "-c", str(config_path),
            "-m", str(ctx["output_name"]),
        ]
        epoch = _parse_last_epoch_from_file(log_dir / "train.log")
        epochs = int(ctx["epochs"])
        progress = min(epoch / epochs, 1.0) if epochs > 0 else 0.0
        self._pause_requested = False
        self._persist_train_state("running", epoch=epoch)
        if progress_callback:
            try:
                progress_callback(
                    status="running",
                    progress=progress,
                    epoch=epoch,
                    total_epochs=epochs,
                    message=f"训练继续中（从 epoch {epoch} 恢复）",
                )
            except Exception as exc:
                logger.warning(f"Progress callback error: {exc}")

        try:
            proc = await self._spawn_train(args)
        except BaseException:
            self._persist_train_state("paused", epoch=epoch)
            raise
        self._process = proc
        self._monitor_task = asyncio.create_task(
            self._monitor_training(
                epochs,
                progress_callback,
                proc,
                str(ctx["output_name"]),
                respawn=None,
                batch_size=int(ctx.get("batch_size") or 0),
            )
        )
        logger.info("So-VITS-SVC training resumed: task_id=%s", self._task_id)
        return self._task_id

    async def stop_training(self, *, preserve_resume: bool = False):
        """停止训练（terminate→kill）；所有出口复位句柄。

        preserve_resume=True：应用关闭/暂停兜底场景，状态落盘为 paused 而非
        stopped，下次启动可从最近检查点继续训练。
        """
        ctx = dict(self._active_ctx) if self._active_ctx else None
        if self._process and self._process.returncode is None:
            # 跨平台一致：先 terminate 给一个优雅退出窗口，再 wait，超时再 kill。
            try:
                self._process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    self._process.wait(), timeout=_TRAIN_STOP_WAIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Training process did not exit after terminate; killing (pid={self._process.pid})"
                )
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(
                        self._process.wait(), timeout=_TRAIN_STOP_WAIT_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"Training process still alive after kill (pid={self._process.pid})"
                    )
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            # join 已取消的监控任务：asyncio.wait 不透传被等待任务自身的
            # CancelledError，任务取消后此处正常返回，后续清理必达。
            done, _ = await asyncio.wait([self._monitor_task])
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc:
                    logger.debug(f"训练监控任务异常退出: {exc}")
        self._process = None
        self._monitor_task = None
        self._pause_requested = False
        if ctx:
            log_dir = Path(str(ctx.get("log_dir") or ""))
            epoch = _parse_last_epoch_from_file(log_dir / "train.log")
            self._persist_train_state(
                "paused" if preserve_resume else "stopped", ctx=ctx, epoch=epoch
            )
        if not preserve_resume:
            self._active_ctx = None
        logger.info("So-VITS-SVC training stopped (preserve_resume=%s)", preserve_resume)

    def export_model(
        self, model_name: str, dest_dir: str, checkpoint: str | None = None
    ) -> dict:
        """导出模型到 <dest_dir>/<模型名>；checkpoint 指定时仅导出该 G 检查点 + 配置。"""
        from modelstation.services.model_export import export_model_dir

        return export_model_dir(self._output_dir, model_name, dest_dir, checkpoint)

    def list_models(self) -> list[dict]:
        models = []
        if self._output_dir.exists():
            for d in self._output_dir.iterdir():
                if d.is_dir():
                    g_files = sorted(d.glob("G_*.pth"), key=lambda p: p.stat().st_mtime)
                    d_files = sorted(d.glob("D_*.pth"), key=lambda p: p.stat().st_mtime)
                    if g_files or d_files:
                        models.append({
                            "name": d.name,
                            "path": str(d),
                            "created": d.stat().st_mtime,
                            "g_model": str(g_files[-1]) if g_files else None,
                            "d_model": str(d_files[-1]) if d_files else None,
                        })
        models.sort(key=lambda m: m["created"], reverse=True)
        return models
