"""
一键成模（提示词/参考音频 → 自动全流程 → 模型文件）测试

change-id: add-one-click-voice-model-pipeline（spec Task 1/2/3/4）。

覆盖：
- POST /api/pipeline/reference：合法音频上传落 data/input 白名单根、非音频 400、
  文件名剥离路径成分（防穿越）；
- POST /api/pipeline/run + GET /run/{run_id}：提示词全流程至 completed（产物回填）、
  上传参考流程（reference=skipped）、二选一校验 400、model_type 缺失 422、
  运行互斥 409、未知 run 404；
- 训练互斥传导：sovits 互斥占用 / melotts trainer 抛 TrainingInProgressError →
  run failed 且 error 含 409 语义；
- Loss 判停：sovits / melotts 判停循环命中 converged → stop_training →
  fitting_status 回填 + 检查点产物兜底收集；
- LossJudge 单测：解析索引（按引擎实码核对）+ converged / overfit_risk /
  max_epochs / improving（min_epochs 内不判停）。

运行时与训练全 mock：数据集 builder 注入 fake runtime client（写伪 wav），
两 trainer 替换为 fake（立即完成 / 等待判停两模式），不触发真实推理/训练/GPU。
数据集目录经 patch security_utils._MS_ROOT/_TRAINING_DATA_ROOT 隔离到 tmp_path。
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import modelstation.config as config_module
import modelstation.services.auto_model_pipeline as ap_module
import modelstation.services.dataset_builder as db_module
import modelstation.services.security_utils as sec_module
from modelstation.config import load_config
from modelstation.main import create_app
from modelstation.services.auto_model_pipeline import (
    DEFAULT_CORPUS_TEXTS,
    MODEL_TYPE_MELOTTS,
    MODEL_TYPE_SOVITS,
    STAGE_IDS,
    AutoModelPipelineService,
    LossJudge,
    PipelineRunConflictError,
    parse_melotts_train_log,
    parse_sovits_train_log,
)
from modelstation.services.dataset_builder import DatasetBuilderService
from modelstation.services.melotts_trainer import TrainingInProgressError
from modelstation.services.training_mutex import (
    TRAINING_MELOTTS,
    TRAINING_SOVITS_SVC,
    current_training,
    reset_registry,
    try_begin_training,
)


# ---------------------------------------------------------------------------
# 测试基建
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_data_root(monkeypatch, tmp_path):
    """隔离训练数据根到 tmp_path，并创建 input/模型产物/引擎目录骨架。"""
    root = tmp_path / "data" / "training"
    root.mkdir(parents=True)
    for rel in (
        "data/input",
        "data/models/sovits_svc",
        "data/models/melotts",
        "engines/so-vits-svc-4.1-Stable",
        "engines/MeloTTS",
    ):
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sec_module, "_MS_ROOT", tmp_path)
    monkeypatch.setattr(sec_module, "_TRAINING_DATA_ROOT", root.resolve())
    return tmp_path


@pytest.fixture
def pipeline_settings(isolated_data_root, monkeypatch):
    """注入指向 tmp 的完整 settings（sovits/melotts 段；训练器与判停日志路径一致）。"""
    tmp = isolated_data_root
    settings = load_config(
        {
            "sovits_svc": {
                "input_dir": str(tmp / "data" / "input"),
                "training_data_dir": str(tmp / "data" / "training" / "sovits_svc"),
                "models_dir": str(tmp / "data" / "models" / "sovits_svc"),
                "so_vits_svc_dir": str(tmp / "engines" / "so-vits-svc-4.1-Stable"),
            },
            "melotts": {
                "engine_dir": str(tmp / "engines" / "MeloTTS"),
                "training_data_dir": str(tmp / "data" / "training" / "melotts"),
                "models_dir": str(tmp / "data" / "models" / "melotts"),
            },
        }
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    return settings


class _FakeRuntimeClient:
    """mock RuntimeTTSClient：记录调用并写伪 WAV（同 test_dataset_engines 模式）"""

    def __init__(self):
        self.voicedesign_calls: list[tuple] = []
        self.cosyvoice_calls: list[tuple] = []

    async def synthesize_voicedesign(self, text, voice_description, output_path):
        self.voicedesign_calls.append((text, voice_description))
        Path(output_path).write_bytes(b"RIFF-rt-" + text.encode("utf-8"))
        return Path(output_path)

    async def synthesize_cosyvoice_zero(self, text, ref_audio_path, ref_text, output_path):
        self.cosyvoice_calls.append((text, str(ref_audio_path), ref_text))
        Path(output_path).write_bytes(b"RIFF-rt-" + text.encode("utf-8"))
        return Path(output_path)


@pytest.fixture
def patched_builder(isolated_data_root, pipeline_settings, monkeypatch):
    """模块级 builder 单例替换为全 mock 注入实例（fake runtime client 写伪 wav）。"""
    rt_fake = _FakeRuntimeClient()
    service = DatasetBuilderService(
        settings=pipeline_settings,
        client_factory=lambda: None,  # 一键成模不使用 voxcpm 引擎
        runtime_client_factory=lambda engine: rt_fake,
    )
    monkeypatch.setattr(db_module, "_builder_instance", service)
    return service, rt_fake


class FakeSovitsTrainer:
    """sovits trainer 桩：三模式（complete=自然完成 / wait_stop=等待判停 / blocked=阻塞）"""

    def __init__(self, engine_dir: Path, models_dir: Path, mode: str = "complete"):
        self.engine_dir = engine_dir
        self.models_dir = models_dir
        self.mode = mode
        self.preprocess_calls: list[tuple] = []
        self.start_calls: list[dict] = []
        self.stop_calls = 0
        self.collect_calls: list[str] = []
        self._cb = None
        self._release = asyncio.Event()

    def _write_products(self, output_name: str) -> Path:
        out = self.models_dir / output_name
        out.mkdir(parents=True, exist_ok=True)
        (out / "G_100.pth").write_bytes(b"fake-g")
        (out / "D_100.pth").write_bytes(b"fake-d")
        (out / "config.json").write_text("{}", encoding="utf-8")
        return out

    def _write_train_log(self, output_name: str, kind: str) -> None:
        """写伪 train.log（Losses 行格式按 sovits train.py 实码，mel 项在 index 3）。"""
        log_dir = self.engine_dir / "logs" / output_name
        log_dir.mkdir(parents=True, exist_ok=True)
        losses = [1.0, 0.9, 0.8, 0.7] if kind == "improving" else [1.0] * 6
        lines = []
        for i, v in enumerate(losses, start=1):
            lines.append(f"2026-10-17 10:00:00\t{output_name}\tINFO\tTrain Epoch: {i} [ 50%]")
            lines.append(
                f"2026-10-17 10:00:00\t{output_name}\tINFO\t"
                f"Losses: [10.0, 5.0, 2.0, {v}, 0.5], step: {i * 10}, "
                "lr: 0.0001, reference_loss: 17.5"
            )
        (log_dir / "train.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    async def preprocess(self, training_data_dir, speaker_name):
        self.preprocess_calls.append((str(training_data_dir), speaker_name))
        return {
            "resample": {"success": True},
            "preprocess_flist_config": {"success": True},
            "preprocess_hubert_f0": {"success": True},
        }

    async def start_training(self, *, epochs, output_name, speaker_name=None,
                             progress_callback=None, eval_interval=None, **kwargs):
        self.start_calls.append(
            {
                "epochs": epochs,
                "output_name": output_name,
                "speaker_name": speaker_name,
                "eval_interval": eval_interval,
            }
        )
        self._cb = progress_callback
        asyncio.get_running_loop().create_task(self._simulate(int(epochs), output_name))
        return "tid-fake-sovits"

    async def _simulate(self, epochs: int, output_name: str) -> None:
        if self.mode == "wait_stop":
            # 写收敛日志供判停器读取，随后挂起直至 stop_training（不触发终态回调，
            # 与真实 monitor 被 stop 取消的行为一致）
            self._write_train_log(output_name, "converged")
            for ep in (1, 2, 3):
                if self._cb:
                    self._cb(progress=ep / epochs, epoch=ep, total_epochs=epochs)
            await self._release.wait()
            return
        if self.mode == "blocked":
            await self._release.wait()
            if self._cb:
                self._cb(progress=1.0, epoch=3, total_epochs=epochs,
                         status="completed", message="released")
            return
        self._write_train_log(output_name, "improving")
        for ep in (1, 2, 3):
            if self._cb:
                self._cb(progress=ep / epochs, epoch=ep, total_epochs=epochs)
        self._write_products(output_name)  # 真实链路由 monitor 收集产物
        if self._cb:
            self._cb(progress=1.0, epoch=3, total_epochs=epochs,
                     status="completed", message="训练完成")

    async def stop_training(self):
        self.stop_calls += 1
        self._release.set()

    def _collect_outputs(self, output_name: str) -> Path:
        self.collect_calls.append(output_name)
        return self._write_products(output_name)


class FakeMelottsTrainer:
    """melotts trainer 桩：无 progress_callback，进度经 status dict 轮询（实码语义）"""

    def __init__(self, engine_dir: Path, models_dir: Path, mode: str = "complete"):
        self.engine_dir = engine_dir
        self.models_dir = models_dir
        self.mode = mode
        self.start_calls: list[dict] = []
        self.stop_calls = 0
        self.collect_calls: list[str] = []
        self.raise_conflict = False
        self._stop_requested = False
        self._resume_event = asyncio.Event()
        self.status = {
            "task_id": None, "status": "idle", "progress": 0.0,
            "epoch": 0, "total_epochs": 0, "message": "",
        }

    def get_status(self) -> dict:
        return dict(self.status)

    def _write_products(self, output_name: str) -> Path:
        out = self.models_dir / output_name
        out.mkdir(parents=True, exist_ok=True)
        (out / "G_100.pth").write_bytes(b"fake-g")
        (out / "D_100.pth").write_bytes(b"fake-d")
        (out / "DUR_100.pth").write_bytes(b"fake-dur")
        (out / "config.json").write_text("{}", encoding="utf-8")
        return out

    def _write_train_log(self, output_name: str, kind: str) -> None:
        """写伪 train.log（损失列表行格式按 melo train.py 实码，末两位 step/lr）。"""
        log_dir = self.engine_dir / "melo" / "logs" / output_name
        log_dir.mkdir(parents=True, exist_ok=True)
        totals = [4.0, 3.5, 3.0, 2.5] if kind == "improving" else [3.95] * 6
        lines = []
        step = 0
        for i, total in enumerate(totals, start=1):
            # [disc, gen, fm, mel, dur, kl] → 生成器总损失 = gen+fm+mel+dur+kl = total
            components = [5.0, total - 1.95, 0.5, 1.0, 0.3, 0.15]
            step += 10
            lines.append(
                f"2026-10-17 10:00:00\tmelo\tINFO\t{repr(components + [step, 0.0003])}"
            )
            lines.append(f"2026-10-17 10:00:00\tmelo\tINFO\t====> Epoch: {i}")
        (log_dir / "train.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    async def start_training(self, *, epochs, output_name=None, eval_interval=None, **kwargs):
        if self.raise_conflict:
            raise TrainingInProgressError(
                "训练任务正在进行中（类型: sovits_svc, task_id: tid-external）",
                current={"owner_type": TRAINING_SOVITS_SVC, "task_id": "tid-external"},
            )
        self.start_calls.append(
            {"epochs": epochs, "output_name": output_name, "eval_interval": eval_interval}
        )
        self.status.update(
            {"task_id": "tid-fake-melotts", "status": "running",
             "total_epochs": int(epochs), "message": "训练进行中"}
        )
        asyncio.get_running_loop().create_task(self._simulate(int(epochs), output_name))
        return "tid-fake-melotts"

    async def _simulate(self, epochs: int, output_name: str) -> None:
        if self.mode == "pause_then_complete":
            # 先写收敛日志再置 paused：验证暂停态不参与 Loss 判停（否则会误写停止哨兵）
            self._write_train_log(output_name, "converged")
            self.status.update(
                {"epoch": 3, "progress": 3 / epochs, "status": "paused",
                 "message": "训练已暂停（epoch 3），检查点已保存"}
            )
            await self._resume_event.wait()
            self._write_products(output_name)
            self.status.update(
                {"status": "completed", "progress": 1.0, "message": "训练完成"}
            )
            return
        if self.mode == "wait_stop":
            self._write_train_log(output_name, "converged")
            for ep in (1, 2, 3):
                self.status.update({"epoch": ep, "progress": ep / epochs})
                await asyncio.sleep(0)
            while not self._stop_requested:
                await asyncio.sleep(0.01)
            return
        self._write_train_log(output_name, "improving")
        for ep in (1, 2, 3):
            self.status.update({"epoch": ep, "progress": ep / epochs})
            await asyncio.sleep(0)
        self._write_products(output_name)
        self.status.update({"status": "completed", "progress": 1.0, "message": "训练完成"})

    async def stop_training(self):
        self.stop_calls += 1
        self._stop_requested = True
        self.status.update({"status": "stopped", "message": "训练已停止"})

    def _collect_outputs(self, output_name: str) -> Path:
        self.collect_calls.append(output_name)
        return self._write_products(output_name)


@pytest.fixture
def fake_trainers(pipeline_settings, monkeypatch):
    """两 trainer 的 getter 替换为 fake 实例（complete 模式，测试可改 mode）。"""
    settings = pipeline_settings
    sovits_fake = FakeSovitsTrainer(
        Path(settings.sovits_svc.so_vits_svc_dir), Path(settings.sovits_svc.models_dir)
    )
    melotts_fake = FakeMelottsTrainer(
        Path(settings.melotts.engine_dir), Path(settings.melotts.models_dir)
    )
    from modelstation.api import sovits_svc as sovits_api
    from modelstation.services import melotts_trainer as mt_module

    monkeypatch.setattr(sovits_api, "get_sovits_trainer", lambda: sovits_fake)
    monkeypatch.setattr(mt_module, "get_trainer", lambda: melotts_fake)
    monkeypatch.setattr(mt_module, "get_train_status", melotts_fake.get_status)
    return sovits_fake, melotts_fake


@pytest.fixture(autouse=True)
def _clean_pipeline_state(monkeypatch):
    """运行/训练互斥与单例状态清理；判停轮询间隔缩短保证测试速度。"""
    from modelstation.services import melotts_trainer as mt_module

    reset_registry()
    mt_module.reset_train_status()
    ap_module.reset_auto_model_pipeline()
    monkeypatch.setattr(ap_module, "_TRAIN_POLL_SECONDS", 0.01)
    yield
    reset_registry()
    mt_module.reset_train_status()
    ap_module.reset_auto_model_pipeline()


@pytest_asyncio.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _wait_run(client: AsyncClient, run_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        resp = await client.get(f"/api/pipeline/run/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        if time.monotonic() > deadline:
            raise TimeoutError(f"run {run_id} did not finish: {body}")
        await asyncio.sleep(0.01)


def _stage_map(run: dict) -> dict:
    return {s["id"]: s["status"] for s in run["stages"]}


async def _wait_preview(client: AsyncClient, task_id: str, timeout: float = 10.0) -> dict:
    """等待试听任务进入 awaiting_review（或提前失败）。"""
    deadline = time.monotonic() + timeout
    while True:
        resp = await client.get(f"/api/datasets/batch-generate/{task_id}")
        assert resp.status_code == 200
        task = resp.json()
        if task["status"] in ("awaiting_review", "failed"):
            return task
        if time.monotonic() > deadline:
            raise TimeoutError(f"preview task {task_id} did not settle: {task}")
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# POST /api/pipeline/reference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_upload_success(client, isolated_data_root, pipeline_settings):
    """合法音频上传：落 data/input 白名单根，返回受控绝对路径"""
    resp = await client.post(
        "/api/pipeline/reference", files={"file": ("ref.wav", b"RIFF-fake", "audio/wav")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    path = Path(body["ref_audio_path"])
    assert path.is_file()
    assert path.parent == (isolated_data_root / "data" / "input").resolve()
    assert path.name == "ref.wav"


@pytest.mark.asyncio
async def test_reference_upload_invalid_ext_400(client, isolated_data_root, pipeline_settings):
    """非音频扩展名 → 400 且不落盘"""
    resp = await client.post(
        "/api/pipeline/reference", files={"file": ("note.txt", b"hello", "text/plain")}
    )
    assert resp.status_code == 400
    assert "Unsupported audio file type" in resp.json()["detail"]
    assert not (isolated_data_root / "data" / "input" / "note.txt").exists()


@pytest.mark.asyncio
async def test_reference_upload_strips_path_components(
    client, isolated_data_root, pipeline_settings
):
    """文件名剥离路径成分（../ 与反斜杠均按 basename 落盘，防穿越）"""
    resp = await client.post(
        "/api/pipeline/reference",
        files={"file": ("../../evil.wav", b"RIFF-evil", "audio/wav")},
    )
    assert resp.status_code == 200
    path = Path(resp.json()["ref_audio_path"])
    assert path.parent == (isolated_data_root / "data" / "input").resolve()
    assert path.name == "evil.wav"
    assert ".." not in str(path)
    assert path.is_file()


# ---------------------------------------------------------------------------
# 默认语料文本集
# ---------------------------------------------------------------------------


def test_default_corpus_texts():
    """内置默认语料 ≥16 条中文短句且非空（spec Task 2.1）"""
    assert len(DEFAULT_CORPUS_TEXTS) >= 16
    for item in DEFAULT_CORPUS_TEXTS:
        assert isinstance(item, dict) and item["text"].strip()


def test_default_corpus_texts_scales_with_target():
    """默认语料池按目标片数放大（≈2.8 句/片，唯一句子）"""
    from modelstation.services.auto_model_pipeline import _default_corpus_texts

    pool = _default_corpus_texts(300)
    assert len(pool) >= int(300 * 2.8)
    assert len({item["text"] for item in pool}) == len(pool)


# ---------------------------------------------------------------------------
# 一键成模全流程
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_pipeline_flow_sovits_completed(
    client, patched_builder, fake_trainers, isolated_data_root, pipeline_settings
):
    """提示词全流程：run completed、DS-001、产物回填、阶段全 completed、互斥释放"""
    resp = await client.post(
        "/api/pipeline/run", json={"model_type": "sovits", "voice_description": "温柔的青年女声"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    run_id = body["run_id"]
    assert {s["id"] for s in body["stages"]} == set(STAGE_IDS)
    assert all(s["status"] == "pending" for s in body["stages"])

    run = await _wait_run(client, run_id)
    assert run["status"] == "completed", run["error"]
    assert run["model_type"] == "sovits"
    assert run["dataset_id"] == "DS-001"
    assert _stage_map(run) == {sid: "completed" for sid in STAGE_IDS}
    assert run["result"]["model_name"] == "auto_DS-001_sovits"
    assert Path(run["result"]["model_dir"]) == (
        Path(pipeline_settings.sovits_svc.models_dir) / "auto_DS-001_sovits"
    )
    assert "G_100.pth" in run["result"]["files"]
    assert run["corpus"]["done"] == run["corpus"]["total"] == int(50 * 2.8) + 1
    assert run["error"] is None
    # 训练互斥已释放
    assert current_training() is None

    # 运行时调用核对：qwen3 参考音频 + cosyvoice 批量克隆（成片模式句子池扩展 2×）
    _service, rt = patched_builder
    assert len(rt.voicedesign_calls) == 1
    assert rt.voicedesign_calls[0][1] == "温柔的青年女声"
    assert len(rt.cosyvoice_calls) == int(50 * 2.8)

    # trainer 调用核对：preprocess/training 用同一 speaker（DS 目录名）、缺省 epochs（10000）
    sovits_fake, _ = fake_trainers
    assert sovits_fake.preprocess_calls[0][1] == "DS-001"
    assert sovits_fake.start_calls[0]["speaker_name"] == "DS-001"
    assert sovits_fake.start_calls[0]["epochs"] == 10000
    # 检查点节拍按总步数规划（10000 epochs × ceil(21/16)步/epoch=20000 步 → /10 封顶 1000）
    assert sovits_fake.start_calls[0]["eval_interval"] == 1000
    assert sovits_fake.collect_calls == []  # 自然完成路径由 trainer 自行收集


@pytest.mark.asyncio
async def test_preview_confirmed_dataset_pipeline_flow_sovits_completed(
    client, patched_builder, fake_trainers, isolated_data_root, pipeline_settings
):
    """试听确认续跑：preview 只生成参考音频 → dataset_id 续跑跳过 reference 阶段"""
    pv = await client.post(
        "/api/datasets/preview",
        json={"voice_description": "温柔的青年女声", "ref_text": "参考播报内容"},
    )
    assert pv.status_code == 200
    assert pv.json()["total"] == 1
    preview = await _wait_preview(client, pv.json()["task_id"])
    assert preview["status"] == "awaiting_review"
    assert preview["phase"] == "reference"
    assert preview["done"] == preview["total"] == 1
    assert preview["ref_file"]
    dataset_id = preview["dataset_id"]
    assert dataset_id == "DS-001"

    resp = await client.post(
        "/api/pipeline/run", json={"model_type": "sovits", "dataset_id": dataset_id}
    )
    assert resp.status_code == 200
    run = await _wait_run(client, resp.json()["run_id"])
    assert run["status"] == "completed", run["error"]
    assert run["dataset_id"] == dataset_id
    stages = _stage_map(run)
    assert stages["reference"] == "completed"
    assert "试听确认" in run["stages"][0]["detail"]
    assert all(stages[sid] == "completed" for sid in STAGE_IDS if sid != "reference")
    assert run["result"]["model_name"] == "auto_DS-001_sovits"
    assert current_training() is None

    # 参考音频只生成一次（preview 阶段），续跑复用该文件
    _service, rt = patched_builder
    assert len(rt.voicedesign_calls) == 1
    assert rt.voicedesign_calls[0] == ("参考播报内容", "温柔的青年女声")
    assert len(rt.cosyvoice_calls) == int(50 * 2.8)
    assert Path(rt.cosyvoice_calls[0][1]).name == preview["ref_file"]
    # ref_text 透传（来自 manifest 参考条目）
    assert rt.cosyvoice_calls[0][2] == "参考播报内容"


@pytest.mark.asyncio
async def test_preview_dataset_missing_ref_audio_400(
    client, patched_builder, pipeline_settings
):
    """dataset_id 未登记 / 缺少参考音频 → 400（必须先预览试听确认）"""
    resp = await client.post(
        "/api/pipeline/run", json={"model_type": "sovits", "dataset_id": "DS-999"}
    )
    assert resp.status_code == 400
    assert "数据集编号不存在" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_upload_ref_pipeline_flow_melotts_completed(
    client, patched_builder, fake_trainers, isolated_data_root
):
    """上传参考流程：reference=skipped，其余阶段 completed，模型名 auto_DS-001_melotts"""
    up = await client.post(
        "/api/pipeline/reference", files={"file": ("ref.wav", b"RIFF-fake", "audio/wav")}
    )
    assert up.status_code == 200
    ref_path = up.json()["ref_audio_path"]

    resp = await client.post(
        "/api/pipeline/run",
        json={"model_type": "melotts", "ref_audio_path": ref_path, "max_epochs": 5},
    )
    assert resp.status_code == 200
    run = await _wait_run(client, resp.json()["run_id"])
    assert run["status"] == "completed", run["error"]
    assert run["model_type"] == "melotts"
    assert run["dataset_id"] == "DS-001"
    stages = _stage_map(run)
    assert stages["reference"] == "skipped"
    assert all(stages[sid] == "completed" for sid in STAGE_IDS if sid != "reference")
    assert run["result"]["model_name"] == "auto_DS-001_melotts"
    assert "G_100.pth" in run["result"]["files"]
    # 成片模式（API 默认 target 50）：合成 2×50 条短句（拼片失败时计数保持句子口径）
    assert run["corpus"]["done"] == run["corpus"]["total"] == int(50 * 2.8)
    assert current_training() is None

    # cosyvoice 以上传参考克隆；melotts trainer 收到 epochs=5，eval_interval 不再强制调小
    _service, rt = patched_builder
    assert len(rt.cosyvoice_calls) == int(50 * 2.8)
    assert Path(rt.cosyvoice_calls[0][1]) == Path(ref_path).resolve()
    _, melotts_fake = fake_trainers
    assert melotts_fake.start_calls[0]["epochs"] == 5
    # 检查点节拍按总步数规划（5 epochs × ceil(140/16)=9 步/epoch = 45 步 → /10 向上取整 = 5）
    assert melotts_fake.start_calls[0]["eval_interval"] == 5
    assert melotts_fake.collect_calls == []  # 自然完成路径由 trainer 自行收集


# ---------------------------------------------------------------------------
# 自带数据集导入（dataset_dir，仅 sovits）
# ---------------------------------------------------------------------------


def _make_user_audio_dir(tmp_path: Path) -> Path:
    """构造用户数据集源目录：4 条混合采样率/声道/格式音频（soundfile 实编码）。"""
    import numpy as np
    import soundfile as sf

    src = tmp_path / "user_audio"
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    sf.write(str(src / "a.wav"), rng.standard_normal(4410).astype("float32") * 0.1, 44100)
    stereo = np.stack(
        [rng.standard_normal(2205), rng.standard_normal(2205)], axis=1
    ).astype("float32") * 0.1
    sf.write(str(src / "b.wav"), stereo, 22050, subtype="PCM_16")
    sf.write(
        str(src / "c.flac"),
        rng.standard_normal(8000).astype("float32") * 0.1,
        32000,
        format="FLAC",
    )
    sub = src / "nested"
    sub.mkdir(exist_ok=True)
    sf.write(str(sub / "d.wav"), rng.standard_normal(16000).astype("float32") * 0.1, 16000)
    return src


@pytest.mark.asyncio
async def test_dataset_import_flow_sovits_completed(
    client, patched_builder, fake_trainers, isolated_data_root
):
    """自带数据集流程：reference=skipped、导入转码 44.1k 单声道、训练按导入条数规划"""
    import soundfile as sf

    src = _make_user_audio_dir(isolated_data_root)
    resp = await client.post(
        "/api/pipeline/run",
        json={"model_type": "sovits", "dataset_dir": str(src), "max_epochs": 5},
    )
    assert resp.status_code == 200
    # 导入含 soundfile/scipy 首次加载，放宽等待（全量首跑冷加载可能明显偏慢）
    run = await _wait_run(client, resp.json()["run_id"], timeout=60.0)
    assert run["status"] == "completed", run["error"]
    assert run["model_type"] == "sovits"
    assert run["dataset_id"] == "DS-001"
    stages = _stage_map(run)
    assert stages["reference"] == "skipped"
    assert all(stages[sid] == "completed" for sid in STAGE_IDS if sid != "reference")
    assert run["result"]["model_name"] == "auto_DS-001_sovits"
    assert run["corpus"]["done"] == run["corpus"]["total"] == 4
    assert current_training() is None

    # 导入转码：raw/DS-001/ 下 4 个 wav，44.1kHz 单声道（22.05k 立体声/16k/32k flac 均已统一）
    ds_dir = isolated_data_root / "data" / "training" / "sovits_svc" / "raw" / "DS-001"
    wavs = sorted(ds_dir.glob("*.wav"))
    assert len(wavs) == 4
    for wav in wavs:
        info = sf.info(str(wav))
        assert info.samplerate == 44100
        assert info.channels == 1

    # 语料生成全程未被调用；存档节拍按导入条数规划（5 epochs × 1 步 → /10 = 1）
    _service, rt = patched_builder
    assert rt.cosyvoice_calls == []
    assert rt.voicedesign_calls == []
    sovits_fake, _ = fake_trainers
    assert sovits_fake.start_calls[0]["epochs"] == 5
    assert sovits_fake.start_calls[0]["eval_interval"] == 1


@pytest.mark.asyncio
async def test_dataset_import_melotts_400(client, isolated_data_root, pipeline_settings):
    """dataset_dir 与 melotts 组合 → 400（MeloTTS 需要音频-文本对）"""
    src = _make_user_audio_dir(isolated_data_root)
    resp = await client.post(
        "/api/pipeline/run",
        json={"model_type": "melotts", "dataset_dir": str(src)},
    )
    assert resp.status_code == 400
    assert "仅支持 sovits" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_dataset_dir_validation_400(client, isolated_data_root, pipeline_settings):
    """dataset_dir 不存在 / 无音频 → 400"""
    resp = await client.post(
        "/api/pipeline/run",
        json={"model_type": "sovits", "dataset_dir": str(isolated_data_root / "nope")},
    )
    assert resp.status_code == 400
    assert "不存在或不是目录" in resp.json()["detail"]

    empty = isolated_data_root / "empty_ds"
    empty.mkdir()
    resp = await client.post(
        "/api/pipeline/run",
        json={"model_type": "sovits", "dataset_dir": str(empty)},
    )
    assert resp.status_code == 400
    assert "音频不足" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_run_dataset_and_prompt_conflict_400(
    client, patched_builder, fake_trainers, isolated_data_root
):
    """voice_description 与 dataset_dir 同时提供 → 400（四选一）"""
    src = _make_user_audio_dir(isolated_data_root)
    resp = await client.post(
        "/api/pipeline/run",
        json={
            "model_type": "sovits",
            "voice_description": "描述",
            "dataset_dir": str(src),
        },
    )
    assert resp.status_code == 400
    assert "四选一" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_prompt_pipeline_corpus_clip_assembly(
    client, patched_builder, fake_trainers, isolated_data_root, monkeypatch
):
    """语料成片：corpus_target_clips 生效 → 句子池截断 2.8×目标 → 拼片 → 计数收敛"""
    import json

    import numpy as np
    import soundfile as sf

    _service, rt = patched_builder

    async def real_wav_cosy(text, ref_audio_path, ref_text, output_path):
        rng = np.random.default_rng(len(str(text)))
        sf.write(
            str(output_path),
            (rng.standard_normal(1102) * 0.1).astype("float32"),
            22050,
            subtype="PCM_16",
        )
        return Path(output_path)

    monkeypatch.setattr(rt, "synthesize_cosyvoice_zero", real_wav_cosy)

    resp = await client.post(
        "/api/pipeline/run",
        json={
            "model_type": "melotts",
            "voice_description": "温柔的青年女声",
            "max_epochs": 5,
            "corpus_target_clips": 2,
        },
    )
    assert resp.status_code == 200
    run = await _wait_run(client, resp.json()["run_id"])
    assert run["status"] == "completed", run["error"]
    # 句子池截断 2.8×2≈5 条 → 全为短句拼成 1 片 → 计数收敛（reference + 1 片）
    assert run["corpus"]["done"] == run["corpus"]["total"] == 2

    ds_dir = isolated_data_root / "data" / "training" / "sovits_svc" / "raw" / "DS-001"
    manifest = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    clips = [e for e in manifest["entries"] if e.get("role") == "corpus_clip"]
    assert len(clips) == 1
    assert clips[0]["file"] == "clip_0001.wav"
    # 文本精确拼接（5 句顺序拼接），原逐句文件被成片替换（目录仅剩 reference + clip）
    assert clips[0]["text"] == "".join(t["text"] for t in DEFAULT_CORPUS_TEXTS[:5])
    wavs = [p.name for p in ds_dir.glob("*.wav")]
    assert len(wavs) == 2
    assert "clip_0001.wav" in wavs


@pytest.mark.asyncio
async def test_sovits_judge_stop_converged(
    client, patched_builder, fake_trainers, monkeypatch
):
    """sovits Loss 判停：收敛日志 → 写优雅停止哨兵 → 超时兜底强杀 → fitting=converged"""
    import modelstation.services.auto_model_pipeline as amp

    monkeypatch.setattr(amp, "_TRAIN_POLL_SECONDS", 0.05)
    monkeypatch.setattr(amp, "_STOP_FALLBACK_SECONDS", 0.2)
    sovits_fake, _ = fake_trainers
    sovits_fake.mode = "wait_stop"

    resp = await client.post(
        "/api/pipeline/run", json={"model_type": "sovits", "voice_description": "低沉男声"}
    )
    assert resp.status_code == 200
    run = await _wait_run(client, resp.json()["run_id"])
    assert run["status"] == "completed", run["error"]
    assert run["training"]["fitting_status"] == "converged"
    assert run["training"]["stop_reason"]
    assert sovits_fake.stop_calls == 1
    # 优雅停止哨兵已写入模型目录（真实引擎在下一个 epoch 边界存盘退出）
    sentinel = sovits_fake.engine_dir / "logs" / "auto_DS-001_sovits" / ".stop_requested"
    assert sentinel.is_file()
    assert _stage_map(run) == {sid: "completed" for sid in STAGE_IDS}
    assert "G_100.pth" in run["result"]["files"]
    # 判停路径：产物由 collect 阶段兜底收集
    assert sovits_fake.collect_calls == ["auto_DS-001_sovits"]
    assert current_training() is None


@pytest.mark.asyncio
async def test_melotts_judge_stop_converged(client, patched_builder, fake_trainers, monkeypatch):
    """melotts Loss 判停：收敛日志 → 写优雅停止哨兵 → 超时兜底强杀 → fitting=converged"""
    import modelstation.services.auto_model_pipeline as amp

    monkeypatch.setattr(amp, "_TRAIN_POLL_SECONDS", 0.05)
    monkeypatch.setattr(amp, "_STOP_FALLBACK_SECONDS", 0.2)
    _, melotts_fake = fake_trainers
    melotts_fake.mode = "wait_stop"

    resp = await client.post(
        "/api/pipeline/run", json={"model_type": "melotts", "voice_description": "清亮女声"}
    )
    assert resp.status_code == 200
    run = await _wait_run(client, resp.json()["run_id"])
    assert run["status"] == "completed", run["error"]
    assert run["training"]["fitting_status"] == "converged"
    assert run["training"]["stop_reason"]
    assert melotts_fake.stop_calls == 1
    sentinel = (
        melotts_fake.engine_dir / "melo" / "logs" / "auto_DS-001_melotts" / ".stop_requested"
    )
    assert sentinel.is_file()
    assert "G_100.pth" in run["result"]["files"]
    assert melotts_fake.collect_calls == ["auto_DS-001_melotts"]
    assert current_training() is None


@pytest.mark.asyncio
async def test_melotts_paused_skips_loss_judgement(
    client, patched_builder, fake_trainers, monkeypatch
):
    """暂停态不参与 Loss 判停：不写停止哨兵/不调 stop_training，恢复后继续至完成"""
    import modelstation.services.auto_model_pipeline as amp

    monkeypatch.setattr(amp, "_TRAIN_POLL_SECONDS", 0.02)
    _, melotts_fake = fake_trainers
    melotts_fake.mode = "pause_then_complete"

    resp = await client.post(
        "/api/pipeline/run", json={"model_type": "melotts", "voice_description": "清亮女声"}
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    # 等待进入暂停态，再让其经历多个轮询周期（收敛日志不应触发判停）
    deadline = time.monotonic() + 10.0
    while melotts_fake.status["status"] != "paused":
        assert time.monotonic() < deadline
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)

    run = (await client.get(f"/api/pipeline/run/{run_id}")).json()
    assert run["status"] == "running"
    assert run["training"]["epoch"] == 3
    assert melotts_fake.stop_calls == 0
    sentinel = (
        melotts_fake.engine_dir / "melo" / "logs" / "auto_DS-001_melotts" / ".stop_requested"
    )
    assert not sentinel.exists()

    melotts_fake._resume_event.set()

    run = await _wait_run(client, run_id)
    assert run["status"] == "completed", run["error"]
    assert "G_100.pth" in run["result"]["files"]


# ---------------------------------------------------------------------------
# 输入校验 / 运行互斥 / 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_either_or_validation_400(client, isolated_data_root, pipeline_settings):
    """输入四选一校验：同时提供多项 / 全缺省 → 400，不创建运行"""
    both = await client.post(
        "/api/pipeline/run",
        json={
            "model_type": "sovits",
            "voice_description": "描述",
            "ref_audio_path": str(isolated_data_root / "data" / "input" / "ref.wav"),
        },
    )
    assert both.status_code == 400
    assert "四选一" in both.json()["detail"]

    neither = await client.post("/api/pipeline/run", json={"model_type": "sovits"})
    assert neither.status_code == 400
    assert "四选一" in neither.json()["detail"]

    service = ap_module.get_auto_model_pipeline()
    assert service._runs == {}


@pytest.mark.asyncio
async def test_run_ref_audio_validation_400(
    client, isolated_data_root, pipeline_settings
):
    """参考音频不在 data/input 白名单根 / 文件不存在 → 400"""
    outside = isolated_data_root / "outside.wav"
    outside.write_bytes(b"RIFF-evil")
    resp = await client.post(
        "/api/pipeline/run",
        json={"model_type": "sovits", "ref_audio_path": str(outside)},
    )
    assert resp.status_code == 400
    assert "must be located under" in resp.json()["detail"]

    ghost = str(isolated_data_root / "data" / "input" / "ghost.wav")
    resp = await client.post(
        "/api/pipeline/run", json={"model_type": "sovits", "ref_audio_path": ghost}
    )
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_run_output_name_validation_400(client):
    """output_name 覆盖值不在白名单（含路径分隔符）→ 400"""
    resp = await client.post(
        "/api/pipeline/run",
        json={
            "model_type": "sovits",
            "voice_description": "描述",
            "output_name": "../evil",
        },
    )
    assert resp.status_code == 400
    assert "Invalid output_name" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_run_model_type_missing_422(client):
    """model_type 缺失/非法 → 422（pydantic Literal 校验）"""
    resp = await client.post("/api/pipeline/run", json={"voice_description": "描述"})
    assert resp.status_code == 422
    resp = await client.post(
        "/api/pipeline/run", json={"model_type": "bert_vits", "voice_description": "描述"}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_run_conflict_409(client, patched_builder, fake_trainers):
    """同一时间仅一个运行：running 存在时重复提交 → 409"""
    sovits_fake, _ = fake_trainers
    sovits_fake.mode = "blocked"  # 阻塞在训练阶段，保证首个 run 保持 running

    first = await client.post(
        "/api/pipeline/run", json={"model_type": "sovits", "voice_description": "第一个"}
    )
    assert first.status_code == 200
    run_id = first.json()["run_id"]

    second = await client.post(
        "/api/pipeline/run", json={"model_type": "sovits", "voice_description": "第二个"}
    )
    assert second.status_code == 409
    assert "已有一键成模运行进行中" in second.json()["detail"]

    # 收尾：放行首个 run 至终态，避免后台任务悬挂
    sovits_fake._release.set()
    run = await _wait_run(client, run_id)
    assert run["status"] == "completed", run["error"]


@pytest.mark.asyncio
async def test_run_rerun_allowed_after_failure(client, patched_builder, fake_trainers):
    """失败运行不占用运行槽位：failed 后可再次提交"""
    sovits_fake, _ = fake_trainers
    ok, _ = try_begin_training(TRAINING_MELOTTS, "tid-external")
    assert ok
    first = await client.post(
        "/api/pipeline/run", json={"model_type": "sovits", "voice_description": "描述"}
    )
    run = await _wait_run(client, first.json()["run_id"])
    assert run["status"] == "failed"
    reset_registry()

    sovits_fake.mode = "blocked"
    second = await client.post(
        "/api/pipeline/run", json={"model_type": "sovits", "voice_description": "描述"}
    )
    assert second.status_code == 200
    sovits_fake._release.set()
    run = await _wait_run(client, second.json()["run_id"])
    assert run["status"] == "completed", run["error"]


@pytest.mark.asyncio
async def test_unknown_run_404(client):
    """未知 run_id → 404"""
    resp = await client.get("/api/pipeline/run/not-exist-run-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 训练互斥传导
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sovits_training_mutex_conflict_run_failed(client, patched_builder, fake_trainers):
    """sovits 训练互斥被外部占用 → run failed 且 error 含 409 语义，互斥不被误清"""
    ok, _ = try_begin_training(TRAINING_MELOTTS, "tid-external-melotts")
    assert ok
    try:
        resp = await client.post(
            "/api/pipeline/run", json={"model_type": "sovits", "voice_description": "描述"}
        )
        assert resp.status_code == 200
        run = await _wait_run(client, resp.json()["run_id"])
        assert run["status"] == "failed"
        assert "409" in run["error"]
        stages = _stage_map(run)
        assert stages["training"] == "failed"
        assert stages["corpus"] == "completed"  # 前序阶段已完成
        # 互斥持有者未被 pipeline 误清
        assert current_training()["task_id"] == "tid-external-melotts"
    finally:
        reset_registry()


@pytest.mark.asyncio
async def test_melotts_training_conflict_run_failed(client, patched_builder, fake_trainers):
    """melotts trainer 抛互斥（TrainingInProgressError）→ run failed 含 409 语义"""
    _, melotts_fake = fake_trainers
    melotts_fake.raise_conflict = True
    ok, _ = try_begin_training(TRAINING_SOVITS_SVC, "tid-external-sovits")
    assert ok
    try:
        resp = await client.post(
            "/api/pipeline/run", json={"model_type": "melotts", "voice_description": "描述"}
        )
        assert resp.status_code == 200
        run = await _wait_run(client, resp.json()["run_id"])
        assert run["status"] == "failed"
        assert "409" in run["error"]
        assert "训练互斥冲突" in run["error"]
        assert _stage_map(run)["training"] == "failed"
    finally:
        reset_registry()


# ---------------------------------------------------------------------------
# LossJudge 单测（解析索引按引擎实码核对）
# ---------------------------------------------------------------------------


def _sovits_log(losses_per_epoch: list[float]) -> str:
    """构造 sovits train.log 行序列（Losses 列表 index 3 为 loss_mel，见 train.py:249）。"""
    lines = []
    for i, v in enumerate(losses_per_epoch, start=1):
        lines.append(f"2026-10-17 10:00:00\tmymodel\tINFO\tTrain Epoch: {i} [ 50%]")
        lines.append(
            f"2026-10-17 10:00:00\tmymodel\tINFO\t"
            f"Losses: [10.0, 5.0, 2.0, {v}, 0.5], step: {i * 10}, "
            "lr: 0.0001, reference_loss: 17.5"
        )
    return "\n".join(lines) + "\n"


def _melotts_log(epoch_totals: list[float]) -> str:
    """构造 melo train.log 行序列（列表末两位为 step/lr，见 train.py:472）。"""
    lines = []
    for i, total in enumerate(epoch_totals, start=1):
        components = [5.0, total - 1.95, 0.5, 1.0, 0.3, 0.15]
        lines.append(
            f"2026-10-17 10:00:00\tmelo\tINFO\t{repr(components + [i * 10, 0.0003])}"
        )
        lines.append(f"2026-10-17 10:00:00\tmelo\tINFO\t====> Epoch: {i}")
    return "\n".join(lines) + "\n"


def test_parse_sovits_mel_loss_index():
    """sovits 主损失取 index 3（loss_mel）。

    索引依据：engines/so-vits-svc-4.1-Stable/train.py:249
    ``losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]`` → index 3 为 mel 项。
    """
    parsed = parse_sovits_train_log(_sovits_log([1.0, 2.0, 3.5]))
    assert parsed == [(1, 1.0), (2, 2.0), (3, 3.5)]


def test_parse_sovits_multi_line_epoch_mean():
    """同 epoch 多条 Losses 行取均值；无 epoch 归属的行跳过"""
    log = (
        "2026-10-17 10:00:00\tm\tINFO\tTrain Epoch: 1 [ 10%]\n"
        "2026-10-17 10:00:01\tm\tINFO\tLosses: [1.0, 2.0, 3.0, 1.0, 5.0], step: 1, lr: 0.1\n"
        "2026-10-17 10:00:02\tm\tINFO\tLosses: [1.0, 2.0, 3.0, 3.0, 5.0], step: 2, lr: 0.1\n"
        "2026-10-17 10:00:03\tm\tINFO\tLosses: [1.0, 2.0, 3.0, 9.0, 5.0], step: 3, lr: 0.1\n"
    )
    assert parse_sovits_train_log(log) == [(1, 13.0 / 3.0)]


def test_parse_melotts_generator_total_and_epoch_boundary():
    """melotts 主损失 = 列表 index 1 起求和（生成器总损失），epoch 边界按收尾行。

    索引依据：engines/MeloTTS/melo/train.py:466
    ``losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]``；
    train.py:452 ``loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl``
    → 生成器总损失 = 除 index 0（discriminator）外求和；
    train.py:472 日志行为 ``[x.item() for x in losses] + [global_step, lr]``
    （末两位剥离）；``====> Epoch: N``（train.py:555）关闭 epoch N 的桶。
    """
    parsed = parse_melotts_train_log(_melotts_log([1.0, 2.0]))
    assert parsed == [(1, pytest.approx(1.0)), (2, pytest.approx(2.0))]


def test_parse_melotts_trailing_bucket_attributed_to_next_epoch():
    """未闭合的训练中 epoch：boundary 之后的损失行归属下一 epoch"""
    components = [5.0, 0.05, 0.5, 1.0, 0.3, 0.15]  # 生成器总损失 = 2.0
    log = (
        f"2026-10-17 10:00:00\tmelo\tINFO\t{repr(components + [10, 0.0003])}\n"
        "2026-10-17 10:00:01\tmelo\tINFO\t====> Epoch: 1\n"
        f"2026-10-17 10:00:02\tmelo\tINFO\t{repr(components + [20, 0.0003])}\n"
    )
    parsed = parse_melotts_train_log(log)
    assert parsed == [(1, pytest.approx(2.0)), (2, pytest.approx(2.0))]


def test_judge_converged():
    """连续 3 个 epoch 相对前窗改善 <1% → converged（超过 min_epochs）"""
    judge = LossJudge(MODEL_TYPE_SOVITS)
    series = [(i, 1.0) for i in range(1, 7)]  # 完全停滞
    status, reason = judge.judge(series, max_epochs=200)
    assert status == "converged"
    assert reason and "改善" in reason


def test_judge_converged_with_tiny_improvements():
    """改善幅度 <1% 视为停滞（历史最优未大幅回升，不误判 overfit_risk）"""
    judge = LossJudge(MODEL_TYPE_SOVITS)
    series = [(1, 1.0), (2, 0.995), (3, 0.99), (4, 0.99), (5, 0.99), (6, 0.99)]
    status, _ = judge.judge(series, max_epochs=200)
    assert status == "converged"


def test_judge_overfit_risk():
    """较历史最优回升 >10% 且持续 3 个 epoch → overfit_risk（优先于 converged）"""
    judge = LossJudge(MODEL_TYPE_SOVITS)
    series = [(1, 0.5), (2, 0.4), (3, 0.35), (4, 0.6), (5, 0.7), (6, 0.75)]
    status, reason = judge.judge(series, max_epochs=200)
    assert status == "overfit_risk"
    assert reason and "回升" in reason


def test_judge_improving_continues():
    """损失持续明显下降 → improving 不判停"""
    judge = LossJudge(MODEL_TYPE_SOVITS)
    series = [(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7), (5, 0.6)]
    status, reason = judge.judge(series, max_epochs=200)
    assert status == "improving"
    assert reason is None


def test_judge_min_epochs_protection():
    """min_epochs 保护：epoch 数不足 3 时不判停（即使已停滞/回升）"""
    judge = LossJudge(MODEL_TYPE_SOVITS)
    assert judge.judge([(1, 1.0), (2, 1.0)], max_epochs=200) == ("improving", None)
    assert judge.judge([(1, 0.4), (2, 0.8)], max_epochs=200) == ("improving", None)


def test_judge_max_epochs():
    """已解析到的最大 epoch 达到上限 → max_epochs（由引擎自然结束）"""
    judge = LossJudge(MODEL_TYPE_SOVITS)
    series = [(i, 1.0 - i * 0.1) for i in range(1, 6)]
    status, reason = judge.judge(series, max_epochs=5)
    assert status == "max_epochs"
    assert reason
    # 未到上限时同序列仍在明显下降 → improving
    assert judge.judge(series, max_epochs=6) == ("improving", None)


def test_judge_max_epochs_priority_over_min_epochs():
    """max_epochs 检查优先于 min_epochs 保护（1 个 epoch 即可判到顶）"""
    judge = LossJudge(MODEL_TYPE_MELOTTS)
    status, _ = judge.judge([(1, 1.0)], max_epochs=1)
    assert status == "max_epochs"


def test_loss_judge_invalid_engine():
    """LossJudge 不支持的引擎 → ValueError"""
    with pytest.raises(ValueError, match="Unsupported engine"):
        LossJudge("bert_vits")


def test_sovits_log_real_format_with_logger_prefix():
    """真实 sovits train.log 行（含路径前缀与 \tINFO\t 分隔）可解析"""
    log = (
        "2026-10-17 10:00:00,123\tDS-001\tINFO\tTrain Epoch: 12 [ 34%]\n"
        "2026-10-17 10:00:00,456\tDS-001\tINFO\t"
        "Losses: [0.5, 0.3, 0.2, 1.234, 0.1], step: 2400, lr: 0.0001, reference_loss: 2.334\n"
    )
    assert parse_sovits_train_log(log) == [(12, 1.234)]


def test_final_fitting_status_epoch_fallback(tmp_path):
    """无损失行（极短训练未达 log_interval）时按 epoch 标记行兜底回填 max_epochs"""
    from modelstation.services.auto_model_pipeline import AutoModelPipelineService

    service = AutoModelPipelineService()
    judge = LossJudge(MODEL_TYPE_SOVITS)
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "2026-10-17 10:00:00,000\tx\tINFO\t====> Epoch: 1\n"
        "2026-10-17 10:00:10,000\tx\tINFO\t====> Epoch: 2\n",
        encoding="utf-8",
    )
    status, reason = service._final_fitting_status(judge, log_path, 2)
    assert status == "max_epochs"
    assert "epoch 2" in (reason or "")

    # 未到上限 → 不回填
    log_path.write_text("====> Epoch: 1\n", encoding="utf-8")
    assert service._final_fitting_status(judge, log_path, 2) is None


def test_planned_eval_interval():
    """eval_interval 规划：按最坏 batch 估总步数，/目标存盘次数，带上限与下限 1"""
    from modelstation.services.auto_model_pipeline import _planned_eval_interval

    # 大语料长训练：总步数远超上限 → 封顶 1000（官方节拍）
    assert _planned_eval_interval(10000, 500) == 1000
    # 短训练：200 epochs × ceil(21/16)=2 步 = 400 步 → /10 = 40
    assert _planned_eval_interval(200, 21) == 40
    # 极短：5 epochs × 2 步 = 10 步 → 1（至少能存一次）
    assert _planned_eval_interval(5, 20) == 1
    # 退化输入：n_texts=0 也不产生 0/负值
    assert _planned_eval_interval(5, 0) == 1


# ---------------------------------------------------------------------------
# 服务层单元（非 API）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_rejects_conflict_without_api():
    """服务层运行互斥：running 存在 → PipelineRunConflictError"""
    service = AutoModelPipelineService()
    service._runs["existing"] = {"status": "running", "run_id": "existing"}
    with pytest.raises(PipelineRunConflictError):
        await service.submit_run(MODEL_TYPE_SOVITS, voice_description="描述")


@pytest.mark.asyncio
async def test_service_rejects_bad_model_type():
    """model_type 非法 → ValueError"""
    service = AutoModelPipelineService()
    with pytest.raises(ValueError, match="Unsupported model_type"):
        await service.submit_run("bert_vits", voice_description="描述")
