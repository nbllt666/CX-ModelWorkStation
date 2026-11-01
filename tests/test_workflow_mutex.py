"""workflow 训练步骤的跨类型互斥冒烟测试（Task 5 补线验证）。

覆盖点：
1. melotts 占用互斥时，workflow training 步骤返回 409（HTTPException 透传，非 500）
2. workflow 训练启动成功后释放语义与 API 路径一致（start 抛错时 end_training 被调用）
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from modelstation.api import workflow as workflow_api
from modelstation.main import app
from modelstation.services import training_mutex

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_mutex():
    training_mutex.reset_registry()
    yield
    training_mutex.reset_registry()


def _patch_trainer(monkeypatch, start_training=None):
    trainer = MagicMock()
    trainer.start_training = start_training or AsyncMock(return_value="wf-task-1")
    monkeypatch.setattr(workflow_api, "_get_sovits_trainer", lambda: trainer)
    return trainer


def test_workflow_training_409_when_melotts_holds_mutex(monkeypatch):
    """melotts 占用互斥时，workflow training 步骤 409 且 detail 含当前训练类型。"""
    ok, current = training_mutex.try_begin_training("melotts", "mt-1")
    assert ok is True
    _patch_trainer(monkeypatch)

    resp = client.post("/api/workflow/step/training/execute", json={"epochs": 1})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["current_training"]["owner_type"] == "melotts"
    assert detail["current_training"]["task_id"] == "mt-1"
    # 互斥未被 workflow 侧误释放（仍由 melotts 持有）
    assert training_mutex.current_training()["owner_type"] == "melotts"


def test_workflow_training_passthrough_not_500(monkeypatch):
    """HTTPException 409 不得被 execute_step 的兜底 except 吞成 500。"""
    training_mutex.try_begin_training("melotts", "mt-2")
    _patch_trainer(monkeypatch)

    resp = client.post("/api/workflow/step/training/execute", json={})

    assert resp.status_code == 409


def test_workflow_training_start_failure_releases_mutex(monkeypatch):
    """start_training 抛错时互斥释放（允许后续重试）。"""
    failing = AsyncMock(side_effect=RuntimeError("boom"))
    _patch_trainer(monkeypatch, start_training=failing)

    resp = client.post("/api/workflow/step/training/execute", json={})

    assert resp.status_code == 500
    assert training_mutex.current_training() is None


def test_workflow_training_success_acquires_and_backfills(monkeypatch):
    """成功路径：互斥被 workflow 路径占用且回填真实 task_id。"""
    captured = {}

    async def fake_start(**kwargs):
        captured["callback"] = kwargs.get("progress_callback")
        return "wf-task-9"

    _patch_trainer(monkeypatch, start_training=fake_start)

    resp = client.post("/api/workflow/step/training/execute", json={})

    assert resp.status_code == 200
    steps = {s["id"]: s for s in resp.json()["steps"]}
    assert steps["training"]["output"]["task_id"] == "wf-task-9"
    current = training_mutex.current_training()
    assert current is not None and current["task_id"] == "wf-task-9"
    assert callable(captured["callback"])
    # 收尾：终态回调释放（模拟 trainer 完成）
    captured["callback"](status="completed")
    assert training_mutex.current_training() is None
