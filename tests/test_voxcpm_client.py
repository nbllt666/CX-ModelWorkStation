"""VoxCPM 客户端 CLI 入口单测（change-id: package-modelstation-desktop-installer / Task 4.5）。

锁定行为：上游引擎 engines/VoxCPM-main/src/voxcpm 无 __main__.py，后端子进程必须走
`-m voxcpm.cli`（console_script `voxcpm = voxcpm.cli:main` 的等价入口），
否则打包态会得到 "No module named voxcpm.__main__"。
"""
from __future__ import annotations

import sys

import pytest

from modelstation.services import voxcpm_client as mod
from modelstation.services.voxcpm_client import VoxCPMClient


def test_module_entry_constant() -> None:
    assert mod._VOXCPM_MODULE == "voxcpm.cli"


def test_build_base_args_uses_voxcpm_cli() -> None:
    args = VoxCPMClient()._build_base_args()
    assert args[:3] == [sys.executable, "-m", "voxcpm.cli"]
    # 不允许再出现不带子模块的 `-m voxcpm`（缺 __main__.py）
    assert "voxcpm" not in args
    assert args[-2:] == ["--model-path", "openbmb/VoxCPM2"]


@pytest.mark.asyncio
async def test_health_check_uses_voxcpm_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    client = VoxCPMClient()
    captured: dict = {}

    async def fake_run(args: list[str], timeout: float = 0.0) -> tuple[int, str, str]:
        captured["args"] = list(args)
        captured["timeout"] = timeout
        return 0, "", ""

    monkeypatch.setattr(client, "_run_subprocess", fake_run)
    assert await client.health_check() is True
    assert captured["args"] == [sys.executable, "-m", "voxcpm.cli", "--help"]