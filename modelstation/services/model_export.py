"""
模型导出（MeloTTS / So-VITS-SVC 共用）：
把 models_dir/<模型名> 复制到用户指定的绝对路径下，目标为 <dest_dir>/<模型名>；
- checkpoint 为空：整目录复制（配置 + 全部检查点）；
- checkpoint 指定（G_*.pth）：仅复制该检查点 + config.json（小体积部署包）。
目标已存在时拒绝（不覆盖既有文件）。
"""
import shutil
from pathlib import Path


def export_model_dir(
    models_dir: Path,
    model_name: str,
    dest_dir: str,
    checkpoint: str | None = None,
) -> dict:
    """导出模型到 <dest_dir>/<模型名>（可只导出单个 G 检查点）。

    Raises:
        ValueError: 模型名/检查点非法、模型不存在、目标非绝对路径、写入失败
        FileExistsError: 目标目录已存在（不覆盖）
    """
    name = str(model_name or "").strip()
    if not name or Path(name).name != name:
        raise ValueError("模型名称非法（不允许包含路径分隔符）")
    src = Path(models_dir) / name
    if not src.is_dir():
        raise ValueError(f"模型不存在: {name}")

    dest = Path(str(dest_dir or "").strip()).expanduser()
    if not dest.is_absolute():
        raise ValueError("目标目录必须是绝对路径")
    if dest.exists() and not dest.is_dir():
        raise ValueError(f"目标路径不是目录: {dest}")

    ckpt_name = str(checkpoint or "").strip()
    ckpt_path: Path | None = None
    if ckpt_name:
        if Path(ckpt_name).name != ckpt_name:
            raise ValueError("检查点名称非法（不允许包含路径分隔符）")
        if not (ckpt_name.startswith("G_") and ckpt_name.endswith(".pth")):
            raise ValueError("仅支持导出 G_*.pth 检查点")
        ckpt_path = src / ckpt_name
        if not ckpt_path.is_file():
            raise ValueError(f"检查点不存在: {ckpt_name}")

    target = dest / name
    if target.exists():
        raise FileExistsError(f"目标目录已存在: {target}")

    try:
        dest.mkdir(parents=True, exist_ok=True)
        if ckpt_path is None:
            shutil.copytree(src, target)
        else:
            # 小体积导出：检查点 + 推理/加载所需配置，不含 D/DUR 与其他检查点
            target.mkdir()
            config = src / "config.json"
            if config.is_file():
                shutil.copy2(config, target / config.name)
            shutil.copy2(ckpt_path, target / ckpt_path.name)
    except OSError as exc:
        # 复制中断时清理半成品，避免留下不完整模型目录
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        raise ValueError(f"导出失败: {exc}") from exc

    files = [p for p in target.rglob("*") if p.is_file()]
    return {
        "export_path": str(target),
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
    }
