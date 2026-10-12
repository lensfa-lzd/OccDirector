from pathlib import Path


def find_ckpt(latest_version_dir) -> str | None:
    # 在最新版本中查找 checkpoint
    checkpoints_dir = latest_version_dir / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"版本 {latest_version_dir.name} 中没有 checkpoints 目录。")
        return None

    # 查找 last.ckpt
    last_ckpt_path = checkpoints_dir / "last.ckpt"
    if last_ckpt_path.exists():
        print(f"找到 'last.ckpt': {last_ckpt_path}")
        return str(last_ckpt_path)

    return None


def find_latest_version(log_dir: str, experiment_name: str) -> str | None:
    """
    自动查找指定实验的最新版本

    Args:
        log_dir (str): Lightning 的根日志目录 (e.g., 'lightning_logs/').
        experiment_name (str): 实验名称。

    Returns:
        str | None: 最新 checkpoint 的路径，如果找不到则返回 None。
    """
    exp_dir = Path(log_dir) / experiment_name
    if not exp_dir.exists():
        print(f"实验目录 {exp_dir} 不存在。将从头开始训练。")
        return None

    # 查找所有版本目录 'version_X'
    version_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("version_")]

    if not version_dirs:
        print(f"在 {exp_dir} 中找不到任何版本。将从头开始训练。")
        return None

    # 按数字对版本进行排序，找到最新的
    latest_version_num = -1
    latest_version_dir = None
    last_ckpt_path = None
    for d in version_dirs:
        try:
            version_num = int(d.name.split("_")[1])
            last_ckpt_path = find_ckpt(d)
            if version_num > latest_version_num and last_ckpt_path is not None:
                latest_version_num = version_num
                latest_version_dir = d
        except (ValueError, IndexError):
            continue

    if latest_version_dir is None:
        print(f"无法确定 {exp_dir} 中的最新版本。将从头开始训练。")
        return None

    last_ckpt_path = find_ckpt(latest_version_dir)
    print(f"找到最新版本: {latest_version_dir.name}, ckpt {last_ckpt_path}")
    return str(last_ckpt_path)
