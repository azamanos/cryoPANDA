# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Iterable, Optional


def _import_wandb():
    try:
        import wandb  # type: ignore

        return wandb
    except ImportError:
        return None


class WandbLogger:
    """Light-weight wrapper around ``wandb`` to make logging optional."""

    def __init__(self, cfg) -> None:
        self._wandb = None
        self._run = None
        self._cfg = cfg

        wandb_cfg = getattr(cfg, "wandb", None)
        if wandb_cfg is None or not getattr(wandb_cfg, "enabled", False):
            return

        wandb_module = _import_wandb()
        if wandb_module is None:
            raise RuntimeError(
                "W&B logging is enabled but the 'wandb' package is not installed. "
                "Run `pip install wandb` to enable experiment tracking."
            )

        project = getattr(wandb_cfg, "project", "")
        if not project:
            raise ValueError("W&B logging requires a project name. Set `wandb.project` in the config file.")
        entity = getattr(wandb_cfg, "entity", "")
        run_name = getattr(wandb_cfg, "run_name", "")
        group = getattr(wandb_cfg, "group", "")
        job_type = getattr(wandb_cfg, "job_type", "")
        notes = getattr(wandb_cfg, "notes", "")
        tags: Iterable[str] = getattr(wandb_cfg, "tags", []) or []
        run_dir = getattr(wandb_cfg, "dir", "") or cfg.train.output_dir
        mode = getattr(wandb_cfg, "mode", "")
        resume = getattr(wandb_cfg, "resume", "")
        run_id = getattr(wandb_cfg, "id", "")

        init_kwargs = dict(
            project=project,
            entity=entity or None,
            name=run_name or None,
            group=group or None,
            job_type=job_type or None,
            dir=run_dir or None,
            mode=mode or None,
            notes=notes or None,
            tags=list(tags) or None,
            resume=resume or None,
            id=run_id or None,
        )

        self._wandb = wandb_module
        self._run = wandb_module.init(**init_kwargs)

    @property
    def enabled(self) -> bool:
        return self._wandb is not None and self._run is not None

    def log_metrics(self, metrics: dict, *, step: Optional[int] = None, commit: Optional[bool] = None) -> None:
        if not self.enabled or self._wandb is None:
            return

        if commit is None:
            commit = True
        self._wandb.log(metrics, step=step, commit=commit)

    def log_summary(self, metrics: dict) -> None:
        if not self.enabled or self._run is None:
            return

        for key, value in metrics.items():
            self._run.summary[key] = value

    def watch_model(self, model, *, log: Optional[str] = None, log_freq: Optional[int] = None) -> None:
        wandb_cfg = getattr(self._cfg, "wandb", None)
        if wandb_cfg is None or not getattr(wandb_cfg, "watch_model", False):
            return

        watch_log = getattr(wandb_cfg, "watch_log", "gradients")
        watch_log_freq = getattr(wandb_cfg, "watch_log_freq", 100)
        if log is not None:
            watch_log = log
        if log_freq is not None:
            watch_log_freq = log_freq

        if self._wandb is None:
            return

        self._wandb.watch(  # type: ignore[attr-defined]
            model,
            log=watch_log,
            log_freq=watch_log_freq,
        )

    def create_image(self, data, *, caption: Optional[str] = None):
        if not self.enabled or self._wandb is None:
            return None
        if caption:
            return self._wandb.Image(data, caption=caption)
        return self._wandb.Image(data)

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        self._run.finish()
