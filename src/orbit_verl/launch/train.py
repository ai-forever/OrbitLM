"""Run verl's PPO trainer on a composed config.

This is what ``python -m verl.trainer.main_ppo`` does after hydra has built the config: set the
device, validate, then ``run_ppo(config, TaskRunnerV1)``. We call verl's own ``run_ppo`` instead of
copying it, so a verl upgrade changes the training path in one place only.

The one addition is the exit drift alarm (``orbit_verl.rltt.drift``), which has to see every
step's metrics inside the trainer. ``run_ppo`` takes the task runner class as an argument, which is
verl's way for recipes to change the trainer side. ``task_runner_class`` passes verl's own task
runner with one change: while it runs, verl's trainer registry hands out the trainer subclass that
runs the alarm.
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator


def validate(cfg: Any) -> None:
    """Fill the device field and run verl's own config validation; raises on an invalid config."""
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    auto_set_device(cfg)
    validate_config(cfg, use_reference_policy=need_reference_policy(cfg), use_critic=need_critic(cfg))


@contextlib.contextmanager
def trainer_with_exit_drift_alarm(config: Any) -> Iterator[type]:
    """Inside this block, verl's trainer registry returns the trainer that runs the exit drift alarm.

    ``TaskRunnerV1.run`` looks its trainer class up by ``trainer.v1.trainer_mode`` and builds the
    trainer in the same call, so the registry entry is the only place where a subclass can go in.
    The entry is swapped in the task runner's own Ray process, which runs one training and exits,
    and the stock class is put back when the block ends. Yields the subclass.
    """
    from verl.trainer.ppo.v1 import trainer_base

    from orbit_verl.rltt.drift import with_exit_drift_alarm

    mode = config.trainer.v1.trainer_mode
    stock = trainer_base.get_trainer_cls(mode)
    trainer_base.TRAINER_REGISTRY[mode] = alarmed = with_exit_drift_alarm(stock)
    try:
        yield alarmed
    finally:
        trainer_base.TRAINER_REGISTRY[mode] = stock


def task_runner_class():
    """verl's ``TaskRunnerV1`` as a Ray actor class whose trainer runs the exit drift alarm.

    verl defines ``TaskRunnerV1`` with ``@ray.remote``, so the module attribute is a Ray actor
    class, and Ray refuses to subclass one. Ray keeps the plain class in the actor class's
    metadata, and the subclass is made from that. Its ``run`` is verl's own ``run``, wrapped in
    ``trainer_with_exit_drift_alarm``.
    """
    import ray
    from verl.trainer.main_ppo import TaskRunnerV1

    stock = TaskRunnerV1.__ray_metadata__.modified_class.__ray_actor_class__

    class OrbitLMTaskRunner(stock):
        def run(self, config):
            with trainer_with_exit_drift_alarm(config):
                return super().run(config)

    return ray.remote(OrbitLMTaskRunner)


def run_training(cfg: Any, *, shutdown_ray: bool = True) -> None:
    """Validate ``cfg`` and train with verl's V1 trainer.

    ``run_ppo`` starts Ray itself unless Ray is already initialized. In local mode it starts a
    private cluster (``ray_kwargs.ray_init.address=local``); on the head of a multi-node job it
    connects to the cluster that the launcher started. ``shutdown_ray`` disconnects at the end,
    also after an error, so an interactive python can start the next run without a stale Ray session.
    """
    import ray
    from verl.trainer.main_ppo import run_ppo

    validate(cfg)
    if not cfg.trainer.get("use_v1", True):
        raise ValueError("trainer.use_v1=false selects verl's deprecated V0 trainer, which this launcher "
                         "does not support.")
    try:
        run_ppo(cfg, task_runner_class())
    finally:
        if shutdown_ray and ray.is_initialized():
            ray.shutdown()
