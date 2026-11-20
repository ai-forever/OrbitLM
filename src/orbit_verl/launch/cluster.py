"""Forming one Ray cluster out of the nodes of a multi-node launch.

Check a rewrite of this logic with a 2-node launch before trusting it. The two nodes must form one
Ray cluster, and an NCCL all-reduce must then run across both of them. The pieces:

- node rank 0 runs ``ray start --head`` and waits until every node and every GPU has joined;
- every other node waits for the head's port, runs ``ray start --address=<head>``, and then stays
  until the head is gone, so every node's process ends cleanly when training ends;
- every ``ray`` command runs as ``<this python> -m ray.scripts.scripts``, which works even when the
  env's ``bin`` folder is not on PATH, and gets the environment without the distributed launcher's
  variables.

Only the standard library is imported at module level. ``ray`` itself is imported where the head
waits for the cluster.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from typing import Callable, Mapping

from orbit_verl.launch.env import strip_launcher_env

DEFAULT_PORT = 6379
# A worker node checks the head's port this often, and leaves after this many misses in a row.
HEAD_POLL_SECONDS = 10
HEAD_MISSES_TO_LEAVE = 3


def own_ip(toward: str | None = None) -> str:
    """The address other nodes can reach this one at.

    Connecting a UDP socket sends nothing. It only makes the kernel pick the interface it would
    route through, which is the one Ray should advertise. Without ``toward`` (on the head node) the
    target is 10.255.255.255, an address of the private 10.0.0.0/8 range. The kernel then picks the
    interface that routes into that range when the node has one, and the default route's otherwise.
    """
    target = toward or "10.255.255.255"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((target, 1))
            return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def port_open(host: str, port: int, timeout: float = 5.0) -> bool:
    """True when a TCP connection to ``host:port`` succeeds."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def resolve_host(name: str) -> str:
    """IPv4 address of a host name; the name itself when it cannot be resolved here."""
    try:
        return socket.gethostbyname(name)
    except OSError:
        return name


def ray_command(args: list[str]) -> list[str]:
    """The full command line of one ``ray`` CLI call."""
    return [sys.executable, "-m", "ray.scripts.scripts", *args]


def head_start_args(*, ip: str, port: int, gpus: int, num_cpus: int | None = None) -> list[str]:
    """Arguments of ``ray start`` on node rank 0.

    The dashboard is off because nothing reads it and it would need another open port.
    """
    args = ["start", "--head", f"--port={port}", f"--node-ip-address={ip}", f"--num-gpus={gpus}",
            "--disable-usage-stats", "--include-dashboard=false"]
    if num_cpus is not None:
        args.append(f"--num-cpus={num_cpus}")
    return args


def worker_start_args(*, head: str, port: int, ip: str, gpus: int, num_cpus: int | None = None) -> list[str]:
    """Arguments of ``ray start`` on the other nodes."""
    args = ["start", f"--address={head}:{port}", f"--node-ip-address={ip}", f"--num-gpus={gpus}",
            "--disable-usage-stats"]
    if num_cpus is not None:
        args.append(f"--num-cpus={num_cpus}")
    return args


def ray_cli(args: list[str], *, check: bool = True, env: Mapping[str, str] | None = None,
            log: Callable[[str], None] = print) -> int:
    """Run one ``ray`` CLI command with the launcher variables removed from its environment.

    The launcher already removed those variables from ``os.environ``. Stripping them again here
    keeps this function safe for any caller, since every Ray process inherits this environment.
    """
    cmd = ray_command(args)
    log("$ " + " ".join(cmd))
    rc = subprocess.call(cmd, env=strip_launcher_env(os.environ if env is None else env))
    if check and rc != 0:
        raise RuntimeError(f"`ray {args[0]}` exited with code {rc}")
    return rc


def start_head(*, ip: str, port: int, gpus: int, num_cpus: int | None = None,
               log: Callable[[str], None] = print) -> int:
    """Start the Ray head on this node; returns the exit code of ``ray start``."""
    return ray_cli(head_start_args(ip=ip, port=port, gpus=gpus, num_cpus=num_cpus), check=False, log=log)


def stop_ray(log: Callable[[str], None] = print) -> int:
    """Stop every Ray process of this node. Failures are ignored: this runs during cleanup."""
    return ray_cli(["stop", "--force"], check=False, log=log)


def wait_for_cluster(address: str, *, nodes: int, gpus_per_node: int, timeout: float,
                     log: Callable[[str], None] = print) -> dict:
    """Wait until the cluster at ``address`` has ``nodes`` alive nodes and all their GPUs.

    The driver disconnects afterwards. verl's ``run_ppo`` calls ``ray.init`` only when Ray is not
    initialized yet, and only that call passes the job's runtime env to the Ray workers.
    """
    import ray

    want_gpus = nodes * gpus_per_node
    ray.init(address=address, log_to_driver=False)
    try:
        deadline, last_log = time.time() + timeout, 0.0
        while True:
            alive = [n for n in ray.nodes() if n.get("Alive")]
            have_gpus = int(ray.cluster_resources().get("GPU", 0))
            if len(alive) >= nodes and have_gpus >= want_gpus:
                return {"ok": True, "nodes": len(alive), "gpus": have_gpus}
            if time.time() - last_log > 15:
                log(f"joined {len(alive)}/{nodes} nodes, {have_gpus}/{want_gpus} GPUs")
                last_log = time.time()
            if time.time() > deadline:
                return {"ok": False, "nodes": len(alive), "gpus": have_gpus,
                        "joined": [n.get("NodeManagerAddress") for n in alive]}
            time.sleep(3)
    finally:
        ray.shutdown()


def run_worker_until_head_gone(*, head: str, port: int, gpus: int, num_cpus: int | None,
                               join_timeout: float, max_lifetime: float,
                               log: Callable[[str], None] = print) -> int:
    """Join the head's cluster and stay until the head is gone. Returns 0, or 3 if joining failed.

    A worker node has no other way to learn that training ended: the head stops Ray, its port
    closes, and the worker leaves after three missed polls. A head node that disappears is caught
    the same way, because its port stops answering too.

    ``max_lifetime`` is an optional hard limit in seconds, and 0 or less means no limit. The limit
    applies even while the head is healthy. Leaving stops this node's Ray processes, which kills
    the training actors on it, so a limit shorter than the training run fails the run.
    """
    deadline = time.time() + join_timeout
    log(f"waiting for the Ray head at {head}:{port}")
    while not port_open(head, port):
        if time.time() > deadline:
            log(f"FAIL: {head}:{port} did not open within {join_timeout:.0f} s")
            return 3
        time.sleep(5)

    args = worker_start_args(head=head, port=port, ip=own_ip(head), gpus=gpus, num_cpus=num_cpus)
    if ray_cli(args, check=False, log=log) != 0:
        log("FAIL: ray start on this worker node failed")
        stop_ray(log=log)
        return 3

    started, misses = time.time(), 0
    limited = max_lifetime > 0
    while misses < HEAD_MISSES_TO_LEAVE and not (limited and time.time() - started >= max_lifetime):
        time.sleep(HEAD_POLL_SECONDS)
        misses = 0 if port_open(head, port) else misses + 1
    if misses >= HEAD_MISSES_TO_LEAVE:
        log("the head is gone; stopping the local Ray node")
    else:
        log(f"max lifetime of {max_lifetime:.0f} s reached; stopping the local Ray node")
    stop_ray(log=log)
    return 0
