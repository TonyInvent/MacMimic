#!/usr/bin/env python3
"""Exercise both physical IMUs with the read-only 100k actor and synthetic joint state."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "sim2real/build/jetson253"
sys.path.insert(0, str(ROOT / "runtime/tools"))
from openimu_interface import find_openimu

LOCK = Path("/home/tony/supermimic-run002-20261106/commissioning.lock")
FRAMES = 803
STATE = struct.Struct("<8sQQQ496x")
MOTOR_HASH = 123  # Matches the synthetic identity feedback configuration.


def motor_tx():
    return [int(Path(f"/sys/class/net/kcan{i}/statistics/tx_packets").read_text())
            for i in range(1, 5)]


def terminate(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_actor(directory, env, fd, producer, freeze=False):
    name = "freeze" if freeze else "normal"
    log_path = directory / f"{name}.log"
    # Keep the Unix socket address short, independently of the evidence directory.
    with tempfile.TemporaryDirectory(prefix="imu-state-") as sockets, \
            socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as server, log_path.open("w") as log:
        address = str(Path(sockets) / "state")
        server.bind(address)
        server.listen(1)
        server.settimeout(8)
        args = ["taskset", "-c", "4,5", BUILD / "build/sprite_supermimic_ipc_readonly",
                ROOT / "policies/opensprite-velocity-100k/policy.onnx",
                ROOT / "sim2real/evidence/xbox803-input.bin", directory / name,
                "--macmimic-100k", "--threads", "2", "--no-ort-spinning", "--isolated-inference",
                "--warmup", "10", "--feedback-config", directory.parent / "feedback.bin",
                "--state-socket", address, "--imu-kind", "icm", "--imu-latest-fd", str(fd)]
        actor = subprocess.Popen(args, pass_fds=(fd,), env=env, stdout=log, stderr=subprocess.STDOUT)
        stopped_at = None
        try:
            with server.accept()[0] as peer:
                peer.settimeout(2)
                deadline = time.monotonic()
                for frame in range(FRAMES):
                    if freeze and frame == 20:
                        producer.send_signal(signal.SIGSTOP)
                        stopped_at = time.monotonic()
                    if stopped_at is not None and time.monotonic() - stopped_at > 1:
                        raise RuntimeError("Actor continued after IMU publication stopped")
                    try:
                        peer.sendall(STATE.pack(b"SPRT\x01\x00\x01\x00", frame + 1,
                                                time.monotonic_ns(), MOTOR_HASH))
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    deadline += 0.020
                    time.sleep(max(0, deadline - time.monotonic()))
            rc = actor.wait(timeout=5)
            if freeze:
                errors = [line for line in log_path.read_text().splitlines() if "IMU stale" in line]
                if stopped_at is None or rc == 0 or not errors:
                    raise RuntimeError(f"Expected a stale IMU rejection; see {log_path}")
                return {"error": errors[-1],
                        "observed_exit_after_stop_ms": round((time.monotonic() - stopped_at) * 1000, 2)}
            if rc:
                raise RuntimeError(f"Read-only actor exited {rc}; see {log_path}")
            timing = json.loads((directory / "normal.timing.json").read_text())
            affinity = json.loads((directory / "normal.affinity.json").read_text())
            if (timing["frames"] != FRAMES or not affinity["verified"] or affinity["io_cpu"] != -1
                    or len(affinity["threads"]) != 2):
                raise RuntimeError("Unexpected inference frame count or thread layout")
            return {"frames": timing["frames"], "imu_age_ms": timing["imu_age_ms"], "affinity": affinity}
        finally:
            try:
                terminate(actor)
            finally:
                if stopped_at is not None and producer.poll() is None:
                    producer.send_signal(signal.SIGCONT)


def run_backend(directory, backend, interface, env):
    directory.mkdir()
    capture = directory / "raw.csv"
    if backend == "spi":
        args = ["capture", "/dev/spidev0.0", "/dev/gpiochip0", "123", "800", "80", capture,
                "X_DOWN_Z_BACK", "CONFIGURE_ICM45686_RAW_NO_CAN"]
        clock_field = "event_ns"
    else:
        args = ["capture-canfd", find_openimu(interface)["interface"], "80", capture]
        clock_field = "kernel_rx_ns"
    log_path = directory / "imu.log"
    fd = os.memfd_create("imu-test", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        with log_path.open("w") as log:
            producer = subprocess.Popen(["taskset", "-c", "3", BUILD / "icm/sprite_icm45686", *args],
                                        env=dict(env, ICM_LATEST_FD=str(fd)), pass_fds=(fd,),
                                        stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 25
                while "BIAS_READY" not in log_path.read_text():
                    if producer.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(f"IMU startup or stationary calibration failed; see {log_path}")
                    time.sleep(0.05)
                if len(list(Path(f"/proc/{producer.pid}/task").iterdir())) != 1:
                    raise RuntimeError("Expected a single IMU acquisition thread")
                normal = run_actor(directory, env, fd, producer)
                freeze = run_actor(directory, env, fd, producer, freeze=True)
                reason = producer.wait(timeout=5)
                if reason == 0:
                    raise RuntimeError("Producer accepted the induced stream interruption")
            finally:
                terminate(producer)
    finally:
        os.close(fd)

    with capture.open() as stream:
        rows = list(csv.DictReader(stream))
    stamps = [int(row[clock_field]) for row in rows]
    gaps = [(end - start) / 1000 for start, end in zip(stamps, stamps[1:])]
    result = dict(normal=normal, freeze=freeze, producer_fault=log_path.read_text().splitlines()[-1],
                  host_rate_hz=round((len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0]), 3),
                  median_host_gap_us=statistics.median(gaps))
    if backend == "spi":
        result["spi_capture"] = json.loads((directory / "raw.csv.json").read_text())
    else:
        ticks = [int(row["sensor_tick_us"]) for row in rows]
        result["sensor_interval_us"] = sorted({(end - start) % 65536 for start, end in zip(ticks, ticks[1:])})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imu-backend", choices=("both", "spi", "canfd"), default="both")
    parser.add_argument("--imu-can-interface", help="select among multiple PCAN-USB FD adapters")
    args = parser.parse_args()
    if args.imu_backend == "spi" and args.imu_can_interface:
        parser.error("--imu-can-interface requires a CAN FD test")
    backends = ("spi", "canfd") if args.imu_backend == "both" else (args.imu_backend,)
    with LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("The commissioning lock is busy")
        before = motor_tx()
        work = Path(tempfile.mkdtemp(prefix="imu-audit-", dir=BUILD))
        print(f"Evidence: {work}", flush=True)
        identity = [float(i % 32 == 0) for i in range(31 * 31)]
        (work / "feedback.bin").write_bytes(struct.pack("<8sQ248x961d", b"SPMSEN01", MOTOR_HASH, *identity))
        library_paths = [str(BUILD / "build"), os.environ.get("LD_LIBRARY_PATH")]
        env = dict(os.environ, LD_LIBRARY_PATH=":".join(filter(None, library_paths)))
        env.pop("ICM_IPC_SOCKET", None)
        report = {}
        for backend in backends:
            print(f"{backend}: keep the robot still for calibration", flush=True)
            try:
                report[backend] = run_backend(work / backend, backend, args.imu_can_interface, env)
                print(f"{backend}: PASS, p99 IMU age {report[backend]['normal']['imu_age_ms']['p99']:.3f} ms",
                      flush=True)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                report[backend] = {"error": str(exc)}
                print(f"{backend}: FAIL, {exc}", file=sys.stderr, flush=True)
        report["motor_tx_delta"] = [end - start for start, end in zip(before, motor_tx())]
        (work / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        passed = not any("error" in report[b] for b in backends) and not any(report["motor_tx_delta"])
        print(f"{'PASS' if passed else 'FAIL'}: {work / 'summary.json'}")
        return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
