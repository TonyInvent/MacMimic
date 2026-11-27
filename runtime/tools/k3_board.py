"""Map the COM260 motor harness and inspect or start its native CAN FD ports."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

from open_sprite_runtime.socketcan import audit_socketcan_active_fd_snapshot

MOTOR_BUSES = {f"kcan{i + 1}": f"can{i}" for i in range(4)}
IMU_BUS = "can4"


def endpoint(name):
    model = Path("/proc/device-tree/model").read_bytes().rstrip(b"\0").decode()
    if "k3 com260" not in model.lower() or name not in (*MOTOR_BUSES.values(), IMU_BUS):
        raise ValueError("Expected a SpacemiT K3 COM260 native CAN endpoint")
    dev = Path("/sys/class/net") / name
    driver = (dev / "device/driver").resolve(strict=True).name
    if driver != "flexcan" or not int((dev / "flags").read_text(), 16) & 0x40000:
        raise ValueError(f"{name} requires FlexCAN driver echo support")
    return dict(interface=name, driver=driver, controller=(dev / "device/of_node").resolve(strict=True).name)


def remap_motor_table(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 31 or set(row["interface"] for row in rows) != MOTOR_BUSES.keys():
        raise ValueError("Unexpected motor harness in the reviewed profile")
    for row in rows:
        row["interface"] = MOTOR_BUSES[row["interface"]]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--up", action="store_true")
    args = parser.parse_args()
    ports = [endpoint(name) for name in (*MOTOR_BUSES.values(), IMU_BUS)]
    for port in ports:
        name = port["interface"]
        def inspect():
            return json.loads(subprocess.check_output(["ip", "-j", "-d", "link", "show", "dev", name]))
        state = inspect()
        if args.up:
            if "UP" not in state[0]["flags"]:
                subprocess.run(["ip", "link", "set", "dev", name, "type", "can", "bitrate", "1000000",
                                "dbitrate", "5000000", "fd", "on"], check=True)
                subprocess.run(["ip", "link", "set", "dev", name, "up"], check=True)
                state = inspect()
            report = audit_socketcan_active_fd_snapshot(state, name)
            if not report.passed:
                raise RuntimeError(report.errors)
        print(json.dumps(dict(port, flags=state[0]["flags"], mtu=state[0]["mtu"])))


if __name__ == "__main__":
    main()
