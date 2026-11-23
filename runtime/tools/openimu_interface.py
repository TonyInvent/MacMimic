"""Locate the dedicated PCAN-USB FD and bring up the OpenIMU link on request."""
import argparse
import json
from pathlib import Path
import subprocess


def find_openimu(interface=None, net=Path("/sys/class/net")):
    matches = []
    for entry in net.iterdir():
        if interface and entry.name != interface:
            continue
        if (entry / "type").read_text().strip() != "280":  # ARPHRD_CAN
            continue
        for usb in (entry / "device").resolve().parents:
            if (usb / "idVendor").exists():
                usb_id = tuple((usb / field).read_text().strip() for field in ("idVendor", "idProduct"))
                if usb_id == ("0c72", "0012"):
                    serial = usb / "serial"
                    matches.append(dict(backend="canfd", interface=entry.name, usb_port=usb.name,
                                        serial=serial.read_text().strip() if serial.exists() else ""))
                break
    if len(matches) != 1:
        raise ValueError("Expected one PCAN-USB FD OpenIMU interface; check USB or select --imu-can-interface")
    return matches[0]


def link(name):
    return json.loads(subprocess.check_output(["ip", "-j", "-d", "link", "show", "dev", name]))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imu-can-interface")
    parser.add_argument("--up", action="store_true", help="configure a stopped adapter for 1 Mbit/s / 5 Mbit/s FD")
    args = parser.parse_args()
    adapter = find_openimu(args.imu_can_interface)
    name = adapter["interface"]
    state = link(name)
    if args.up and "UP" not in state["flags"]:
        subprocess.run(["ip", "link", "set", "dev", name, "type", "can", "bitrate", "1000000",
                        "dbitrate", "5000000", "fd", "on"], check=True)
        subprocess.run(["ip", "link", "set", "dev", name, "up"], check=True)
        state = link(name)
    data = state["linkinfo"]["info_data"]
    print(json.dumps(dict(adapter=adapter, flags=state["flags"], mtu=state["mtu"], can=data)))
    if args.up and ("UP" not in state["flags"] or state["mtu"] != 72 or
                    data.get("bittiming", {}).get("bitrate") != 1000000 or
                    data.get("data_bittiming", {}).get("bitrate") != 5000000 or data.get("state") == "BUS-OFF"):
        raise SystemExit("OpenIMU link is not active at 1 Mbit/s / 5 Mbit/s FD")


if __name__ == "__main__":
    main()
