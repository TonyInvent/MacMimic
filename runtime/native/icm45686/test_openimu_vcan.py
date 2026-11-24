"""Exercise SocketCAN reception, bias, IPC and stream loss on an isolated vcan."""
import csv
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time


def main(binary, interface):
    link = json.loads(subprocess.check_output(["ip", "-d", "-j", "link", "show", "dev", interface]))[0]
    if link["linkinfo"]["info_kind"] != "vcan":
        raise ValueError("A virtual CAN interface is required")
    with tempfile.TemporaryDirectory() as directory, socket.socket(
        socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW
    ) as tx, socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
        tx.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
        tx.bind((interface,))
        path = directory + "/imu.sock"
        capture = Path(directory) / "imu.csv"
        client.bind(directory + "/client")
        client.setblocking(False)
        fd = os.memfd_create("openimu-vcan", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
        reader = None
        with os.fdopen(fd, "rb+") as shared, subprocess.Popen([binary, "capture-canfd", interface, "30", capture],
                              env=dict(os.environ, ICM_IPC_SOCKET=path, ICM_LATEST_FD=str(shared.fileno())),
                              pass_fds=(fd,),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            try:
                start = time.monotonic_ns()
                counts = (-8192, 0, 0, 1, -2, 3)
                deadline = start
                packet = None
                while time.monotonic_ns()-start < 19_000_000_000 and proc.poll() is None:
                    now = time.monotonic_ns()
                    data = struct.pack(">H7h", (now//1000) & 65535, *counts, -128)
                    tx.send(struct.pack("=IBB2x64s", 1, len(data), 1, data))
                    if reader is None and now-start > 15_000_000_000 and os.path.exists(path):
                        client.sendto(b"GETIMU01", path)
                        try:
                            packet = client.recv(193)
                            reader = subprocess.Popen([Path(binary).with_name("icm_latest_test"), str(fd)],
                                                      pass_fds=(fd,))
                        except BlockingIOError:
                            pass
                    if reader is not None and reader.poll() is not None:
                        assert reader.returncode == 0
                        break
                    deadline = max(deadline+1_250_000, now+100_000)
                    time.sleep(max(0, (deadline-time.monotonic_ns())*1e-9))
                output = proc.communicate(timeout=2)[0]
                assert reader is not None and reader.poll() == 0, output
                assert packet and packet[:8] == b"SPIMU003" and len(packet) == 192, output
                assert struct.unpack_from("<6h", packet, 32) == counts
                assert "BIAS_READY" in output and "stream timeout" in output and proc.returncode == 1, output
                with capture.open() as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) > 10000 and rows[-1]["raw_temp"] == "-128"
                print(f"{len(rows)} vcan samples, calibrated IPC and stream timeout passed")
            finally:
                if reader is not None and reader.poll() is None:
                    reader.kill()
                    reader.wait()
                if proc.poll() is None:
                    proc.kill()


if __name__ == "__main__":
    main(*sys.argv[1:])
