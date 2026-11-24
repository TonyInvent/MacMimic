# OpenSprite v5 on Jetson 253

The 100k MacMimic actor runs at 50 Hz against live joint and ICM45686 feedback. The native CAN controller drives 31 joints across four KCAN FD interfaces. In the 300-second powered Xbox trial the controller completed 16,300 action groups over 326 seconds, including standing and braking. The maximum host time from oldest feedback to the final hardware TX completion report was 11.396 ms against a 20 ms limit. All 31 drives reported disabled at exit. The controller used the full training Kp/Kd. [Powered report](evidence/xbox300-commissioning.json) · [Motor TX audit](evidence/xbox300-action-tx-audit.json) · [Command range and completion timing](evidence/xbox300-summary.json)

## Controls and hardware evidence

| Profile | Xbox input | Forward / reverse (m/s) | Lateral (m/s) | Yaw (rad/s) | Recorded result |
| --- | ---: | --- | --- | --- | --- |
| Initial | 30 s | +0.40 / −0.05 | ±0.10 | ±0.20 | [Powered pass](evidence/xbox30-commissioning.json) |
| Extended | 120 s | +0.40 / −0.05 | ±0.10 | ±0.20 | [Powered pass](evidence/xbox120-commissioning.json) |
| Wide | 120 s | +0.70 / −0.50 | ±0.42 | ±0.70 | [Powered pass](evidence/xbox120-wide-commissioning.json) |
| Wide | 300 s | +0.70 / −0.50 | ±0.42 | ±0.70 | [Powered pass](evidence/xbox300-commissioning.json) |
| Wide, SPI IMU | 300 s | +0.70 / −0.50 | ±0.50 | ±0.70 | Left and right strafing confirmed on hardware. [Trial ended on feedback torque protection](evidence/xbox300-vy050-commissioning.json). |
| Wide, OpenIMU CAN FD | 300 s configured | +0.70 / −0.50 | ±0.50 | ±0.70 | Full-gain standing passed. Operator confirmed normal walking. All 31 drives reported disabled before the full window elapsed. [Battery test record](evidence/openimu-battery-validation.json) |

These ranges are command limits. In the completed ±0.42 m/s trial the policy received lateral commands from −0.068 to +0.061 m/s and yaw commands from −0.70 to +0.70 rad/s. Measured lateral ground speed was not available. Hardware results apply to the robot, calibration and fingerprints recorded with each trial.

The left stick controls forward/reverse and lateral velocity (left positive). Right-stick X controls yaw. After the control window opens, center the sticks, release LT and hold it again to command movement. Releasing LT brakes the command. In the current build, maximum lateral command takes about 5 s from rest and returns to zero in about 2.9 s on LT release. Each powered session begins with stationary IMU calibration and 20 s of supported standing. Window closure starts braking and the controller checks individual disable feedback at shutdown.

### Bidirectional strafing at the ±0.50 m/s command limit

On November 23, 2026, the onsite operator confirmed normal leftward and rightward walking on Jetson 253 in trial `trial-20261123-164524-MBuZ`. The recorded policy inputs reached both lateral command limits, −0.50 and +0.50 m/s. Lateral ground speed was not measured.

The 300-second input window ended early. The left hip pitch motor reported −28.0 Nm, triggering its feedback torque guard (threshold magnitude 27.9932 Nm). All 31 drives then confirmed disabled. The TX audit passed for all 4,831 dispatched action groups. [Commissioning report](evidence/xbox300-vy050-commissioning.json) · [TX audit](evidence/xbox300-vy050-action-tx-audit.json)

## Rebuild from this checkout

Run on the prepared Jetson with a matching ONNX Runtime CPU library and C++ headers (the recorded version was 1.23.2), CMake, a C++17 toolchain and Python with NumPy and ONNX Runtime. The checkout includes the [actor and bound profile](../policies/opensprite-velocity-100k/), [native sources](../runtime/native/), and [253 hardware configuration](hardware/). The latter contains robot-specific joint signs, offsets and limits. The existing Jetson Python and ORT locations are defaults. Override them with `SPRITE_PYTHON`, `SPRITE_ORT_INCLUDE` and `SPRITE_ORT_LIBRARY` when necessary.

```bash
bash sim2real/build-jetson.sh
```

This file-only build runs the native and ICM tests, verifies the model bundle, and compares an 803-frame recorded-input replay against its known output hash. It writes to `sim2real/build/jetson253`. Rebuild after moving the checkout because the local checksum manifest includes source paths.

## Commission on the prepared robot

The operator entry uses the existing 253 commissioning lock, CPU-frequency lease service, Xbox joystick at `/dev/input/js0`, an ICM-45686 and four motor CAN interfaces. The CAN thread runs FIFO 50 on CPU 2. Inference uses CPU 4 and CPU 5 with RR 1 and a 1.344 GHz lease. An earlier trace found a 9.905 ms runnable wait on CPU 4 during a missed deadline. [Scheduler comparison](evidence/wakeup-ab-summary.json)

After a Jetson reboot, check the loaded KCAN module. The tested echo-order module has srcversion `404E8D65A6EFF60079EAEC1` on kernel `5.15.199-tegra`. The [253 loader](../scripts/load-kcan-253.sh) checks and loads the existing host-specific modules. The [driver patch and setup files](kcan/) record the change and bus configuration. The loader expects the candidate and rollback modules already installed on the commissioning host.

```bash
bash scripts/load-kcan-253.sh --check
# If a reload is needed, with the drives disabled:
sudo bash scripts/load-kcan-253.sh
sudo python3 runtime/tools/openimu_interface.py --up
bash scripts/xbox300-vy050-macmimic-253.sh --qualify
```

Use `--qualify` when a build or IMU lacks matching evidence. This option runs the disabled transport and intentional deadline rejection tests as needed. The launcher keeps separate records for CAN FD and SPI and can recover earlier records from the build's trial directories. Use `--check-cache` to validate saved records. For routine walking, run the launcher at the robot with fall protection in place and the motor cutoff within reach. Enter `XBOX300` to start the powered phase using the saved records. If a required record is missing, the launcher exits.

```bash
bash scripts/xbox300-vy050-macmimic-253.sh
```

During powered standing or walking with either IMU, hold Xbox LB and RB together for three continuous seconds to end the run and disable all 31 drives. Start with both bumpers released. Releasing either bumper before three seconds resets the timer. The buttons work with LT released and the sticks in any position. The controller reports the request immediately and confirms shutdown only after individual drive feedback. An operator stop is recorded as an interrupted trial.

In the powered OpenIMU trial on Jetson 253, the controller recorded `XBOX_READY`, then `OPERATOR_DISABLE`, then `DISABLED_31`. All 31 drives reported disabled. [Powered stop record](evidence/xbox-bumper-disable.json)

Other [253 launchers](../scripts/) and the [initial preflight and trace analysis](jetson253/) refer to their original commissioning directories. [Incremental patches](patches/) record the development sequence from upstream through the 100k port, Xbox 30 s, Xbox 120 s, wide range, Xbox 300 s and lateral ±0.50 m/s. The checkout already contains the final source. Trial reports retain their original hardware qualification fields and binary hashes. A fresh build must collect its own admission evidence.

### IMU selection

The current Xbox launcher defaults to `--imu-backend canfd`. It discovers the dedicated PCAN-USB FD adapter (USB ID `0c72:0012`) on each start. Use `--imu-can-interface IFACE` if multiple matching adapters are present. The setup command above configures that adapter at 1 Mbit/s arbitration and 5 Mbit/s data. To use the original SPI wiring, select `--imu-backend spi` for qualification and each subsequent run. Qualification records include the selected source. Both paths use the X-down/Z-back sensor mounting, 15-second stationary bias and the same policy IPC format.

Both commissioning backends publish a complete snapshot to shared memory after each accepted, calibrated sample (nominally 800 Hz). Each 50 Hz actor step reads the latest committed snapshot directly and checks the sample timestamp against the 30 ms age limit. SPI integrates between GPIO data-ready timestamps and discards register reads that miss their sample period. OpenIMU integrates between sensor timestamps in consecutive CAN FD frames. The earlier Jetson measurements below used a 200 Hz socket client.

On Jetson 253, the 100k read-only actor ran 803 inference steps per backend using each physical IMU and synthetic joint feedback. SPI sample age at inference was 0.73 ms at the median and 1.37 ms at p99, measured from the GPIO edge. OpenIMU estimated sample age was 0.64 ms at the median and 1.33 ms at p99. Each actor rejected stale data when its producer was paused. All four motor CAN transmit counters stayed constant. [Direct-read measurements](evidence/imu-direct-read-253-audit.json)

In that capture every adjacent OpenIMU sensor timestamp advanced by 1,250 µs (800 Hz on the sensor clock). The Jetson kernel timestamps gave a mean arrival rate of 792.86 Hz on the host clock. The estimator uses the sensor-clock intervals for integration. [Capture summary](evidence/imu-direct-read-253-audit.json)

The onsite operator subsequently tested the direct-read runtime with Xbox control on both physical IMUs. The operator reported normal walking with OpenIMU and with the original SPI sensor. The battery ran out during the SPI trial, before the 300-second control window finished.

To repeat the read-only check on a built Jetson checkout with the OpenIMU interface up, run `python3 scripts/test-imu-253.py` as `tony`. Keep the robot stationary for both bias calibrations. Use `--imu-backend spi` or `--imu-backend canfd` to test one sensor. The script prints the path to a new `summary.json` and exits with a failure status if either actor test fails or a motor CAN transmit counter changes.

The OpenIMU board publishes a 16-byte CAN FD frame with standard ID `0x01`, a sensor timestamp and raw accelerometer, gyroscope and temperature readings. USB interface names can change when the KCAN driver reloads. During the battery test, the PCAN-USB FD at port `1-2.3` appeared as `can3`. The four motor channels appeared as `kcan1` through `kcan4` on a separate USB device. Reload the motor driver first, then bring up the OpenIMU interface. [OpenIMU firmware and frame format](https://github.com/TonyInvent/OpenIMU)

To repeat the full-gain zero-speed check on the prepared robot, run `bash scripts/stand-openimu-253.sh` at the operator terminal. It runs the disabled and deadline-fault checks, then prompts for `STAND` before enabling the drives. For walking, run `bash scripts/xbox300-vy050-macmimic-253.sh` at the terminal. It checks the Xbox profile's qualification and prompts for `XBOX300`. Both scripts accept `--imu-backend spi` for the original sensor.

For a standalone 60-second stationary capture, run the setup command above, read the interface name from its output and substitute it for `IFACE` below. This entry opens only the IMU socket. The CSV records the 16-bit sensor tick, monotonic kernel arrival and process read times, estimated sample time and raw counts. The sensor tick wraps every 65.536 ms. The estimate tracks excess delay relative to kernel arrivals. Absolute sensor-to-host latency needs an independent timing reference.

```bash
sim2real/build/jetson253/icm/sprite_icm45686 capture-canfd IFACE 60 openimu.csv
```

On Jetson 253, a 600-second stationary test collected 475,807 frames during recorded-input inference. Every adjacent sensor timestamp advanced by 1,250 µs. Host arrival rate averaged 793.01 Hz. Across 29,000 IPC reads, estimated sample age was 1.42 ms at p99 and 3.79 ms at maximum. The production policy client also accepted 3,000 consecutive snapshots in a separate 60-second window. [Stationary validation](evidence/openimu-stationary-validation.json)

With the initial mount, the two IMUs' mean gravity directions differed by 2.67° at rest. During manual rotation their body-frame angular velocities had correlations of 0.9995, 0.9999 and 0.9974 on X, Y and Z. When the OpenIMU producer was frozen, the policy client rejected stale data after 29.74 ms. [Motion and fault tests](evidence/openimu-dynamic-validation.json)

After OpenIMU was secured in its final position, the mean gravity directions differed by 0.26° using the same mounting rotation. On battery power the robot passed a 30.5-second, full-gain zero-command standing test. The subsequent Xbox trial passed its disabled and deadline-fault checks, entered the powered control window and walked normally according to the onsite operator. Individual motor feedback confirmed all 31 drives disabled about 175 seconds after the standing checkpoint. [Battery test record](evidence/openimu-battery-validation.json)

## Source and licensing

The runtime derives from [OpenSprite-SuperMimic at fe4e74f](https://github.com/TonyInvent/OpenSprite-SuperMimic/tree/fe4e74f96117debbd66e5511af256418e1edcd23/runtime). The runtime package declares AGPL-3.0-only. Its [license](../runtime/LICENSE) and the [TDK IMU license](../runtime/native/third_party/tdk_icm45686/LICENSE) are included. The MacMimic changes bind the 100k contract, retain raw previous actions and the training pose handoff, and add lateral Xbox control and bounded hardware profiles.
