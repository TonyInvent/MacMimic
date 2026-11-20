"""Select a tested R1 package for local Xbox simulation."""
import json
import os
import statistics
import subprocess
import sys
import time

from compare import rank, read, root, run

checks = root / "tools/opensprite_validation"


def checked_report(script, package, name):
    path = package / name
    if not path.is_file():
        with (package / (script.stem + ".log")).open("w") as log:
            subprocess.run([sys.executable, str(script), str(package)], cwd=root, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    return read(path)


def fraction(rows, label):
    phase = next(row for row in rows if row["phase"] == label)
    return phase["measured"][2] / phase["command"][2]


def publish(iteration, packages, final):
    ranking = rank(packages)
    eligible = []
    for result in ranking:
        if result["failed_motion_clips"] or result["falls_in_seven_cases"] or result["falls_in_eight_stick_settings"]:
            continue
        if not result["both_turn_directions_in_four_cases"]:
            continue
        package = run / f'velocity_{result["iteration"]:06d}'
        sdl = checked_report(checks / "sdl_driving.py", package, "sdl_driving.json")["rows"]
        if any(row["fallen"] for row in sdl):
            continue
        phases = {row["phase"]: row["measured"] for row in sdl}
        if final:
            if min(phases["cruise"][0], phases["restart"][0]) <= .05 or abs(phases["brake"][0]) >= .05:
                continue
            sequences = read(package / "xbox_varied_starts.json").values()
            fractions = [statistics.median(fraction(seq, label) for seq in sequences)
                         for label in ("walking left", "walking right")]
            fractions.extend(fraction(sdl, label) for label in ("left", "right"))
            result["weakest_turn_fraction"] = min(fractions)
            if result["weakest_turn_fraction"] <= 0:
                continue
        eligible.append((package, result))
    if final:
        eligible.sort(key=lambda item: (-item[1]["both_turn_directions_in_four_cases"],
                                        -min(1., item[1]["weakest_turn_fraction"]), item[1]["score"]))
    for package, result in eligible:
        if final:
            gait = checked_report(checks / "gait.py", package, "gait.json")
            walk = gait["policy_walk_040"]
            if (walk.get("fallen") or min(walk["swing_onsets_left_right"]) == 0
                    or walk["same_foot_consecutive_onsets"] or any(gait["policy_stand"]["swing_onsets_left_right"])):
                continue
        replay = checked_report(checks / "upstream_history.py", package, "upstream_history_replay.json")
        if replay["completed"] and replay["max_abs_action_difference"] < 2e-4:
            result["selected"] = True
            temp = run / ".xbox_selected_next"
            temp.symlink_to(package.name, target_is_directory=True)
            os.replace(temp, run / "xbox_selected")
            print(f"After iteration {iteration}, selected {package.name}", flush=True)
            break
    else:
        if final:
            raise SystemExit("No package passed the full driving and gait checks.")
    temp = run / "checkpoint_comparison.tmp"
    temp.write_text(json.dumps(ranking, indent=2) + "\n")
    os.replace(temp, run / "checkpoint_comparison.json")


if __name__ == "__main__":
    final = "--all" in sys.argv[1:]
    stride = 100 if final else 500
    for iteration in ([6000] if final else range(500, 6001, 500)):
        packages = [run / f"velocity_{i:06d}" for i in range(stride, iteration + 1, stride)]
        while not all((p / "xbox_analog.json").is_file() for p in packages):
            time.sleep(15)
        publish(iteration, packages, final)
