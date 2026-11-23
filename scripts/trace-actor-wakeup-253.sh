#!/bin/bash
# Observe one no-enable commissioning actor using private tracefs instances.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
[[ $EUID == 0 && $(hostname -s) == tony-jetson ]] || exit 2
[[ $# -eq 0 || ( $# -eq 1 && $1 == --compare-gdm ) ]] || exit 2
compare=${1:-}
root=/sys/kernel/tracing/instances
out=$(mktemp -d /var/tmp/macmimic-wakeup.XXXXXXXX)
chmod 755 "$out"
main="$root/macmimic-actor-$$"
detail="$root/macmimic-cpus-$$"
desktop_pid=
saved_affinity=
desktop_start=
cleanup() {
  set +e
  if [[ -n $saved_affinity && -r /proc/$desktop_pid/stat ]] &&
     [[ $(awk '{print $22}' "/proc/$desktop_pid/stat") == "$desktop_start" ]]; then
    taskset -pc "$saved_affinity" "$desktop_pid" > "$out/desktop-restored.txt"
  fi
  for instance in "$main" "$detail"; do
    if [[ -d $instance ]]; then
      echo 0 > "$instance/tracing_on"
      rmdir "$instance"
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT TERM
for instance in "$main" "$detail"; do
  mkdir "$instance"
  echo 0 > "$instance/tracing_on"
  echo mono > "$instance/trace_clock"
  echo 8192 > "$instance/buffer_size_kb"
done
# Keep a high-detail rolling window on the two compute CPUs. The other
# instance retains only events for the actor main thread on all CPUs.
echo 30 > "$detail/tracing_cpumask"
for event in sched/sched_switch power/cpu_idle irq/irq_handler_entry irq/irq_handler_exit irq/softirq_entry irq/softirq_exit; do
  if [[ -e "$detail/events/$event/enable" ]]; then echo 1 > "$detail/events/$event/enable"; fi
done
if [[ -n $compare ]]; then
  # Retain the same tracepoints with more history for the compute CPUs.
  for cpu in 4 5; do echo 32768 > "$detail/per_cpu/cpu$cpu/buffer_size_kb"; done
  mapfile -t desktop < <(pgrep -u gdm -x gnome-shell)
  [[ ${#desktop[@]} -eq 1 ]] || { echo "Expected one GDM shell."; exit 3; }
  desktop_pid=${desktop[0]}
  desktop_start=$(awk '{print $22}' "/proc/$desktop_pid/stat")
  saved_affinity=$(awk '/^Cpus_allowed_list:/ {print $2}' "/proc/$desktop_pid/status")
  [[ $saved_affinity == 0-5 ]] || { echo "Unexpected initial GDM affinity."; exit 3; }
  taskset -pc 0,1 "$desktop_pid" > "$out/desktop-test.txt"
  echo "GDM main thread $desktop_pid temporarily restricted to CPU 0/1."
fi
echo "READY: waiting up to 10 minutes for the staged actor; output $out"
touch "$out/ready"
binary=/home/tony/macmimic-s49-100k-7947489/build/sprite_supermimic_protected_ipc
end=$((SECONDS+600))
pid=
until [[ -n $pid ]]; do
  (( SECONDS < end )) || exit 3
  for candidate in $(pgrep -x sprite_supermim || true); do
    if [[ $(readlink "/proc/$candidate/exe" || true) == "$binary" ]]; then pid=$candidate; break; fi
  done
  [[ -n $pid ]] || sleep 0.25
done
echo "$pid" > "$out/pid"
for event in sched_waking sched_wakeup; do
  echo "pid == $pid" > "$main/events/sched/$event/filter"
  echo 1 > "$main/events/sched/$event/enable"
done
echo "prev_pid == $pid || next_pid == $pid" > "$main/events/sched/sched_switch/filter"
echo 1 > "$main/events/sched/sched_switch/enable"
for event in sys_enter_ppoll sys_exit_ppoll sys_enter_poll sys_exit_poll sys_enter_recvfrom sys_exit_recvfrom; do
  if [[ -e "$main/events/syscalls/$event/enable" ]]; then
    echo "common_pid == $pid" > "$main/events/syscalls/$event/filter"
    echo 1 > "$main/events/syscalls/$event/enable"
  fi
done
echo 1 > "$detail/tracing_on"
echo 1 > "$main/tracing_on"
echo "Tracing actor $pid."
end=$((SECONDS+360))
while [[ -d /proc/$pid ]] && (( SECONDS < end )); do sleep 0.2; done
for instance in "$main" "$detail"; do echo 0 > "$instance/tracing_on"; done
for label in main detail; do
  instance=${!label}
  cat "$instance/trace" > "$out/$label.trace"
  for stat in "$instance"/per_cpu/cpu*/stats; do
    echo "$stat"; cat "$stat"
  done > "$out/$label.stats"
done
chmod 644 "$out"/*.trace "$out"/*.stats "$out/pid"
echo "Saved scheduler evidence in $out"
