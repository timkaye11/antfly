#!/bin/bash
# Memory-pressure watchdog for GLiNER2 Metal training runs on small-RAM machines.
#
# Usage: gliner2_memory_watchdog.sh <logfile> -- <cmd> [args...]
#
# Samples the kernel memory-pressure level (kern.memorystatus_vm_pressure_level:
# 1=normal, 2=warn, 4=critical) every second and kills the wrapped command's
# whole process group at the first critical reading, or after 60s of sustained
# warn. Free-page counts (vm_stat) are NOT a reliable OOM signal on macOS with
# unified memory — the pressure level is what the jetsam killer acts on.
#
# Exit codes: 42 = killed on critical pressure, 43 = killed on sustained warn,
# otherwise the wrapped command's exit code.
set -u
LOG="$1"; shift
[ "${1:-}" = "--" ] && shift
set -m # job control: background job gets its own process group
"$@" >"$LOG" 2>&1 &
PID=$!
cleanup() { kill -9 -- -"$PID" 2>/dev/null; }
trap cleanup EXIT INT TERM
WARN_STREAK=0
while kill -0 "$PID" 2>/dev/null; do
  LEVEL=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1)
  if [ "$LEVEL" -ge 4 ]; then
    echo "WATCHDOG: pressure CRITICAL (level $LEVEL) — killing pgid $PID" | tee -a "$LOG" >&2
    kill -9 -- -"$PID" 2>/dev/null
    trap - EXIT
    exit 42
  elif [ "$LEVEL" -ge 2 ]; then
    WARN_STREAK=$((WARN_STREAK + 1))
    echo "WATCHDOG: pressure warn (level $LEVEL, streak $WARN_STREAK)" >>"$LOG"
    if [ "$WARN_STREAK" -ge 60 ]; then
      echo "WATCHDOG: sustained warn pressure ${WARN_STREAK}s — killing pgid $PID" | tee -a "$LOG" >&2
      kill -9 -- -"$PID" 2>/dev/null
      trap - EXIT
      exit 43
    fi
  else
    WARN_STREAK=0
  fi
  sleep 1
done
trap - EXIT
wait "$PID"
exit $?
