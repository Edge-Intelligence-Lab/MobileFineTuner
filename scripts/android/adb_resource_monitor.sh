#!/system/bin/sh
set -u

PID=""
OUT=""
SUMMARY=""
INTERVAL_MS="5"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pid)
      PID="$2"
      shift 2
      ;;
    --csv)
      OUT="$2"
      shift 2
      ;;
    --summary)
      SUMMARY="$2"
      shift 2
      ;;
    --interval_ms)
      INTERVAL_MS="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$PID" ] || [ -z "$OUT" ] || [ -z "$SUMMARY" ]; then
  echo "usage: $0 --pid PID --interval_ms 5 --csv OUT --summary OUT" >&2
  exit 2
fi

dir=$(dirname "$OUT")
mkdir -p "$dir"

echo "sample,t_epoch_ns,dt_ms,requested_interval_ms,pid,VmRSS_kB,VmHWM_kB,RssAnon_kB,RssFile_kB,dumpsys_total_pss_kB,dumpsys_total_rss_kB,current_now_uA,current_average_uA,voltage_mV,charge_counter_uAh,temp_decic,level_percent,estimated_power_mW,estimated_power_abs_mW,status" > "$OUT"
echo "pid=$PID requested_interval_ms=$INTERVAL_MS start_epoch_ns=$(date +%s%N) note=adb_shell_can_sample_proc_status_lightly_but_dumpsys_meminfo_and_battery_service_are_best_effort" > "$SUMMARY"

i=0
prev=$(date +%s%N)
sleep_s=$(awk -v ms="$INTERVAL_MS" 'BEGIN {printf "%.6f", ms / 1000.0}')

while [ -d "/proc/$PID" ]; do
  now=$(date +%s%N)
  dt=$(awk -v n="$now" -v p="$prev" 'BEGIN {printf "%.3f", (n - p) / 1000000.0}')
  prev=$now

  proc_values=$(awk '
    /^VmRSS:/ {rss=$2}
    /^VmHWM:/ {hwm=$2}
    /^RssAnon:/ {anon=$2}
    /^RssFile:/ {file=$2}
    END {
      if (rss == "") rss = 0
      if (hwm == "") hwm = 0
      if (anon == "") anon = 0
      if (file == "") file = 0
      printf "%s,%s,%s,%s", rss, hwm, anon, file
    }
  ' "/proc/$PID/status" 2>/dev/null)

  meminfo=$(dumpsys meminfo "$PID" 2>/dev/null)
  total_pss=$(echo "$meminfo" | awk '/TOTAL PSS:/ {print $3; exit}')
  total_rss=$(echo "$meminfo" | awk '/TOTAL PSS:/ {print $6; exit}')
  [ -n "$total_pss" ] || total_pss="NA"
  [ -n "$total_rss" ] || total_rss="NA"

  cur=$(cmd battery get -f current_now 2>/dev/null | tr -d '\r')
  avg=$(cmd battery get -f current_average 2>/dev/null | tr -d '\r')
  cnt=$(cmd battery get -f counter 2>/dev/null | tr -d '\r')
  temp=$(cmd battery get -f temp 2>/dev/null | tr -d '\r')
  lvl=$(cmd battery get -f level 2>/dev/null | tr -d '\r')
  status=$(cmd battery get -f status 2>/dev/null | tr -d '\r')
  volt=$(dumpsys battery 2>/dev/null | awk '$1 == "voltage:" {print $2; exit}')

  power=$(awk -v c="$avg" -v v="$volt" 'BEGIN {
    if (c ~ /^-?[0-9]+$/ && v ~ /^[0-9]+$/) {
      printf "%.6f", (c * v) / 1000000.0
    } else {
      printf "NA"
    }
  }')
  power_abs=$(awk -v c="$avg" -v v="$volt" 'BEGIN {
    if (c ~ /^-?[0-9]+$/ && v ~ /^[0-9]+$/) {
      if (c < 0) c = -c
      printf "%.6f", (c * v) / 1000000.0
    } else {
      printf "NA"
    }
  }')

  i=$((i + 1))
  echo "$i,$now,$dt,$INTERVAL_MS,$PID,$proc_values,$total_pss,$total_rss,$cur,$avg,$volt,$cnt,$temp,$lvl,$power,$power_abs,$status" >> "$OUT"
  sleep "$sleep_s"
done

end=$(date +%s%N)
echo "end_epoch_ns=$end samples=$i" >> "$SUMMARY"
