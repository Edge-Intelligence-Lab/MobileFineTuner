#!/usr/bin/env bash
set -euo pipefail

# One-click on-device QA runner for MobileFineTuner Visualizer.
# - Build + install APK
# - Launch app and optionally load demo runs
# - Auto switch tabs while screen-recording
# - Capture a deterministic screenshot suite for acceptance
#
# Example:
#   ./scripts/qa/mobile_device_qa.sh --serial ABCDEF --out-dir verification/phone_qa

APP_ID="com.mobilefinetuner.visualizer"
MAIN_ACTIVITY="${APP_ID}.MainActivity"
APK_REL_PATH="app/build/outputs/apk/debug/app-debug.apk"

ADB_BIN="${ADB_BIN:-adb}"
SERIAL=""
OUT_DIR=""
VIDEO_SECONDS=120
SHOT_INTERVAL=12
VIDEO_BITRATE=12000000
SKIP_BUILD=0
SKIP_INSTALL=0
LOAD_DEMO=1
ENABLE_RECORD=1

SCREEN_W=0
SCREEN_H=0
RECORD_PID=""
WARNINGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    cat <<'USAGE'
Usage:
  mobile_device_qa.sh [options]

Options:
  --serial <id>          ADB serial to use (required when multiple devices are connected)
  --out-dir <path>       Output directory for screenshots/video/report
  --video-seconds <n>    Screen recording duration in seconds (default: 120, max: 180)
  --shot-interval <n>    Interval between auto snapshots during recording (default: 12)
  --skip-build           Skip Gradle build step
  --skip-install         Skip APK install step
  --no-demo              Do not tap "Load Demo"
  --no-record            Do not run screen recording
  -h, --help             Show this help
USAGE
}

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

warn() {
    log "WARN: $*"
    WARNINGS+=("$*")
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

resolve_adb_bin() {
    if command -v "$ADB_BIN" >/dev/null 2>&1; then
        ADB_BIN="$(command -v "$ADB_BIN")"
        return
    fi

    local candidates=(
        "${HOME}/Library/Android/sdk/platform-tools/adb"
        "${ANDROID_SDK_ROOT:-}/platform-tools/adb"
        "${ANDROID_HOME:-}/platform-tools/adb"
    )

    local c
    for c in "${candidates[@]}"; do
        if [[ -n "$c" && -x "$c" ]]; then
            ADB_BIN="$c"
            return
        fi
    done

    die "adb not found. Set ADB_BIN or install Android platform-tools."
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --serial)
                SERIAL="${2:-}"
                [[ -n "$SERIAL" ]] || die "--serial requires a value"
                shift 2
                ;;
            --out-dir)
                OUT_DIR="${2:-}"
                [[ -n "$OUT_DIR" ]] || die "--out-dir requires a value"
                shift 2
                ;;
            --video-seconds)
                VIDEO_SECONDS="${2:-}"
                shift 2
                ;;
            --shot-interval)
                SHOT_INTERVAL="${2:-}"
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD=1
                shift
                ;;
            --skip-install)
                SKIP_INSTALL=1
                shift
                ;;
            --no-demo)
                LOAD_DEMO=0
                shift
                ;;
            --no-record)
                ENABLE_RECORD=0
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac
    done
}

ADB_ARGS=()
adb_cmd() {
    "$ADB_BIN" "${ADB_ARGS[@]}" "$@"
}

select_device() {
    if [[ -n "$SERIAL" ]]; then
        ADB_ARGS=(-s "$SERIAL")
        return
    fi

    local devices
    devices="$("$ADB_BIN" devices | awk 'NR>1 && $2=="device"{print $1}')"
    local count
    count="$(printf '%s\n' "$devices" | sed '/^$/d' | wc -l | tr -d ' ')"

    if [[ "$count" -eq 0 ]]; then
        die "No connected adb device found."
    fi
    if [[ "$count" -gt 1 ]]; then
        die "Multiple adb devices found; pass --serial."
    fi

    SERIAL="$(printf '%s\n' "$devices" | head -n 1)"
    ADB_ARGS=(-s "$SERIAL")
}

ensure_device_ready() {
    local state
    state="$(adb_cmd get-state 2>/dev/null || true)"
    [[ "$state" == "device" ]] || die "Device is not ready (state=$state)"
}

is_app_foreground() {
    local focus
    focus="$(adb_cmd shell dumpsys activity activities 2>/dev/null | tr -d '\r' | grep -m1 -E 'topResumedActivity=|ResumedActivity:' || true)"
    if [[ "$focus" == *"$APP_ID"* ]]; then
        return 0
    fi

    focus="$(adb_cmd shell dumpsys window windows 2>/dev/null | tr -d '\r' | grep -m1 -E 'Window\{.*'"${APP_ID}"'' || true)"
    [[ "$focus" == *"$APP_ID"* ]]
}

wait_for_app_ready() {
    local retries="${1:-12}"
    local i
    for (( i=1; i<=retries; i++ )); do
        if is_app_foreground; then
            return 0
        fi
        if (( i % 3 == 0 )); then
            adb_cmd shell am start -n "${APP_ID}/${MAIN_ACTIVITY}" >/dev/null 2>&1 || true
        fi
        sleep 1
    done
    return 1
}

launch_app() {
    adb_cmd shell am force-stop "$APP_ID" >/dev/null 2>&1 || true
    adb_cmd shell am start -n "${APP_ID}/${MAIN_ACTIVITY}" >/dev/null
    sleep 2
    wait_for_app_ready 15 || die "Failed to bring ${APP_ID} to foreground."
}

dump_ui_xml_once() {
    adb_cmd shell uiautomator dump /sdcard/window_dump.xml >/dev/null 2>&1 || return 1
    adb_cmd shell cat /sdcard/window_dump.xml 2>/dev/null || return 1
}

dump_ui_xml() {
    local attempt xml
    for attempt in 1 2 3 4 5 6; do
        xml="$(dump_ui_xml_once || true)"
        if [[ -n "$xml" && "$xml" == *"package=\"$APP_ID\""* ]]; then
            printf '%s' "$xml"
            return 0
        fi
        sleep 1
    done
    return 1
}

ui_contains_text() {
    local needle="$1"
    local xml
    xml="$(dump_ui_xml || true)"
    [[ -n "$xml" ]] || return 1
    printf '%s' "$xml" | grep -Fq "$needle"
}

wait_for_text() {
    local needle="$1"
    local retries="${2:-12}"
    local i
    for (( i=1; i<=retries; i++ )); do
        if ui_contains_text "$needle"; then
            return 0
        fi
        sleep 1
    done
    return 1
}

find_center_by_label() {
    local label="$1"
    local xml node bounds

    xml="$(dump_ui_xml || true)"
    [[ -n "$xml" ]] || return 1

    node="$(printf '%s' "$xml" | tr '>' '\n' | grep -F "text=\"$label\"" | head -n 1 || true)"
    if [[ -z "$node" ]]; then
        node="$(printf '%s' "$xml" | tr '>' '\n' | grep -F "content-desc=\"$label\"" | head -n 1 || true)"
    fi
    [[ -n "$node" ]] || return 1

    bounds="$(printf '%s\n' "$node" | sed -E -n 's/.*bounds="\[([0-9]+),([0-9]+)\]\[([0-9]+),([0-9]+)\]".*/\1 \2 \3 \4/p')"
    [[ -n "$bounds" ]] || return 1

    local x1 y1 x2 y2 x y
    read -r x1 y1 x2 y2 <<<"$bounds"
    x=$(( (x1 + x2) / 2 ))
    y=$(( (y1 + y2) / 2 ))
    printf '%s %s\n' "$x" "$y"
}

tap_xy() {
    local x="$1"
    local y="$2"
    adb_cmd shell input tap "$x" "$y" >/dev/null
}

tap_xy_percent() {
    local x_pct="$1"
    local y_pct="$2"
    local x y
    x=$(( SCREEN_W * x_pct / 100 ))
    y=$(( SCREEN_H * y_pct / 100 ))
    tap_xy "$x" "$y"
}

tap_label() {
    local label="$1"
    local retries="${2:-5}"
    local quiet="${3:-0}"
    local i center x y
    for (( i=1; i<=retries; i++ )); do
        center="$(find_center_by_label "$label" || true)"
        if [[ -n "$center" ]]; then
            read -r x y <<<"$center"
            tap_xy "$x" "$y"
            return 0
        fi
        sleep 1
    done
    if [[ "$quiet" -ne 1 ]]; then
        warn "Could not locate label '$label'"
    fi
    return 1
}

tap_tab() {
    local tab="$1"
    local x_pct
    case "$tab" in
        Home) x_pct=10 ;;
        Train) x_pct=30 ;;
        Versus) x_pct=50 ;;
        Logs) x_pct=70 ;;
        Runs) x_pct=90 ;;
        *) warn "Unknown tab '$tab'"; return 1 ;;
    esac

    tap_label "$tab" 3 && return 0
    tap_xy_percent "$x_pct" 34
    sleep 1
    return 0
}

capture_png() {
    local name="$1"
    local path="${OUT_DIR}/${name}"
    adb_cmd exec-out screencap -p >"$path"
    if [[ ! -s "$path" ]]; then
        warn "Screenshot appears empty: $name"
    fi
    log "Captured $name"
}

set_screen_size() {
    local size_line width height fallback
    size_line="$(adb_cmd shell wm size 2>/dev/null | tr -d '\r' | grep -m1 -E 'Physical size: [0-9]+x[0-9]+' || true)"

    width=""
    height=""
    if [[ -n "$size_line" ]]; then
        width="$(printf '%s\n' "$size_line" | sed -E -n 's/.*Physical size: ([0-9]+)x([0-9]+).*/\1/p')"
        height="$(printf '%s\n' "$size_line" | sed -E -n 's/.*Physical size: ([0-9]+)x([0-9]+).*/\2/p')"
    fi

    if [[ -z "$width" || -z "$height" ]]; then
        fallback="$(adb_cmd shell dumpsys window displays 2>/dev/null | tr -d '\r' | grep -m1 -oE 'cur=[0-9]+x[0-9]+' || true)"
        if [[ -n "$fallback" ]]; then
            width="$(printf '%s\n' "$fallback" | sed -E -n 's/cur=([0-9]+)x([0-9]+)/\1/p')"
            height="$(printf '%s\n' "$fallback" | sed -E -n 's/cur=([0-9]+)x([0-9]+)/\2/p')"
        fi
    fi

    [[ "$width" =~ ^[0-9]+$ && "$height" =~ ^[0-9]+$ ]] || die "Failed to read device screen size."
    SCREEN_W="$width"
    SCREEN_H="$height"
    log "Screen size: ${SCREEN_W}x${SCREEN_H}"
}

swipe_up() {
    local duration_ms="${1:-420}"
    local x from_y to_y
    x=$(( SCREEN_W / 2 ))
    from_y=$(( SCREEN_H * 88 / 100 ))
    to_y=$(( SCREEN_H * 34 / 100 ))
    adb_cmd shell input swipe "$x" "$from_y" "$x" "$to_y" "$duration_ms" >/dev/null
}

start_recording() {
    local remote_file="$1"
    adb_cmd shell rm -f "/sdcard/${remote_file}" >/dev/null 2>&1 || true
    adb_cmd shell "screenrecord --bit-rate ${VIDEO_BITRATE} --time-limit ${VIDEO_SECONDS} /sdcard/${remote_file}" >/dev/null 2>&1 &
    RECORD_PID=$!
    log "Started screenrecord (pid=${RECORD_PID})"
}

stop_recording_and_pull() {
    local remote_file="$1"
    if [[ -n "$RECORD_PID" ]]; then
        wait "$RECORD_PID" || true
    fi
    adb_cmd pull "/sdcard/${remote_file}" "${OUT_DIR}/${remote_file}" >/dev/null
    log "Pulled video ${remote_file}"
}

write_report() {
    local report="${OUT_DIR}/QA_REPORT.txt"
    local model android sdk density version_name
    model="$(adb_cmd shell getprop ro.product.model | tr -d '\r')"
    android="$(adb_cmd shell getprop ro.build.version.release | tr -d '\r')"
    sdk="$(adb_cmd shell getprop ro.build.version.sdk | tr -d '\r')"
    density="$(adb_cmd shell wm density | tr -d '\r')"
    version_name="$(adb_cmd shell dumpsys package "$APP_ID" 2>/dev/null | tr -d '\r' | grep -m1 'versionName=' | cut -d'=' -f2 || true)"

    {
        echo "MobileFineTuner Visualizer - Device QA Report"
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Device serial: ${SERIAL}"
        echo "Model: ${model}"
        echo "Android: ${android} (SDK ${sdk})"
        echo "Density: ${density}"
        echo "App version: ${version_name:-unknown}"
        echo "Screen size: ${SCREEN_W}x${SCREEN_H}"
        echo "Output directory: ${OUT_DIR}"
        echo "Video seconds: ${VIDEO_SECONDS}"
        echo "Shot interval: ${SHOT_INTERVAL}"
        echo "Load demo: ${LOAD_DEMO}"
        echo
        echo "Warnings (${#WARNINGS[@]}):"
        if [[ ${#WARNINGS[@]} -eq 0 ]]; then
            echo "  - none"
        else
            local w
            for w in "${WARNINGS[@]}"; do
                echo "  - $w"
            done
        fi
        echo
        echo "Artifacts:"
        ls -1 "${OUT_DIR}" | sed 's/^/  - /'
    } >"${report}"

    log "Wrote ${report}"
}

main() {
    parse_args "$@"

    resolve_adb_bin
    require_cmd sed
    require_cmd awk
    require_cmd grep
    require_cmd tr
    require_cmd cut

    select_device
    ensure_device_ready

    if [[ -z "$OUT_DIR" ]]; then
        OUT_DIR="${PROJECT_ROOT}/verification/device_qa_$(date '+%Y%m%d_%H%M%S')"
    fi
    mkdir -p "$OUT_DIR"

    if ! [[ "$VIDEO_SECONDS" =~ ^[0-9]+$ ]] || [[ "$VIDEO_SECONDS" -le 0 ]] || [[ "$VIDEO_SECONDS" -gt 180 ]]; then
        die "--video-seconds must be an integer in [1, 180]"
    fi
    if ! [[ "$SHOT_INTERVAL" =~ ^[0-9]+$ ]] || [[ "$SHOT_INTERVAL" -le 0 ]]; then
        die "--shot-interval must be a positive integer"
    fi

    log "Using device serial: ${SERIAL}"
    log "Output directory: ${OUT_DIR}"
    log "Using adb binary: ${ADB_BIN}"

    if [[ "$SKIP_BUILD" -eq 0 ]]; then
        log "Building APK..."
        (cd "$PROJECT_ROOT" && ./gradlew :app:assembleDebug >/dev/null)
    fi

    local apk_path="${PROJECT_ROOT}/${APK_REL_PATH}"
    [[ -f "$apk_path" ]] || die "APK not found: ${apk_path}"

    if [[ "$SKIP_INSTALL" -eq 0 ]]; then
        log "Installing APK..."
        adb_cmd install -r "$apk_path" >/dev/null
    fi

    log "Launching app..."
    launch_app
    set_screen_size
    capture_png "00_launch.png"

    if [[ "$LOAD_DEMO" -eq 1 ]]; then
        if ! tap_label "Load Demo" 6; then
            tap_xy_percent 55 14
        fi
        if ! wait_for_text "demo/primary_gpt2" 18; then
            warn "Demo run did not appear within timeout."
        fi
        sleep 1
        capture_png "01_overview_demo.png"
    fi

    local chart_tap_x chart_tap_y
    chart_tap_x=$(( SCREEN_W * 76 / 100 ))
    chart_tap_y=$(( SCREEN_H * 56 / 100 ))

    local record_file
    record_file="training_session_$(date '+%Y%m%d_%H%M%S').mp4"

    if [[ "$ENABLE_RECORD" -eq 1 ]]; then
        start_recording "$record_file"

        local cycle=("Train" "Versus" "Logs" "Train" "Home")
        local start_ts now idx page
        start_ts="$(date +%s)"
        idx=0
        while true; do
            now="$(date +%s)"
            if [[ $(( now - start_ts )) -ge "$VIDEO_SECONDS" ]]; then
                break
            fi

            page="${cycle[$(( idx % ${#cycle[@]} ))]}"
            tap_tab "$page" || true
            sleep 1

            capture_png "$(printf 'record_%02d_%s.png' "$idx" "$(printf '%s' "$page" | tr '[:upper:]' '[:lower:]')")"
            if [[ "$page" == "Train" ]]; then
                tap_xy "$chart_tap_x" "$chart_tap_y"
                sleep 1
                capture_png "$(printf 'record_%02d_train_focus.png' "$idx")"
            fi

            idx=$(( idx + 1 ))
            sleep "$SHOT_INTERVAL"
        done

        stop_recording_and_pull "$record_file"
    fi

    tap_tab "Home"
    sleep 1
    capture_png "10_overview_final.png"

    tap_tab "Train"
    sleep 1
    capture_png "11_training_top.png"
    tap_xy "$chart_tap_x" "$chart_tap_y"
    sleep 1
    capture_png "12_training_focus.png"
    swipe_up 460
    sleep 1
    capture_png "13_training_mid.png"
    swipe_up 460
    sleep 1
    capture_png "14_training_lower.png"
    swipe_up 460
    sleep 1
    capture_png "15_training_bottom.png"

    tap_tab "Versus"
    sleep 1
    capture_png "20_versus_top.png"
    swipe_up 460
    sleep 1
    capture_png "21_versus_mid.png"
    swipe_up 460
    sleep 1
    capture_png "22_versus_lower.png"

    tap_tab "Logs"
    sleep 1
    capture_png "30_logs_top.png"
    if tap_label "Filter" 2 1; then
        adb_cmd shell input text Train >/dev/null 2>&1 || true
        sleep 1
        capture_png "31_logs_filter_train.png"
    fi
    swipe_up 460
    sleep 1
    capture_png "32_logs_scrolled.png"

    tap_tab "Runs"
    sleep 1
    capture_png "40_runs_top.png"
    swipe_up 460
    sleep 1
    capture_png "41_runs_scrolled.png"

    write_report
    log "QA run complete."
}

main "$@"
