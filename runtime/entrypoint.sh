#!/usr/bin/env bash

set -euxo pipefail

log() {
    set +x
    local level="$1"; shift
    printf '%s | %-4s | %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S.%3N')" \
        "$level" \
        "$*"
    set -x
}

main () {
    expected_filename=main.py

    cd /code_execution

    track="${KIDSASR_TRACK:-unknown}"
    log INFO "Track is: $track"
    if [ "${KIDSASR_IS_SMOKE:-0}" -eq 1 ]; then
        log INFO "This is a smoke test run."
    fi

    # Check that expected entrypoint script exists
    submission_files=$(zip -sf ./submission/submission.zip)
    if ! grep -F -q -- "$expected_filename" <<<"$submission_files"; then
        log ERROR "Submission zip archive must include $expected_filename";
        return 1;
    fi

    log INFO "Unpacking submission into src/..."
    unzip ./submission/submission.zip -d ./src

    # In the actual runtime, a tar archive of the audio files will be
    # mounted at ./data-archive/ and then unpacked into ./data/
    if [ "${KIDSASR_UNPACK_DATA:-0}" -eq 1 ]; then
        log INFO "Unpacking data archive into data/..."
        mkdir -p ./data
        cp ./data-archive/submission_format.jsonl ./data/
        cp ./data-archive/utterance_metadata.jsonl ./data/
        tar -xf ./data-archive/audio.tar -C ./data
    fi

    log INFO "Showing current working directory contents:"
    ls -alh

    log INFO "Showing data/ directory contents:"
    ls -alh data/

    log INFO "Showing src/ directory contents:"
    find src/

    log INFO "Running submission..."

    uv run src/main.py
}

main |& tee "/code_execution/submission/log.txt"
exit_code=${PIPESTATUS[0]}

cp /code_execution/submission/log.txt /tmp/log

log INFO "Submission run completed with exit code: $exit_code"

exit $exit_code
