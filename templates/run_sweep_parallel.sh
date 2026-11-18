#!/bin/bash
cd "$(dirname "$0")"
JOBS=${1:-$(nproc)}
folders=( */ )
folders=( "${folders[@]%/}" )

trap 'trap - INT TERM; kill 0; exit 130' INT TERM

ls -d */ | xargs -P "$JOBS" -I{} bash -c 'cd "{}" && bash run_local.sh' &
sweep_pid=$!

show_status() {
    printf '\033[H\033[J'
    printf '%-40s %-16s %-10s %-12s %s\n' run progress pos_acc E/N state
    for d in "${folders[@]}"; do
        label="$d"
        log="$d/run.log"
        progress=-; pos=-; energy=-
        if [[ -f "$log" ]]; then
            line=$(grep '^Step ' "$log" | tail -n 1)
            if [[ -n "$line" ]]; then
                read -r _ frac <<< "$line"
                progress=$(awk -F/ '{printf "%.1f%%", 100*$1/$2}' <<< "$frac")
            fi
            pos_line=$(grep 'Position acceptance rate so far:' "$log" | tail -n 1)
            [[ -n "$pos_line" ]] && pos=$(awk '{print $NF}' <<< "$pos_line")
        fi
        if [[ -f "$log" ]] && grep -q '^Measurements = ' "$log"; then
            state=done
            energy=$(awk '/^E_VMC_per_particle/ {print $2}' "$d/VMC/results.txt")
        elif [[ ! -f "$log" ]]; then
            state=queued
        elif kill -0 "$sweep_pid" 2>/dev/null; then
            state=running
        else
            state=failed
        fi
        printf '%-40s %-16s %-10s %-12s %s\n' "$label" "$progress" "$pos" "$energy" "$state"
    done
}

while kill -0 "$sweep_pid" 2>/dev/null; do
    show_status
    sleep 5
done
show_status
wait "$sweep_pid"
rc=$?
echo "Sweep finished (exit code $rc)."
exit $rc
