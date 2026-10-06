#!/bin/bash
#SBATCH --job-name=cpu56_energy
#SBATCH --output=cpu56_energy_%A_%a.log
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=IscrC_FOCAL_0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=tmpfs:10g
#SBATCH --time=00:15:00
#SBATCH --profile=energy
#SBATCH --exclusive
#SBATCH --array=0-39%1

# Forty array tasks implement four configurations times ten independent
# sessions.  Each task starts one fresh benchmark process and executes ten
# searches of 10,000 queries, hence one monitored 100,000-query session.

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.7
module load gcc/12.2.0

SUBMIT_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}
source "${SUBMIT_DIR}/venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is not set}

NLISTS=(2048 2048 1024 512)
NPROBES=(4 16 32 64)
SESSIONS_PER_CONFIG=10
SEARCH_REPETITIONS=10

# Interleave configurations across array indices to reduce ordering effects.
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % ${#NLISTS[@]}))
SESSION_NUMBER=$((SLURM_ARRAY_TASK_ID / ${#NLISTS[@]} + 1))
NLIST=${NLISTS[$CONFIG_INDEX]}
NPROBE=${NPROBES[$CONFIG_INDEX]}

if (( SESSION_NUMBER > SESSIONS_PER_CONFIG )); then
    echo "ERROR: array index maps to an invalid session" >&2
    exit 1
fi

BENCHMARK_SCRIPT="${SUBMIT_DIR}/run_faiss_benchmark_energy.py"
DATASET="${SUBMIT_DIR}/glove-100-angular.hdf5"
CENTROIDS="${SUBMIT_DIR}/centroids-glove-100-angular-${NLIST}.npy"

for required_file in "$BENCHMARK_SCRIPT" "$DATASET" "$CENTROIDS"; do
    if [[ ! -r "$required_file" ]]; then
        echo "ERROR: required file is not readable: $required_file" >&2
        exit 1
    fi
done

ENERGY_ROOT="${SUBMIT_DIR}/energy_raw/cpu_56/job_${SLURM_ARRAY_JOB_ID}"
SESSION_DIR="${ENERGY_ROOT}/nlist${NLIST}_nprobe${NPROBE}/session_${SESSION_NUMBER}_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SESSION_DIR"

# The benchmark resolves the saved centroid filename in its current directory.
# A per-session directory also prevents concurrent result CSV writes.
ln -s "$CENTROIDS" "${SESSION_DIR}/$(basename "$CENTROIDS")"

RAPL_FILES=()
RAPL_NAMES=()
RAPL_RANGES=()

for zone_dir in /sys/class/powercap/intel-rapl:*; do
    [[ -e "$zone_dir" ]] || continue
    zone_base=${zone_dir##*/}
    [[ "$zone_base" =~ ^intel-rapl:[0-9]+$ ]] || continue

    energy_file="${zone_dir}/energy_uj"
    name_file="${zone_dir}/name"
    range_file="${zone_dir}/max_energy_range_uj"
    [[ -r "$energy_file" && -r "$name_file" ]] || continue

    IFS= read -r zone_name < "$name_file"
    [[ "$zone_name" == package-* ]] || continue

    max_range=""
    if [[ -r "$range_file" ]]; then
        IFS= read -r max_range < "$range_file"
    fi

    RAPL_FILES+=("$energy_file")
    RAPL_NAMES+=("$zone_name")
    RAPL_RANGES+=("$max_range")
done

if (( ${#RAPL_FILES[@]} == 0 )); then
    echo "ERROR: no readable top-level RAPL package counters were found" >&2
    exit 1
fi

RAPL_LOG="${SESSION_DIR}/host_rapl.csv"
RAPL_METADATA="${SESSION_DIR}/rapl_zones.csv"
EVENT_LOG="${SESSION_DIR}/events.csv"

{
    printf 'Column,ZoneName,EnergyPath,MaxEnergyRange_uJ\n'
    for index in "${!RAPL_FILES[@]}"; do
        printf 'zone%d_energy_uJ,%s,%s,%s\n' \
            "$index" "${RAPL_NAMES[$index]}" "${RAPL_FILES[$index]}" \
            "${RAPL_RANGES[$index]}"
    done
} > "$RAPL_METADATA"

{
    printf 'Timestamp'
    for index in "${!RAPL_FILES[@]}"; do
        printf ',zone%d_energy_uJ' "$index"
    done
    printf '\n'
} > "$RAPL_LOG"

printf 'Event,Timestamp\n' > "$EVENT_LOG"

record_event() {
    printf '%s,%s\n' "$1" "$(date +%s.%N)" >> "$EVENT_LOG"
}

monitor_rapl() {
    local timestamp line value energy_file
    local row_valid

    while true; do
        timestamp=$(date +%s.%N)
        line=$timestamp
        row_valid=true

        for energy_file in "${RAPL_FILES[@]}"; do
            if IFS= read -r value < "$energy_file"; then
                line+=",${value}"
            else
                row_valid=false
                break
            fi
        done

        if [[ "$row_valid" == true ]]; then
            printf '%s\n' "$line" >> "$RAPL_LOG"
        fi
        sleep 0.1
    done
}

MONITOR_PIDS=()

cleanup_monitors() {
    local pid
    for pid in "${MONITOR_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    MONITOR_PIDS=()
}

trap cleanup_monitors EXIT
trap 'exit 130' INT TERM

{
    echo "JobID=${SLURM_JOB_ID}"
    echo "ArrayJobID=${SLURM_ARRAY_JOB_ID}"
    echo "ArrayTaskID=${SLURM_ARRAY_TASK_ID}"
    echo "Hostname=$(hostname)"
    echo "NList=${NLIST}"
    echo "NProbe=${NPROBE}"
    echo "Session=${SESSION_NUMBER}"
    echo "SearchRepetitions=${SEARCH_REPETITIONS}"
    echo "QueriesPerRepetition=10000"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    echo "CpusAllowed=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
    echo
    lscpu
    echo
    scontrol --oneliner show job "$SLURM_JOB_ID"
} > "${SESSION_DIR}/environment.txt"

echo "Starting CPU-56 energy session"
echo "Configuration: NList=${NLIST}, NProbe=${NPROBE}"
echo "Independent session: ${SESSION_NUMBER}/${SESSIONS_PER_CONFIG}"
echo "Output directory: ${SESSION_DIR}"
printf 'RAPL zones:'
printf ' %s' "${RAPL_NAMES[@]}"
printf '\n'

record_event monitor_start
monitor_rapl &
MONITOR_PIDS+=("$!")

# Retain samples on both sides of the exact benchmark process boundary.
sleep 0.3
record_event benchmark_start

cd "$SESSION_DIR"
set +e
python3 -u "$BENCHMARK_SCRIPT" \
    --dataset "$DATASET" \
    --run_id "cpu56_nlist${NLIST}_nprobe${NPROBE}_session${SESSION_NUMBER}" \
    --nlist "$NLIST" \
    --nprobe "$NPROBE" \
    --k 10 \
    --runs "$SEARCH_REPETITIONS" \
    > "${SESSION_DIR}/benchmark.log" 2>&1
BENCHMARK_STATUS=$?
set -e

record_event benchmark_end
sleep 0.3
cleanup_monitors
record_event monitor_end

if (( BENCHMARK_STATUS != 0 )); then
    echo "ERROR: benchmark exited with status ${BENCHMARK_STATUS}" >&2
    exit "$BENCHMARK_STATUS"
fi

RAPL_SAMPLE_COUNT=$(( $(wc -l < "$RAPL_LOG") - 1 ))
if (( RAPL_SAMPLE_COUNT < 2 )); then
    echo "ERROR: fewer than two RAPL samples were recorded" >&2
    exit 1
fi

echo "Completed CPU-56 energy session with ${RAPL_SAMPLE_COUNT} RAPL samples"
