#!/bin/bash
#SBATCH --job-name=a100_energy
#SBATCH --output=a100_energy_%A_%a.log
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_FOCAL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --array=0-39%1

# Forty array tasks implement four configurations times ten independent
# sessions.  Each task starts one fresh benchmark process and executes ten
# searches of 10,000 queries, hence one monitored 100,000-query session.

set -euo pipefail

module purge
module load profile/deeplrn
module load python/3.11.7
module load cuda/12.2
module load gcc/12.2.0

SUBMIT_DIR=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}
source "${SUBMIT_DIR}/venv/bin/activate"

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cublas/lib:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages:${PYTHONPATH:-}"

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

ENERGY_ROOT="${SUBMIT_DIR}/energy_raw/gpu_a100/job_${SLURM_ARRAY_JOB_ID}"
SESSION_DIR="${ENERGY_ROOT}/nlist${NLIST}_nprobe${NPROBE}/session_${SESSION_NUMBER}_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SESSION_DIR"

# The benchmark resolves the saved centroid filename in its current directory.
# A per-session directory also prevents concurrent result CSV writes.
ln -s "$CENTROIDS" "${SESSION_DIR}/$(basename "$CENTROIDS")"

GPU_INDEX=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed -n '1p')
GPU_INDEX=${GPU_INDEX//[[:space:]]/}
if [[ -z "$GPU_INDEX" ]]; then
    echo "ERROR: no allocated GPU was visible through nvidia-smi" >&2
    exit 1
fi

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
    echo "ERROR: no readable top-level host RAPL package counters were found" >&2
    exit 1
fi

GPU_LOG="${SESSION_DIR}/gpu_power.csv"
RAPL_LOG="${SESSION_DIR}/host_rapl.csv"
RAPL_METADATA="${SESSION_DIR}/rapl_zones.csv"
EVENT_LOG="${SESSION_DIR}/events.csv"

printf 'Timestamp,Power_W\n' > "$GPU_LOG"
printf 'Event,Timestamp\n' > "$EVENT_LOG"

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

record_event() {
    printf '%s,%s\n' "$1" "$(date +%s.%N)" >> "$EVENT_LOG"
}

monitor_gpu() {
    local timestamp power_w

    while true; do
        power_w=$(nvidia-smi \
            --id="$GPU_INDEX" \
            --query-gpu=power.draw \
            --format=csv,noheader,nounits 2>/dev/null || true)
        timestamp=$(date +%s.%N)

        if [[ "$power_w" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            printf '%s,%s\n' "$timestamp" "$power_w" >> "$GPU_LOG"
        fi
        sleep 0.05
    done
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
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
    echo "CpusAllowed=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
    echo
    lscpu
    echo
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit --format=csv
    echo
    scontrol --oneliner show job "$SLURM_JOB_ID"
} > "${SESSION_DIR}/environment.txt"

echo "Starting A100 energy session"
echo "Configuration: NList=${NLIST}, NProbe=${NPROBE}"
echo "Independent session: ${SESSION_NUMBER}/${SESSIONS_PER_CONFIG}"
echo "Output directory: ${SESSION_DIR}"
printf 'Host RAPL zones:'
printf ' %s' "${RAPL_NAMES[@]}"
printf '\n'

record_event monitor_start
monitor_gpu &
MONITOR_PIDS+=("$!")
monitor_rapl &
MONITOR_PIDS+=("$!")

# Retain samples on both sides of the exact benchmark process boundary.
sleep 0.3
record_event benchmark_start

cd "$SESSION_DIR"
set +e
python3 -u "$BENCHMARK_SCRIPT" \
    --dataset "$DATASET" \
    --run_id "gpu_nlist${NLIST}_nprobe${NPROBE}_session${SESSION_NUMBER}" \
    --nlist "$NLIST" \
    --nprobe "$NPROBE" \
    --k 10 \
    --runs "$SEARCH_REPETITIONS" \
    --gpu \
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

GPU_SAMPLE_COUNT=$(( $(wc -l < "$GPU_LOG") - 1 ))
RAPL_SAMPLE_COUNT=$(( $(wc -l < "$RAPL_LOG") - 1 ))

if (( GPU_SAMPLE_COUNT < 2 )); then
    echo "ERROR: fewer than two GPU power samples were recorded" >&2
    exit 1
fi
if (( RAPL_SAMPLE_COUNT < 2 )); then
    echo "ERROR: fewer than two host RAPL samples were recorded" >&2
    exit 1
fi

echo "Completed A100 energy session"
echo "GPU samples: ${GPU_SAMPLE_COUNT}"
echo "Host RAPL samples: ${RAPL_SAMPLE_COUNT}"
