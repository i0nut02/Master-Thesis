#!/bin/bash
#SBATCH --job-name=check_cpu_boost_gpu
#SBATCH --output=check_cpu_boost_gpu_%j.log
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_FOCAL
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:05:00

set -eu

echo "=== Slurm allocation ==="
echo "Job ID:              ${SLURM_JOB_ID:-not-set}"
echo "Hostname:            $(hostname)"
echo "Partition:           ${SLURM_JOB_PARTITION:-not-set}"
echo "CPUs per task:       ${SLURM_CPUS_PER_TASK:-not-set}"
echo "CPUs on allocated node: ${SLURM_JOB_CPUS_PER_NODE:-not-set}"
echo "Visible GPU IDs:     ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Batch-shell affinity:"
grep -E 'Cpus_allowed|Mems_allowed' /proc/self/status

echo
echo "=== CPU topology of the GPU node ==="
lscpu | grep -E '^(Architecture|CPU\(s\)|On-line CPU|Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)|NUMA node[0-9]+ CPU\(s\)):'

echo
echo "=== CPUs visible to this job ==="
echo "nproc: $(nproc)"

if command -v numactl >/dev/null 2>&1; then
    echo
    echo "=== NUMA policy ==="
    numactl --show
fi

echo
echo "=== Allocated GPU ==="
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit --format=csv,noheader

echo
echo "=== Slurm job record ==="
scontrol --oneliner show job "$SLURM_JOB_ID"
