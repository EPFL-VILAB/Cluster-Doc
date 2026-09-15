# ALPS documentation

---

## 1. Logging in the cluster

* **Setup MFA authentication:** Follow the provided [guide](https://docs.cscs.ch/).
* **SSH keys:** Use the provided [link](https://sshservice.cscs.ch/) to get SSH keys, which need to be refreshed every 24 hours.
* **Login:** `ssh -A <Username>@ela.cscs.ch`
* **Go to Clariden:** `ssh clariden`
* **Swiss AI User Day slides:** See the [2026 Swiss AI User Day documentation](https://eth-cscs.github.io/2026-swiss-ai-yearly-meeting/).
* **Quick start:** A concise cluster quick-start guide is also available [here](https://hackmd.io/@WUI3j853QwWKd7Vx6XfwGw/ByP3w4e7be).

---

## 2. Getting an interactive session

* **Command:** `srun -A a143 --pty bash`
* **Debug job:** `srun -A a143 -p debug --pty bash`
* **Duration:** This creates an interactive session that's valid for 1 hour.
* **Checking running jobs:** `squeue -l --me`
* **Killing a job:** `scancel <JOBID>`
* **Checking all nodes:** `sinfo`

---

## 3. Building a Docker image

The cluster is ARM-based, so we use a Docker image to run training jobs.

### Use pre-built image

An example Docker image to run Flextok is in `/capstor/scratch/cscs/zgao/container/v3/flextok.toml`.

```bash
srun -A a143 -p debug --environment=/capstor/scratch/cscs/zgao/container/v3/flextok.toml --pty bash
cd /capstor/scratch/cscs/<your_user_name>
```

### General Steps

* **Get an interactive node:**
  * `srun -A a143 --pty bash` (creates a 1-hour job and connects in one step)
* **Navigate to the Dockerfile directory:** `cd /store/swissai/a143/containers/<IMAGE_DIR>`
* **Build the image:** `podman build -t <IMAGE_NAME>.`
* **Compress the image:** `enroot import -x mount -o <IMAGE_NAME>.sqsh podman://<IMAGE_NAME>`
* **Set up toml file:** example in `/capstor/scratch/cscs/zgao/container/v3/flextok.toml`
* **Exit:** `exit` (to go back to the login node)

### Example - Building Docker on Clariden

* **Navigate to directory:** `cd /capstor/store/cscs/swissai/a143/containers/flextok/v3/`
* **Create a copy:** Don't edit files directly. Make a copy of `build_script.sh` and `Dockerfile`.
* **Run build script:** `sh build_script.sh`
* **Configure Podman:** When Vim pops up, paste the following and save:

  ```ini
  [storage]
  driver = "overlay"
  runroot = "/tmp/$USER/podman_runroot"
  graphroot = "/tmp/$USER/podman_graphroot"
  ```

---

## 4. sbatch scripts (Update June - 2025)

* **Project ID:** `a143`

### Debugging Job

You can obtain a debug node for testing your code by using the following `srun` command:

```bash
srun -A a143 -p debug -t 90 --environment=/capstor/scratch/cscs/zgao/container/v3/flextok.toml --pty bash
```

Once your job starts running, you can connect to the debug node via SSH with this command:

```bash
srun --interactive --jobid <replace-this-with-your-job-id> --environment=/capstor/scratch/cscs/zgao/container/v3/flextok.toml --pty bash
```

### Multi-node training job

```bash
#!/bin/bash
#SBATCH --job-name=train         # create a short name for your job
#SBATCH --time=12:00:00
#SBATCH --nodes=2                # total number of nodes
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=450GB
#SBATCH --output=logs/%x_%j.log  # control where the stdout will be
#SBATCH --error=logs/%x_%j.err   # control where the error messages will be
#SBATCH --account=a143
#SBATCH --environment=/capstor/scratch/cscs/zgao/container/v3/flextok.toml

# Initialization.

export MASTER_PORT=25678
export MASTER_ADDR=$(hostname)

srun --cpu-bind=none -ul bash -c "
  # Change cwd and run the main training script.
  cd /iopsstor/scratch/cscs/mkhattak/june_experiments/proper_scaling_experiments/J12
  pip install wandb[media]
  TORCHRUN_ARGS=\"
   --node-rank=${SLURM_PROCID} \\
   --master-addr=${MASTER_ADDR} \\
   --master-port=${MASTER_PORT} \\
   --nnodes=${SLURM_NNODES} \\
   --nproc-per-node=${SLURM_GPUS_PER_NODE} \\
  \"

OMP_NUM_THREADS=1 && NCCL_P2P_DISABLE=1 torchrun ${TORCHRUN_ARGS} run_training_4m_fsdp.py --config cfgs/default/4m/models/main/4m_large_depth_rgb_normal_caption.yaml --output_dir J12 --wandb_run_name J12
```

You can set the `--environment` path to your own Docker image and modify the training commands inside the `srun` block. Then you can use the `sbatch` command to submit the training jobs.

---

## 5. CSCS Storage

Your project ID is **a143**. For long-term file storage, use `/capstor/store/cscs/swissai/a143` and create a subfolder with your username. Be sure to check your storage usage with the `quota` command, as exceeding the limits on total size or the number of files can cause issues for the entire lab.

Each user also has a temporary scratch storage located at `$SCRATCH` (e.g., `/iopsstor/scratch/cscs/username` or `/capstor/scratch/cscs/username`), which is automatically cleaned every **30 days**. Don't use this for any files you need to keep long-term.

---

## 6. GPU and Node Usage

There are two convenient ways to check GPU and node usage.

### Command line

The following script reports the total GPU-hours and node-hours for a given user under a specific Slurm account, together with a breakdown by job name.

Save it as `count_gpus.sh`:

```bash
#!/bin/bash

# Usage:
#   ./count_gpus.sh <account> [username] [starttime]
#
# If username is not provided, use current user.
# If starttime is not provided, default to 2025-10-01.

ACCOUNT="$1"
USERNAME="${2:-${USER:-$(whoami)}}"
STARTTIME="${3:-2025-10-01}"

if [ -z "$ACCOUNT" ]; then
    echo "Usage: $0 <account> [username] [starttime]"
    exit 1
fi

sacct \
    -u "$USERNAME" \
    -A "$ACCOUNT" \
    -X \
    -D \
    -n \
    --starttime="$STARTTIME" \
    --format=JobName,ElapsedRaw,AllocTRES%200 \
    -P |
awk -F'|' -v user="$USERNAME" -v account="$ACCOUNT" -v start="$STARTTIME" '
{
    gpu=0
    node=0

    if (match($3, /(^|,)(gpu|gres\/gpu)=([0-9]+)/, m))
        gpu=m[3]+0
    if (match($3, /(^|,)node=([0-9]+)/, n))
        node=n[2]+0

    t = ($2 == "" ? 0 : $2) / 3600
    tag = ($1 == "" ? "(none)" : $1)

    gh = gpu * t
    nh = node * t

    gpu_h[tag]  += gh
    node_h[tag] += nh
    total_gpu_h  += gh
    total_node_h += nh
}
END {
    printf "User:             %s\n", user
    printf "Account:          %s\n", account
    printf "Start date:       %s\n", start
    printf "Total GPU-hours:  %.2f\n", total_gpu_h
    printf "Total Node-hours: %.2f\n", total_node_h
    printf "\n"
    printf "%-30s %-12s %-12s\n", "JobName", "GPU-hours", "Node-hours"

    for (i in gpu_h)
        printf "%-30s %-12.2f %-12.2f\n", i, gpu_h[i], node_h[i]
}'
```

Usage:

```bash
./count_gpus.sh <account> [username] [start_date]
```

If `username` is not provided, the current user is used. If `start_date` is not provided, it defaults to `2025-10-01`.

### CSCS Portal

You can also check project usage through the [CSCS Portal](https://portal.cscs.ch/):

1. Log in to the [CSCS Portal](https://portal.cscs.ch/).
2. Open [Organizations](https://portal.cscs.ch/organizations/).
3. Select [SwissAI Initiative](https://portal.cscs.ch/organizations/6d66bda48e704c6b82822fd0b2316b01/dashboard/).
4. Open [Projects](https://portal.cscs.ch/organizations/6d66bda48e704c6b82822fd0b2316b01/projects/).
5. Select [ab037](https://portal.cscs.ch/projects/d94af76f0d8e46e9a41cf3a44e6e4439/).
6. Open [HPC](https://portal.cscs.ch/projects/d94af76f0d8e46e9a41cf3a44e6e4439/resources/).
7. Select `swissai-ab037-clariden-on-alpssvg`.
8. Open the **Usage** view to check the project usage.

---

## 7. General Rules

* **Be a good neighbor**: Start with small-scale experiments first and run large-scale training only after debugging. Check your GPU-hour usage using the methods above.
* **Watch GPU utilization**: Ensure your jobs maintain reasonably high GPU utilization. If not, release the GPU resources.
* **Getting Help**:
  - CSCS [documentation](https://docs.cscs.ch/)
  - First, ask questions on the **#cluster** Slack channel where someone most likely knows the answer
  - Contact the cluster leads: **@Mingqiao** and **@Zhitong**
  - For technical support with RCP infrastructure, open a ticket at [support.cscs.ch](https://support.cscs.ch/)
