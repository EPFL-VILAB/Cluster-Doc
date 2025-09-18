# ALPS documentation

---

## 1. Logging in the cluster

* **Setup MFA authentication:** Follow the provided [guide](https://docs.cscs.ch/).
* **SSH keys:** Use the provided [link](https://sshservice.cscs.ch/) to get SSH keys, which need to be refreshed every 24 hours.
* **Login:** `ssh -A <Username>@ela.cscs.ch`
* **Go to Clariden:** `ssh clariden`
* **Tutorial slides:** Use the [SwissAI workshop slides](https://docs.google.com/presentation/d/1IL8DvXee5s8IbyMRECcurJZaKJRs9QMz9s17j7Hpd_k/edit?slide=id.g2e1b744006e_0_47#slide=id.g2e1b744006e_0_47) for reference.

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
An example docker image to run flextok is in `/capstor/scratch/cscs/zgao/container/v3/flextok.toml`.
```
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

###  Debugging Job

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

srun --mem=460000 -ul bash -c "
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

You can set the `--environment` path to your own Docker image and modify the training commands inside the `srun` block. Then you can use `sbatch` command to submit the training jobs.




## 5. CSCS Storage

Your project ID is **a143**. For long-term file storage, use `/capstor/store/cscs/swissai/a143` and create a subfolder with your username. Be sure to check your storage usage with the `quota` command, as exceeding the limits on total size or the number of files can cause issues for the entire lab.

Each user also has a temporary scratch storage located at `$SCRATCH` (e.g., `/iopsstor/scratch/cscs/username` or `/capstor/scratch/cscs/username`), which is automatically cleaned every **30 days**. Don't use this for any files you need to keep long-term.

***

## 6. General Rules
* **Be a good neighbor**: Start with small-scale experiments first and run large-scale training only after debugging. Check your GPU hour usage [here](https://portal.cscs.ch/resource-details/e85760b210604cf9908bdffdffa489d7?tab=usage-history).
* **Watch GPU utilization**: Ensure your jobs maintain reasonably high GPU utilization. If not, release the GPU resources.
* **Getting Help**: 
  - CSCS [documentation](https://docs.cscs.ch/)
  - First, ask questions on the **#cluster** Slack channel where someone most likely knows the answer
  - Contact the cluster leads: **@Mingqiao** and **@Zhitong**
  - For technical support with RCP infrastructure, open a ticket at [support.cscs.ch](https://support.cscs.ch/)