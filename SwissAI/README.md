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

* **Command:** `srun --pty bash`
* **Duration:** This creates an interactive session that's valid for 1 hour.
* **Checking running jobs:** `squeue -l --me`
* **Killing a job:** `scancel <JOBID>`
* **Checking all nodes:** `sinfo`

---

## 3. Building a Docker image

### General Steps

* **Get an interactive node:**
    * `salloc -t240` (allocates a node for 4 hours, note the job ID)
    * `srun --overlap --jobid <JOBID> --pty bash` (connects to the allocated node)
    * **Alternatively:** `srun --pty bash` (creates a 1-hour job and connects in one step)
* **Navigate to the Dockerfile directory:** `cd /store/swissai/a08/containers/<IMAGE_DIR>`
* **Build the image:** `podman build -t <IMAGE_NAME>.`
* **Compress the image:** `enroot import -x mount -o <IMAGE_NAME>.sqsh podman://<IMAGE_NAME>`
* **Exit:** `exit` (to go back to the login node)

### Example - Using Docker on Clariden

* **Navigate to directory:** `cd /capstor/store/cscs/swissai/a143/uzair/uzair_containers/video_env/`
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

## 4. Running 4M demo

* **Get interactive node:** `salloc -t240` (for 4 hours)
* **Connect to node:** `srun --overlap --jobid <JOBID> --environment=/store/swissai/a08/containers/4m_embodied/4m_embodied_check.toml --container-workdir=$PWD --pty bash`
* **Navigate:** `cd /workspace/ml-4m`
* **Run demo code:**
    ```python
    from fourm.demo_4M_sampler import Demo4MSampler, img_from_url
    sampler = Demo4MSampler(fm='EPFL-VILAB/4M-21_XL').cuda()
    img = img_from_url('[https://storage.googleapis.com/four_m_site/images/demo_rgb.png](https://storage.googleapis.com/four_m_site/images/demo_rgb.png)')
    preds = sampler({'rgb@224': img.cuda()}, seed=None)
    ```
* **Note:** If you just want to run 4M code, ask Kunal for a pre-built docker image path.

---

### Running 4M training

* **If `torchrun` doesn't work:**
    `OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node 4 --use-env run_training_4m.py --config ./cfgs/ssl/main/4m-7/ade20k/rgb/4m-b_beit-rgb_70b_lre-05.yaml --finetune ../four_checkpoints/4m-B-cc12m.safetensors --no_log_wandb`
* **If `torchrun` works (with latest image):**
    `OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 run_training_4m.py --config ./cfgs/ssl/main/4m-7/ade20k/rgb/4m-b_beit-rgb_70b_lre-05.yaml --finetune ../four_checkpoints/4m-B-cc12m.safetensors --no_log_wandb`
* **Training jobs:** Ask Kunal for an `sbatch` script.

---

## 5. sbatch scripts (Update June - 2025)

* **Project ID:** `a143`

### Multi-node training job

```bash
#!/bin/bash
#SBATCH --job-name=video-modeling      # create a short name for your job
#SBATCH --nodes=64                # total number of nodes
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log  # control where the stdout will be
#SBATCH --error=logs/%x_%j.err   # control where the error messages will be
#SBATCH --account=a-a143

# Initialization.

export MASTER_PORT=25678

export MASTER_ADDR=$(hostname)

srun --mem=460000 -ul --environment=/capstor/store/cscs/swissai/a08/containers/uzair_containers/video_env/video.toml bash -c "
#srun --mem=460000 -ul bash -c
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

You can change the --environment variable with your own docker image file path.


###  Debugging Job

You can obtain a debug node for testing your code by using the following `sbatch` script:

```bash
#!/bin/bash
#SBATCH --job-name=video-modeling   # create a short name for your job
#SBATCH --nodes=1        # total number of nodes
#SBATCH --ntasks-per-node=1   # total number of tasks per node
#SBATCH --gpus-per-node=1
#SBATCH --time=00:59:59
#SBATCH --output=logs/%x_%j.log # control where the stdout will be
#SBATCH --error=logs/%x_%j.err  # control where the error messages will be
#SBATCH --account=a-a08
#SBATCH --partition debug
# Initialization.
###/capstor/store/cscs/swissai/a08/containers/uzair_containers/video_env/pseudolabeling/normal.toml
###/capstor/store/cscs/swissai/a08/containers/uzair_containers/video_env/video.toml
#srun --mem=460000 -ul bash -c "
srun --mem=460000 -ul --environment=/capstor/store/cscs/swissai/a08/containers/uzair_containers/video_env/video.toml bash -c "
pip install imageio==2.19.3
pip install imageio-ffmpeg==0.4.7
pip install huggingface-hub==0.25.2
sleep 500m
```

Once your job starts running, you can connect to the debug node via SSH with this command:
```bash
srun --interactive --jobid <replace-this-with-your-job-id> --environment=/capstor/store/cscs/swissai/a08/containers/uzair_containers/video_env/video.toml --pty bash
```

## 6. CSCS Storage

Your project ID is **a143**. For long-term file storage, use `/capstor/store/cscs/swissai/a143` and create a subfolder with your username. Be sure to check your storage usage with the `quota` command, as exceeding the limits on total size or the number of files can cause issues for the entire lab.

Each user also has a temporary scratch storage located at `$SCRATCH` (e.g., `/iopsstor/scratch/cscs/username`), which is automatically cleaned every **30 days**. Don't use this for any files you need to keep long-term.

***
