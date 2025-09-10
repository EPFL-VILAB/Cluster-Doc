# RCP Cluster Guide 🚀

The RCP cluster [[Introduction1](https://www.epfl.ch/research/facilities/rcp/rcp-cluster-service-description/)] [[Introduction2](https://wiki.rcp.epfl.ch/en/home/CaaS)] is our main EPFL research hub, packed with powerful **H100**, **A100** and **V100** GPUs and managed by the **Run:AI** scheduler. This guide covers everything you need to get started.

---

## 1. Getting Access

To get on the cluster, you must first be on the access list.

1.  **Check the Prerequisites**: Ensure you have your **Gaspar credentials**, have the **EPFL VPN** set up, and a working EPFL email.
2.  **Request Access**: Send an email to **Stéphanie** (or **Amir** if she's unavailable). Ask them to add you to the `runai-vilab` list in [groups.epfl.ch](https:/groups.epfl.ch).
3.  **Wait**: Access is typically granted within two hours after you're added to the list.

---

## 2. Setting Up Your Local Environment

You'll need a couple of tools on your computer to interact with the cluster. 

### Install `Run:AI` CLI
You can follow instructions on [How to prepare your environment](https://wiki.rcp.epfl.ch/home/CaaS/FAQ/how-to-prepare-environment). Here are examples for Mac.
1.  **Download `Run:AI`**
    ```bash
    curl -sLo /tmp/runai https://rcp-caas-prod.rcp.epfl.ch/cli/darwin
    sudo install /tmp/runai /usr/local/bin/runai
    ```
2. Web Interface Login

    To login into Run:AI application open http://rcpepfl.run.ai in the browser and click on Sign in with SSO.

### Test Your Setup
You can follow instructions on [How to use RunAI](https://wiki.rcp.epfl.ch/home/CaaS/FAQ/how-to-use-runai). Here are examples for Mac.

1. Configuration file

    Download the file using wget/curl or copy/paste the content below into ```.kube/config```.
    ```bash
    curl https://wiki.rcp.epfl.ch/public/files/kube-config.yaml -o ~/.kube/config && chmod 600 ~/.kube/config
    # OR
    wget https://wiki.rcp.epfl.ch/public/files/kube-config.yaml -O ~/.kube/config && chmod 600 ~/.kube/config
    ```
    
    Once the configuration file has been created, you have to set the default cluster with the following commands:
    ```bash
    # With runai CLI
    runai config cluster rcp-caas-prod
    ```

2. Login
    To login, use following command.
    ```bash
    runai login
    ```

3. Config default project
    List projects you are belong to:
    ```bash
    runai list project
    ```
    
    Select your default project:
    ```bash
    runai config project vilab-<Gaspar Name>
    ```


1.  Run a simple test job:
    ```bash
    runai submit --name test -i ubuntu -g 0.5 --interactive --command -- sleep 300
    ```
2.  List your jobs to see it running:
    ```bash
    runai list
    ```
    You should also see it on the [Run:AI dashboard](https://rcpepfl.run.ai/dashboards/now).
3.  Delete the test job:
    ```bash
    runai delete job <JOB_NAME>
    ```
    *The `JOB_NAME` is found in the output of `runai list`.*

### A Note on Run:AI Scheduling
Run:AI ensures a fair distribution of GPU resources. It distinguishes between **`train`** and **`interactive`** jobs.
* **Preemption**: If a lab goes over its GPU quota, the scheduler can stop a `train` job to free up resources. The job will restart automatically later.
* **No Preemption for Interactive Jobs**: `Interactive` jobs are not preempted but are limited to 12 hours to prevent resource hogging.

---

## 3. Creating Your Docker Image

A Docker image is a self-contained package that includes all your code, dependencies, and settings. You can follow instructions on [How to build a container](https://wiki.rcp.epfl.ch/en/home/CaaS/FAQ/how-to-build-a-container-part1) and [How to use RCP registry](https://wiki.rcp.epfl.ch/en/home/CaaS/FAQ/how-to-registry) to prepare the environment. Here are examples for Mac.

1.  **Get the Dockerfile**: Ask a PhD student to invite you to the GitHub repository with the example `Dockerfile`.
2.  **Find your User ID**: You need this for user permissions within the container. It's on your EPFL directory page under "Administrative data." If you don't have a page, email IC support.
3.  **Log in to registries**:
    * Log in to [NVIDIA NGC](https://ngc.nvidia.com/) to pull base images.
    * Log in to the IC Registry where you'll upload your image:
        ```bash
        docker login registry.rcp.epfl.ch
        ```
4.  **Build your image**: In the directory with your `Dockerfile`, run:
    ```bash
    docker build . -t registry.rcp.epfl.ch/vil/YOUR_IMAGE_NAME
    ```
    * **Apple Silicon (M1/M2) users**: Add a platform flag for compatibility:
        ```bash
        docker build --platform linux/x86_64 . -t registry.rcp.epfl.ch/vil/YOUR_IMAGE_NAME
        ```
5.  **Push your image**: Upload your image so the cluster can access it:
    ```bash
    docker push registry.rcp.epfl.ch/vil/YOUR_IMAGE_NAME
    ```

---

## 4. Launching and Managing Jobs

You can follow [How to use RunAI submit](https://wiki.rcp.epfl.ch/en/home/CaaS/FAQ/how-to-runai-submit) for your job submission. Here are examples for Mac.

1.  **Submit the job**:
    ```bash
    runai submit \
        --name my-job-name \
        --image registry.rcp.epfl.ch/lab-username/docker-image-name:tag \
        --gpu 1 \
        --environment CUSTOM-ENV-VAR="custom-env-var-value" \
        --supplemental-groups ints \
        --existing-pvc claimname=persistent-volume-claim-name,path=/mont-path-in-the-container \
        --command \
        -- /bin/bash -ic "command-line-to-run"
    ```
4.  **Monitor your job**:
    * `runai list`: See a summary of your jobs.
    * `runai describe job YOUR_JOB_NAME`: Get detailed status information.
    * `runai logs YOUR_JOB_NAME`: View the terminal output.
5.  **Connect to a running job**: For interactive jobs, open a shell inside the container:
    ```bash
    runai bash YOUR_JOB_NAME
    ```
6.  **Delete your job**: When you are finished, always delete your job to free up resources.
    ```bash
    runai delete job YOUR_JOB_NAME
    ```

---

## 5. Tools & Data Management

### Data Volumes
The cluster has several storage options. The example YAML files already include the code to mount them.
* **`/scratch`**: **Fast but not backed up**. Good for temporary files. (100TB total shared)
* **`/datasets`**: **Slower but backed up**. This is your main persistent storage for code and models.
* **S3**: Best for **very large datasets**.
* **Local SSD**: Fast, but **volatile**. It gets wiped after the job. (~400GB)
* **RAM**: You can use a portion of the machine's RAM for ultra-fast temporary storage by increasing `dshm` in your job submission.

### SSH Server
There is a jump box node, **haas001**, provided by RCP for data transfer. It does not have GPUs, so you can only use it to update code or transfer data. For GPU-based model training, you will need to submit a job on the cluster.

    ```bash
    ssh <Gaspar Name>@haas001.rcp.epfl.ch
    cd /mnt/vilab/scratch/<Gaspar Name>
    ```


### Running a Visual Studio Code Server
After connecting to `haas001` via SSH, you can use VS Code’s **Remote - SSH** extension to edit files directly on the server.  

1. Open VS Code on your local machine.  
2. Install the extension **Remote - SSH** if not already installed.  
3. Add a new SSH target using your Gaspar account:  ```ssh <Gaspar Name>@haas001.rcp.epfl.ch```
4. Connect to this target from the VS Code Remote Explorer.  
5. Once connected, you can open the `/mnt/vilab/scratch/<Gaspar Name>` directory and work with your code as if it were local.  


### General Rules
* **Be a good neighbor**: Use CPU pods for tasks that don't need a GPU, and delete your interactive pods when not in use (e.g., at night) to free up resources.
* **Watch GPU utilization**: Ensure your jobs have reasonably high GPU utilization. If not, release the GPU.
* **Docker Cleanup**: Regularly delete old Docker containers on your local machine to save space.
