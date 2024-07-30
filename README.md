## Run Using Docker

To run the project using a Docker image, follow these steps:

### 1. Pull the Docker Image

First, pull the Docker image from Docker Hub:

```bash
docker pull mubtasimahasan/agent-vqa:v1
```

### 2. Run the Docker Container

Launch the Docker container with GPU support and bind the `./logs` directory to save output files:

```bash
docker run --rm -it --gpus all \
  -v ./logs:/app/logs \
  mubtasimahasan/agent-vqa:v1
```

This command will start the container, mapping the `logs` directory inside the container to the `logs` directory on your local machine.

### 3. Log in to Huggingface

To run the code, you need access to a gated Huggingface model `mistralai/Mistral-7B-Instruct-v0.2`. Simply request access to the [model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) and login to your Huggingface account using the command line. To login, huggingface_hub requires a token generated from [settings/token](https://huggingface.co/settings/token).




```
huggingface-cli login

huggingface-cli login --token {ENTER YOUR TOKEN HERE}
```

### 4. Execute the Script

Inside the running Docker container, execute the `main_script.sh` script. This script will download the dataset, and perform evaluation on all pipelines:

```bash
./main_script.sh
```

Alternatively, evaluate the dataset on a specific pipeline by providing an argument when running the script. You can choose between `static`, `dynamic`, or `no_plan`:

```bash
./main_script.sh static
./main_script.sh dynamic
./main_script.sh no_plan
```

To debug the pipeline on three data samples, append the `debug` argument to the shell file:

```bash
./main_script.sh debug
```

### Notes

- **GPU Support**: Ensure that your system is set up with the necessary NVIDIA drivers and Docker's GPU support to utilize the `--gpus all` option.

- **Output Files**: All output files will be stored in the `./logs` directory on your host machine.

- **Permissions**: Make sure that the current user has write permissions for the `./logs` directory on the host machine.


### Check Logs

Model training logs will be automatically saved to the `./logs` directory. You can find all outputs in `./logs/output.log`, check evaluation results in `./logs/results_{pipeline name}_{random number}.txt`, and get predicted answers in `./logs/predictions_val_{pipeline name}_{random number}.json`.
