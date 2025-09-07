# ARQ

## Setup

To set up the environment for this project, follow these steps:

1. **Install Conda**: Ensure you have Conda installed on your system. Refer to the following link for an installation guide. https://conda.io/projects/conda/en/latest/user-guide/install/index.html.

2. **Create Environment and Install Dependencies**:
   - Create a new Conda environment:
     ```bash
     conda create --name your_env_name python=3.8
     conda activate your_env_name
     ```
   - Install PyTorch and related packages with CUDA 11.3 support:
     ```bash
     pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
     ```
   - Install the remaining dependencies listed in `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

## Training the Initial Model

To train the initial model, use the following command as an example:
```bash
bash run/pretrain_resnet20.sh
```

## Running Policy Search

To perform the policy search, use the following command as an example:
```bash
bash run/search_resnet20.sh
```

## Evaluation

To evaluate the quantization policy, follow these steps:

1. Replace the policy in `rl_quantize.py` with the policy obtained from the search phase.

2. Run the evaluation script like:
   ```bash
   bash run/eval_resnet20.sh
   ```
