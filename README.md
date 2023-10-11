# Investigating the Impact of Randomness in Reproducibility in Computer Vision
This repository contains the source code of master's thesis titled "Investigating the Impact of Randomness in Reproducibility in Computer Vision: A Study on Applications in Civil Engineering and Medicine "


# Reproducing Results for Thesis

This repository contains the code used in the thesis. Below are the datasets used and instructions on how to reproduce the results.

## Datasets

The datasets used in this thesis are:

- **CIFAR-10 Dataset:** [Link to dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- **SDNET2018 Dataset:** [Link to dataset](https://digitalcommons.usu.edu/all_datasets/48/)
- **CBIS-DDSM:** [Link to dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629)

## Code Organization

In the repository, the code is organized as:

- **task1:** CIFAR-10 experiments
- **task2:** Concrete Crack Detection experiments
- **task3:** Breast Cancer Imaging experiments

## Instructions to Reproduce the Results

### CIFAR-10 and SDNET2018

These datasets are automatically downloaded if they are not present in the input directory.

### Breast Cancer Imaging

This dataset needs to be downloaded manually due to its large size and placed in the input directory. For preprocessing, please refer to this repository.

### Using Weights & Biases (WandB) for Tracking

To reproduce results, the Weights & Biases (WandB) platform is used. Follow the steps below:

#### Obtain the Weights & Biases (WandB) API Key

1. Visit the [Weights & Biases website](https://www.wandb.com/).
2. Sign in or create an account.
3. Navigate to your settings page to find your API key.
4. Copy the API key for later use.

For a detailed guide on obtaining the WandB API key, check [here](https://docs.wandb.ai/quickstart).

#### Clone the Repository

```bash
git clone https://github.com/bahadirery/repincv.git
cd repincv
```

#### Login to WandB

```bash
wandb login
```

When prompted, paste the API key you obtained earlier.

#### Create a Conda Environment

```bash
conda create --name repincv_env python=3.8
conda activate repincv_env
```

#### Install the Requirements

```bash
pip install -r requirements.txt
```

#### Run the Experiments

```bash
python run_experiments.py
```

If you want to run the experiments on a specific task, use:

```bash
python run_experiments.py --task <task_name>
```

where `task_name` can be one of: `task1`, `task2`, `task3`.
