# Investigating the Impact of Randomness in Reproducibility in Computer Vision
This repository contains the source code of master's thesis titled "Investigating the Impact of Randomness in Reproducibility in Computer Vision: A Study on Applications in Civil Engineering and Medicine "

Note: The repository move to https://github.com/mcmi-group/repincv. Please refer to that repository as this one will not get any update in the future!
## Abstract

**Purpose**: Reproducibility stands as a cornerstone in scientific research. However, in the realm of computer vision, achieving consistent results is challenging due to various factors, with CUDA-induced randomness being a prominent one. Despite CUDA's capability to facilitate high-performance execution of computer vision algorithms on GPUs, it lacks determinism across multiple runs. This thesis delves into the aftermath of CUDA-induced variability on reproducibility to understand its nature better, focusing on diverse datasets: CIFAR for image classification, a dataset pertaining to civil engineering for concrete crack detection, and a medical dataset centered on breast cancer diagnosis.

**Methods**: To discern the influence of CUDA randomness, we thoroughly controlled other potential variability sources, including weight initialization, data shuffling, and data augmentation. This allowed us to exclusively study the effects of CUDA randomness by comparing outcomes over several runs, both in deterministic and non-deterministic CUDA settings. Additionally, an in-depth analysis of model weights was conducted to offer insights into the internal workings of the models, further corroborating our findings.

**Results**: Our empirical investigations revealed that while achieving full determinism is feasible, it occasionally comes at the expense of performance. Interestingly, model sensitivity to CUDA randomness varied with different configurations and settings. Upholding the principles of responsible research, we also delineate the environmental implications of our experiments, emphasizing the associated carbon footprint.

**Conclusion**: This thesis provides a rigorous evaluation of CUDA randomness and its implications on various computer vision applications. By doing so, it contributes to the advancement of reproducible research in computer vision by providing a systematic and comprehensive evaluation of CUDA randomness and its effects on different computer vision tasks and domains. Furthermore, we give recommendations to guide future research endeavors. The complex relationship between GPU utilization and the inherent randomness in Deep Learning requires further exploration.


# Reproducing Results for Thesis

This repository contains the code used in the thesis. Below are the datasets used and instructions on how to reproduce the results.

## Datasets

The datasets used in this thesis are:

- **CIFAR-10 Dataset:** [University of Toronto's CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html)
- **SDNET2018 Dataset:** [Utah State University's digital commons page](https://digitalcommons.usu.edu/all_datasets/48/)
- **CBIS-DDSM:** [Cancer Imaging Archive wiki](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629)

## Code Organization

In the repository, the code is organized as:

- **task1:** CIFAR-10 experiments
- **task2:** Concrete Crack Detection experiments
- **task3:** Breast Cancer Imaging experiments

## Instructions to Reproduce the Results

### CIFAR-10 and SDNET2018

These datasets are automatically downloaded if they are not present in the input directory.

### Breast Cancer Imaging

Due to the large size of the dataset, it needs to be **downloaded manually** and placed in the `input` directory.

For preprocessing the dataset, you can utilize the `image_cleaning.py` script. The algorithm for this script was developed by:

> **M.Sc Shreyasi Pathak**  
> PhD student at Twente University

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
