# Cross-Device EEG Benchmark

## Overview
This repository provides the code for evaluating cross-hardware generalization in zero-calibration Motor Imagery Brain-Computer Interfaces. It includes a harmonization pipeline that aggregates five public EEG datasets (PhysioNet, Cho2017, Lee2019, Shin2017A, Weibo2014) into a standard 20-channel format. It also contains a PyTorch evaluation framework to test standard CNN baselines against a Domain Adversarial Neural Network (DANN) using a Leave-One-Dataset-Out (LODO) protocol.

## Dataset Access
The fully harmonized and aligned dataset is publicly available on Zenodo.
* Download the dataset here: [Insert Zenodo DOI Link Here]
* Place the downloaded `cross_device_eeg_aligned_FINAL.npz` file in your working directory, or update the file path inside the evaluation script.

## Required Libraries
Ensure you have Python 3.8 or newer installed. You will need the following packages:
* `numpy`
* `scipy`
* `scikit-learn`
* `torch`
* `mne`
* `moabb`
* `matplotlib`
* `seaborn`
* `pandas`
* `tqdm`

You can install the core dependencies via pip:
`pip install numpy scipy scikit-learn torch mne moabb matplotlib seaborn pandas tqdm`

## Instructions

**1. Data Harmonization (Optional)**
If you wish to recreate the dataset from scratch instead of downloading it from Zenodo, run the pipeline script. This downloads the raw files via MOABB, applies spatial interpolation, and performs Euclidean Alignment.
`python harmonization_pipeline.py`

**2. Model Evaluation**
To train and evaluate SimpleCNN, EEGNet, ShallowConvNet, and DANN on the dataset, run the final evaluation script. 
`python run_final_evaluation.py`

**3. Figure Generation**
To generate the 300 DPI publication-ready figures used in the manuscript, run the visualization script.
`python generate_figures.py`
