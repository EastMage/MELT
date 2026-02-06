
# Task-Aware Multimodal EHR Representation Learning via Variational Information Bottleneck

This is the official repository for "Task-Aware Multimodal EHR Representation Learning via Variational Information Bottleneck"

<!-- PROJECT SHIELDS -->

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://anonymous.4open.science/r/MELT-8156/">
    <img src="overview.pdf" alt="Logo" width="180" height="180">
  </a>

  <h3 align="center">Task-Aware Multimodal EHR Representation Learning via Variational Information Bottleneck

</p>




###### QuickStart

1. git clone this repos
2. 
```sh
cd MELT
pip install -e requirements.txt
```
3. Data Preparation
To access the datasets, please visit https://www.physionet.org/ and apply the data usage.

4. Train 
```sh
python train.py
```

### Files Structure

```
project 
├── README.md
├── LICENSE.txt
├── model.py
├── train.py
├── requirements.txt
│  ├── /utils/
│  │  ├── metrics.py
│  │  ├── metrics_los.py
│  │  ├── utils.py
│  ├── /data/
│  │  ├── data_mimic.py

```

