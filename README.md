
# Task-Aware Multimodal EHR Representation Learning via Variational Information Bottleneck

This is the official repository for "Task-Aware Multimodal EHR Representation Learning via Variational Information Bottleneck"

MELT first introduces learnable task prompts to explicitly guide the model in capturing task-modality dependency. Then with a gating network that performs feature-wise modulation and compression, it generates a sparse, task-aware weight matrix from the prompts and input features. This process is regularized under the Variational Information Bottleneck (VIB) theory via a composite sparsity constraint, ensuring the model actively filters out redundant information and focuses on predictive features. Furthermore, an asymmetric fusion architecture aligns dynamic and static data in a manner consistent with clinical reasoning. Together, these components enable MELT to construct robust, compressed patient representations that are intrinsically tailored to specific clinical tasks. 


###### QuickStart

1. git clone this repos
2. 
```sh
cd MELT
pip install -e requirements.txt
```
3. Data Preparation<br />
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

