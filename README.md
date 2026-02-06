# MELT
This is the official repository for "Task-Aware Multimodal EHR Representation Learning via Variational Information Bottleneck"

MELT introduces learnable task prompts to explicitly guide the model in capturing task-modality dependency. Then with a gating network that performs feature-wise modulation and compression, it generates a sparse, task-aware weight matrix from the prompts and input features. This process is regularized under the Variational Information Bottleneck (VIB) theory via a composite sparsity constraint, ensuring the model actively filters out redundant information and focuses on predictive features. Furthermore, an asymmetric fusion architecture aligns dynamic and static data in a manner consistent with clinical reasoning.

Quick Start
1. Installation
git clone this git
cd MELT

pip install -r requirements.txt
