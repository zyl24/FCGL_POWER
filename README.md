Official code repository of the paper "Federated Continual Graph Learning" (ICDE 2025 submission)

[![arXiv](https://img.shields.io/badge/arXiv-2404.14061-b31b1b.svg)](https://arxiv.org/pdf/2411.18919)


**Requirements**

The experimental machine is an Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz and NVIDIA A100 with 80GB memory and CUDA 12.4. The operating system is Ubuntu 22.04.5 with 251GB memory.

Please refer to [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install the environments;

Please follow this repository to install the [OpenFGL](https://github.com/xkLi-Allen/OpenFGL) package.


**Training**

Here we take Cora-Louvain-3 Clients as an example:

```python
nohup python optuna_search.py > optuna_search_result.out
```

