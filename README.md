# $S^2TAT$
Synchronous Spatio-Temporal Graph Transformer: A New Framework for Traffic Data Prediction

[[Paper](https://ieeexplore.ieee.org/abstract/document/9770130)]

## Installation

```
Tensorflow > 2.0
```
This repo was tested with Ubuntu 18.04.1 LTS, Python 3.8, tensorflow 2.5.0, and CUDA 11. But it should be runnable with recent tensoflow versions >=2.0

## Data Preparation
The dataset is PEMS data for traffic prediction. Download and extract them under S2TAT/data, and make them look like this:
```
S2TAT/
  ├──data/
    ├──PEMS03/
    ├──PEMS04/
    ├──PEMS07/
    ├──PeMSD7L/
    └──PeMSD7M/
  ├──main.py
  ├──Model.py
  ├──Layers.py
  ├──utils.py
  └── ...
```

## Usage
```
cd S2TAT
python main.py --config $config_path$
```

## Results
The results of our $S^2TAT$ is tabulated as below
| Dataset | MAE | MAPE(%) | MSE | #Parameters(M)  |
|----------|:----:|:---:|:---:|:---:|
|  PEMS03 | 15.12 | 15.38  | 25.98 | 3.2  |
|  PEMS04 | 19.08 | 12.58  | 30.79  | 2.9  |
|  PEMS07 | 21.06 | 8.72  | 34.02  | 6.8  |
|  PEMS08 | 15.41 | 9.85  | 24.36  | 2.0  |
|  PeMSD7M | 2.67 | 6.61 | 5.41 | 2.4  |
|  PeMSD7L | 2.85 | 7.25 | 5.83  | 7.7  |

For any questions, please contact Jiahui Chen (jiahui.chen@buaa.edu.cn).

## Citation
If you find this repo useful for your research, please consider citing the paper
```
@article{wang2022synchronous,
  title={Synchronous Spatiotemporal Graph Transformer: A New Framework for Traffic Data Prediction},
  author={Wang, Tian and Chen, Jiahui and L{\"u}, Jinhu and Liu, Kexin and Zhu, Aichun and Snoussi, Hichem and Zhang, Baochang},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}
```