# PDO-eConvs
The implementation of the paper "[PDO-eConvs: Partial Differential Operator Based Equivariant Convolutions](https://arxiv.org/pdf/2007.10408.pdf)" (ICML2020).
Please contact shenzhy@pku.edu.cn if you have any question.

## Prerequisites
Tensorflow

Your may refer to https://github.com/Roderickzzc/Pdo-econv-pytorch for an implementation by Pytorch.

## Usage
### MNIST-rot-12k:
python3 mnist.py

### CIFAR10/100
python3 cifar.py --aug True --dataset cifar10


## Experimental Results
Error rates on MNIST-rot-12k (without data augmentation).
Network  | Test Error (%)  | params
 ---- | ----- | ------  
 CNN  | 5.03 | 22k 
 G-CNN  | 2.28 | 25k
 PDO-eConv  | 1.87 | 26k
 
Error rates on CIFAR.
Method  | G | Depth | C10 | C100  | params
------ | ------ | ------ | ------ | ------ | ------
ResNet | Z^2 | 26 | 11.5 | 31.66 | 0.37M
HexaConv | p6 | 26 | 9.98 |  | 0.34M
| | p6m | 26 | 8.64 |  | 0.34M
PDO-eConv | p6 | 26 | 5.65 | 27.13 | 0.36M
| | p6m | 26 | 5.38 | 27.00 | 0.37M
------ | ------ | ------ | ------ | ------ | ------
ResNet | Z^2 | 44 | 5.61 | 24.08 | 2.64M
G-CNN | p4m | 44 | 4.94 | 23.19 | 2.62M
ResNet | p8 | 44 | 3.68 | 20.01 | 2.62M
------ | ------ | ------ | ------ | ------ | ------
ResNet | Z^2 | 1001 | 4.92 | 22.71 | 10.3M
| | Z^2 | 26 | 4.00 | 19.25 | 36.5M
G-CNN | p4m | 26 | 4.17 |  | 7.2M
PDO-eConv | p8 | 26 | 3.50 | 18.40 | 4.6M


## Citation

If you found this package useful, please cite
```
@inproceedings{shen2020pdo,
  title={PDO-eConvs: Partial Differential Operator Based Equivariant Convolutions},
  author={Shen, Zhengyang and He, Lingshen and Lin, Zhouchen and Ma, Jinwen},
  booktitle={International Conference on Machine Learning},
  pages={8697--8706},
  year={2020},
  organization={PMLR}
}
```
