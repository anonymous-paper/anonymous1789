# Anti-Bandit Neural Architecture Search for Model Defense

Deep convolutional neural networks (DCNNs) have dominated as the best performers in machine learning, but can be challenged by adversarial attacks that modify samples in a  subtle way to fool the network into misdetection or misidentification. In this paper, we describe the design of robust structures to defend adversarial attacks based on a comprehensive search space including denoising blocks,  weight-free operations, Gabor filters and convolutions. This task becomes more challenging than traditional neural architecture search (NAS) due to the more complicated search space and the adversarial training process. To solve the problem, we introduce an anti-bandit NAS (ABanditNAS) algorithm to significantly improve the search efficiency in a flexible manner. We build a bridge between our new anti-bandit strategy and potential operations based on both a lower and an upper bound. Extensive experiments demonstrate that the proposed ABanditNAS is about twice as fast as the state-of-the-art PC-DARTS with a better performance in accuracy. Under adversarial attacks, we achieve much better accuracy than state-of-the-arts  on the MNIST and CIFAR databases. For example, under the $7$-iteration PGD white-box attacks, ABanditNAS obtains an accuracy improvement of $8.73\%$  over the prior art on CIFAR-10. The source code will be publicly released when the paper is accepted for publication.

Here we provide our test codes and pretrained models.

## Requirements

- python 3.6
- PyTorch 1.0.0

## Run examples
You need to modified your path to dataset using ```--data_path_cifar```.

To evaluate the model in **CIFAR-10**, just run

```bash
sh script/cifar10_darts_defense_all.sh
```