# Anti-Bandit Neural Architecture Search for Model Defense

Deep convolutional neural networks (DCNNs) have dominated as the best performers in machine learning, but can be challenged by adversarial attacks. In this paper, we defend adversarial attacks by neural architecture search (NAS) based on a comprehensive search space including denoising blocks, weight-free operations, Gabor filters and convolutions. It becomes more challenging than traditional NAS due to  more complicated search space and  adversarial training. We introduce an anti-bandit NAS (ABanditNAS) by designing new operation evaluation measure and search process based on the lower and upper confidence bound (LCB and UCB). Unlike the conventional bandit algorithm using UCB for evaluation only, we use not only UCB to abandon arms for search efficiency, but LCB for a fair competition between arms. Extensive experiments demonstrate that ABanditNAS is about twice as fast as the state-of-the-art PC-DARTS with a better performance. We achieve a much better performance than prior arts, such as  $8.73\%$ improvement on CIFAR-10 under PGD-$7$.

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