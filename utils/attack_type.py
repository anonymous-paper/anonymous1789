from attacks import *
import numpy as np

attackers = {
    'fgsm': lambda predict, eps, nb_iter, eps_iter: FGSM(predict, eps=eps),
    'pgd': lambda predict, eps, nb_iter, eps_iter: PGDAttack(predict, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True,\
                                                        ord=np.inf, l1_sparsity=None),
}