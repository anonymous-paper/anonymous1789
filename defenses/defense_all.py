import os
import argparse
import time
import numpy as np
from tensorboardX import SummaryWriter
import sys
# setting the global param
root_path = '/weight/chl/nas'
data_path = {
             'cifar10': '/data/CIFAR',
             }
data_size = {
             'cifar10': '/data/CIFAR',
             }
parser = argparse.ArgumentParser("train_parser")
# data argument
parser.add_argument('--data', type=str, default='cifar10', choices=['cifar10'], help='dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size') # 512
parser.add_argument('--workers', type=int, default=0, help='worker to load the image') # 16
parser.add_argument('--data_loader_type', type=str, default='torch', choices=['torch', 'dali'],
                    help='choose different data loader')
parser.add_argument('--data_path_cifar', type=str, default='./data/CIFAR', help='date dir')
# model
parser.add_argument('--model', type=str, default='darts', choices=['darts'], help='model name')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels') # 16
parser.add_argument('--nchannel', default=1, type=int, metavar='N', help='nChannel (default: 1)')
parser.add_argument('--layers', type=int, default=6, help='total number of layers') # layers
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower') # False
parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--model_name', type=str, default='DDP_MNIST_FGSM_1', help='model name ')
parser.add_argument('--resume', '-r', type=str, default='./weights/DDP_MNIST_FGSM_1.pth.tar', help='resume from checkpoint') # action='store_true',
# training
parser.add_argument('--print_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', default="1", help='gpu device id')
# parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--no_decay_keys', type=str, default='bn', choices=[None, 'bn', 'bn#bias'])
parser.add_argument('--no_nesterov', action='store_true')  # opt_param

# attacker flags:
parser.add_argument('--nb_iter', help='Adversarial attack iteration', type=int, default=40)
parser.add_argument('--eps', help='Adversarial attack maximal perturbation', type=float, default=0.3)
parser.add_argument('--eps_iter', help='Adversarial attack step size', type=float, default=0.01)

parser.add_argument('--root_path', type=str, default='./weight/chl/nas', help='save dir name')
parser.add_argument('--save', type=str, default='try', help='save dir name')
parser.add_argument('--manual_seed', default=0, type=int)
args = parser.parse_args()

root_path = args.root_path
data_path['cifar10'] = args.data_path_cifar
data_size['cifar10'] = args.data_path_cifar

save_dir_str = args.data + '_' + args.model + '_' \
               + time.asctime(time.localtime()).replace(' ', '_') + '_' + args.model_name
out_path = os.path.join(root_path, args.save, save_dir_str)
# set GPU
args.gpu = [int(i) for i in args.gpu.split(',')]
# the environ should before import torch!
import torch
import torch.nn as nn
sys.path.append('.')
from models.darts_cnn_bandit_all import NetworkCIFAR
from utils import utils
from data_loader import get_data
from utils.utils import CrossEntropyLabelSmooth

from utils.genotypes_all import genotype_array

from utils.attack_type import attackers

# set device
torch.cuda.set_device(args.gpu[0])
device = torch.device("cuda")
writer = SummaryWriter(log_dir=os.path.join(out_path, "tb"))
writer.add_text('config', utils.as_markdown(vars(args)), 0)
logger = utils.get_logger(os.path.join(out_path, "logger.log"))
# set logger
logger.info("Logger is set - training start")
utils.print_params(vars(args), logger.info)
# set seed
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
torch.backends.cudnn.benchmark = True


def main():
    pin_memory = False if args.data == 'cifar10' else True
    [input_size, input_channels, n_classes, train_data, val_data] = \
        get_data.get_data(args.data, data_path[args.data], True, attack=True)

    val_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size=args.batch_size,
                                                sampler=None,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=pin_memory)
    
    # print(genotype_array)
    genotype = genotype_array[args.model_name]
    
    model_array = {'cifar10': NetworkCIFAR}
    model = model_array[args.data](input_channels, args.init_channels, n_classes, args.layers, args.auxiliary, False, genotype,
                                    input_size=input_size, dataset=args.data)
    
    model.set_bn_param(args.bn_momentum, args.bn_eps)
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} M".format(mb_params))
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()

    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        del checkpoint

    model_attack = model

    attack_type = ['clean', 'fgsm']
    attackers_val = [None]
    for i in attack_type:
        if i != 'clean':
            attacker = attackers[i](model_attack, eps=args.eps, nb_iter=args.nb_iter, eps_iter=args.eps_iter)
            attackers_val.append(attacker)

    attack_type.append('pgd7')
    attacker = attackers['pgd'](model_attack, eps=args.eps, nb_iter=7, eps_iter=args.eps_iter)
    attackers_val.append(attacker)

    attack_type.append('pgd20')
    attacker = attackers['pgd'](model_attack, eps=args.eps, nb_iter=20, eps_iter=args.eps_iter)
    attackers_val.append(attacker)

    # validation
    for i, attacker_ in enumerate(attackers_val):
        logger.info("-"*28 + attack_type[i] + "-"*28)
        validate(val_loader, model, attacker_)
    print("")


def validate(valid_queue, model, attacker=None):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.module.eval()
    len_val_quene = len(valid_queue)

    for step, data in enumerate(valid_queue):
        input = data[0].cuda(non_blocking=True)
        target = data[1].cuda(non_blocking=True)
        
        if attacker == None:
            xadv = input
        else:
            xadv = attacker.perturb(input, target)

        with torch.no_grad():
            result = model(xadv)
            if isinstance(result, tuple):
                logits = result[-2]
            else:
                logits = result
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = xadv.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            if step % args.print_freq == 0 or step == len_val_quene - 1:
                logger.info(
                    "Valid: Step {:03d}/{:03d} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        step, len_val_quene - 1,
                        top1=top1, top5=top5))

    logger.info("Valid: Final Prec@1 {:.4%} Final Prec@5 {:.4%}".format(top1.avg, top5.avg))
    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
