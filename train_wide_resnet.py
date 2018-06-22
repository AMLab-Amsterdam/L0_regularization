import argparse
import os
import shutil
import time

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn

from models import L0WideResNet
from dataloaders import cifar10, cifar100
from utils import save_checkpoint, AverageMeter, accuracy
from torch.optim import lr_scheduler


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--depth', default=28, type=int,
                    help='total depth of the network (default: 28)')
parser.add_argument('--width', default=10, type=int,
                    help='total width of the network (default: 10)')
parser.add_argument('--droprate_init', default=0.3, type=float,
                    help='dropout probability (default: 0.3)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='L0WideResNet', type=str,
                    help='name of experiment')
parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                    help='whether to use tensorboard (default: True)')
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--lamba', type=float, default=0.001,
                    help='Coefficient for the L0 term.')
parser.add_argument('--beta_ema', type=float, default=0.99)
parser.add_argument('--lr_decay_ratio', type=float, default=0.2)
parser.add_argument('--dataset', choices=['c10', 'c100'], default='c10')
parser.add_argument('--local_rep', action='store_true')
parser.add_argument('--epoch_drop', nargs='*', type=int, default=(60, 120, 160))
parser.add_argument('--temp', type=float, default=2./3.)
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)
parser.set_defaults(tensorboard=True)

best_prec1 = 100
writer = None
time_acc = [(0, 0, 0)]
total_steps = 0
exp_flops, exp_l0 = [], []


def main():
    global args, best_prec1, writer, time_acc, total_steps, exp_flops, exp_l0
    args = parser.parse_args()
    log_dir_net = args.name
    args.name += '_{}_{}'.format(args.depth, args.width)
    if args.dataset == 'c100':
        args.name += '_c100'
    print('model:', args.name)
    if args.tensorboard:
        # used for logging to TensorBoard
        from tensorboardX import SummaryWriter
        directory = 'logs/{}/{}'.format(log_dir_net, args.name)
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            os.makedirs(directory)
        writer = SummaryWriter(directory)

    # Data loading code
    dataload = cifar10 if args.dataset == 'c10' else cifar100
    train_loader, val_loader, num_classes = dataload(augment=args.augment, batch_size=args.batch_size)

    # create model
    model = L0WideResNet(args.depth, num_classes, widen_factor=args.width, droprate_init=args.droprate_init,
                         N=50000, beta_ema=args.beta_ema, weight_decay=args.weight_decay, local_rep=args.local_rep,
                         lamba=args.lamba, temperature=args.temp)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        if torch.cuda.is_available():
            model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, nesterov=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            total_steps = checkpoint['total_steps']
            time_acc = checkpoint['time_acc']
            exp_flops = checkpoint['exp_flops']
            exp_l0 = checkpoint['exp_l0']
            if checkpoint['beta_ema'] > 0:
                if not args.multi_gpu:
                    model.beta_ema = checkpoint['beta_ema']
                    model.avg_param = checkpoint['avg_params']
                    model.steps_ema = checkpoint['steps_ema']
                else:
                    model.module.beta_ema = checkpoint['beta_ema']
                    model.module.avg_param = checkpoint['avg_params']
                    model.module.steps_ema = checkpoint['steps_ema']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            total_steps, exp_flops, exp_l0 = 0, [], []

    cudnn.benchmark = True

    loglike = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loglike = loglike.cuda()

    # define loss function (criterion) and optimizer
    def loss_function(output, target_var, model):
        loss = loglike(output, target_var)
        reg = model.regularization() if not args.multi_gpu else model.module.regularization()
        total_loss = loss + reg
        if torch.cuda.is_available():
            total_loss = total_loss.cuda()
        return total_loss

    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=args.epoch_drop, gamma=args.lr_decay_ratio)

    for epoch in range(args.start_epoch, args.epochs):
        time_glob = time.time()

        # train for one epoch
        prec1_tr = train(train_loader, model, loss_function, optimizer, lr_schedule, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, loss_function, epoch)
        time_ep = time.time() - time_glob
        time_acc.append((time_ep + time_acc[-1][0], prec1_tr, prec1))

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'curr_prec1': prec1,
            'optimizer': optimizer.state_dict(),
            'type_net': args.type_net,
            'total_steps': total_steps,
            'time_acc': time_acc,
            'exp_flops': exp_flops,
            'exp_l0': exp_l0
        }
        if not args.multi_gpu:
            state['beta_ema'] = model.beta_ema
            if model.beta_ema > 0:
                state['avg_params'] = model.avg_param
                state['steps_ema'] = model.steps_ema
        else:
            state['beta_ema'] = model.module.beta_ema
            if model.module.beta_ema > 0:
                state['avg_params'] = model.module.avg_param
                state['steps_ema'] = model.module.steps_ema
        save_checkpoint(state, is_best, args.name)
    print('Best error: ', best_prec1)
    if args.tensorboard:
        writer.close()


def train(train_loader, model, criterion, optimizer, lr_schedule, epoch):
    """Train for one epoch on the training set"""
    global total_steps, exp_flops, exp_l0, args, writer
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    lr_schedule.step(epoch=epoch)
    if writer is not None:
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    end = time.time()
    for i, (input_, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        total_steps += 1
        if torch.cuda.is_available():
            target = target.cuda(async=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var, model)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input_.size(0))
        top1.update(100 - prec1[0], input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clamp the parameters
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            layer.constrain_parameters()

        e_fl, e_l0 = model.get_exp_flops_l0() if not args.multi_gpu else \
            model.module.get_exp_flops_l0()
        exp_flops.append(e_fl)
        exp_l0.append(e_l0)
        if writer is not None:
            writer.add_scalar('stats_comp/exp_flops', e_fl, total_steps)
            writer.add_scalar('stats_comp/exp_l0', e_l0, total_steps)

        if not args.multi_gpu:
            if model.beta_ema > 0.:
                model.update_ema()
        else:
            if model.module.beta_ema > 0.:
                model.module.update_ema()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # input()
        if i % args.print_freq == 0:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('train/loss', losses.avg, epoch)
        writer.add_scalar('train/err', top1.avg, epoch)

    return top1.avg


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    global args, writer
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if not args.multi_gpu:
        if model.beta_ema > 0:
            old_params = model.get_params()
            model.load_ema_params()
    else:
        if model.module.beta_ema > 0:
            old_params = model.module.get_params()
            model.module.load_ema_params()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(async=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var, model)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input_.size(0))
        top1.update(100 - prec1[0], input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Err@1 {top1.avg:.3f}'.format(top1=top1))
    if not args.multi_gpu:
        if model.beta_ema > 0:
            model.load_params(old_params)
    else:
        if model.module.beta_ema > 0:
            model.module.load_params(old_params)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/err', top1.avg, epoch)
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)

    return top1.avg


if __name__ == '__main__':
    main()
