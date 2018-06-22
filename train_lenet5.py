import argparse
import shutil
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import L0LeNet5
from utils import save_checkpoint
from dataloaders import mnist
from utils import AverageMeter, accuracy


parser = argparse.ArgumentParser(description='PyTorch LeNet5 Training')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='L0LeNet5', type=str,
                    help='name of experiment')
parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                    help='whether to use tensorboard (default: True)')
parser.add_argument('--beta_ema', type=float, default=0.999)
parser.add_argument('--lambas', nargs='*', type=float, default=[1., 1., 1., 1.])
parser.add_argument('--local_rep', action='store_true')
parser.add_argument('--temp', type=float, default=2./3.)
parser.add_argument('--multi_gpu', action='store_true')
parser.set_defaults(tensorboard=True)

best_prec1 = 100
writer = None
total_steps = 0
exp_flops, exp_l0 = [], []


def main():
    global args, best_prec1, writer, total_steps, exp_flops, exp_l0
    args = parser.parse_args()
    log_dir_net = args.name
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
    print('[0, 1] normalization of input')
    train_loader, val_loader, num_classes = mnist(args.batch_size, pm=False)

    # create model
    model = L0LeNet5(num_classes, input_size=(1, 28, 28), conv_dims=(20, 50), fc_dims=500, N=60000,
                     weight_decay=args.weight_decay, lambas=args.lambas, local_rep=args.local_rep,
                     temperature=args.temp)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        model = model.cuda()

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
            exp_flops = checkpoint['exp_flops']
            exp_l0 = checkpoint['exp_l0']
            if checkpoint['beta_ema'] > 0:
                model.beta_ema = checkpoint['beta_ema']
                model.avg_param = checkpoint['avg_params']
                model.steps_ema = checkpoint['steps_ema']
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
        total_loss = loss + model.regularization()
        if torch.cuda.is_available():
            total_loss = total_loss.cuda()
        return total_loss

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, loss_function, optimizer, epoch)
        # evaluate on validation set
        prec1 = validate(val_loader, model, loss_function, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'curr_prec1': prec1,
            'beta_ema': model.beta_ema,
            'optimizer': optimizer.state_dict(),
            'total_steps': total_steps,
            'exp_flops': exp_flops,
            'exp_l0': exp_l0
        }
        if model.beta_ema > 0:
            state['avg_params'] = model.avg_param
            state['steps_ema'] = model.steps_ema
        save_checkpoint(state, is_best, args.name)
    print('Best error: ', best_prec1)
    if args.tensorboard:
        writer.close()


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    global total_steps, exp_flops, exp_l0, args, writer
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

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
