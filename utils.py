import torch
import numpy as np
import os
import shutil

prng = np.random.RandomState(1)
torch.manual_seed(1)


def change_random_seed(seed):
    global prng
    prng = np.random.RandomState(seed)
    torch.manual_seed(seed)


def to_one_hot(x, n_cats=10):
    y = np.zeros((x.shape[0], n_cats))
    y[np.arange(x.shape[0]), x] = 1
    return y.astype(np.float32)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def get_flat_fts(in_size, fts):
    dummy_input = torch.ones(1, *in_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    f = fts(torch.autograd.Variable(dummy_input))
    print('conv_out_size: {}'.format(f.size()))
    return int(np.prod(f.size()[1:]))


def adjust_learning_rate(optimizer, epoch, lr=0.1, lr_decay_ratio=0.1, epoch_drop=(), writer=None):
    """Simple learning rate drop according to the provided parameters"""
    optim_factor = 0
    for i, ep in enumerate(epoch_drop):
        if epoch > ep:
            optim_factor = i + 1
    lr = lr * lr_decay_ratio ** optim_factor

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, name, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % name + 'model_best.pth.tar')
