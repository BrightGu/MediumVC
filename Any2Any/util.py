from tensorboardX import SummaryWriter
import logging
import os
import shutil
import time
import sys
import torch
import numpy as np

def cc(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(device)


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))



class Logger(object):
    def __init__(self, save,log_file):
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        #fh = logging.FileHandler(os.path.join(save, 'log.txt'))
        fh = logging.FileHandler(os.path.join(save, log_file))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        self.start_time = time.time()

    def info(self, string, *args):
        elapsed_time = time.time() - self.start_time
        elapsed_time = time.strftime(
            '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
        if isinstance(string, str):
            string = elapsed_time + string
        else:
            logging.info(elapsed_time)
        logging.info(string, *args)


class Writer(object):
    def __init__(self,save):
        self.writer = SummaryWriter(log_dir=save, flush_secs=20)

    def add_scalar(self, *args, **kwargs):
        self.writer.add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        self.writer.add_figure(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        self.writer.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        self.writer.add_histogram(*args, **kwargs)

    def add_histogram_if(self, write, *args, **kwargs):
        self.add_histogram(*args, **kwargs)

    def close(self, *args, **kwargs):
        self.writer.close()


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

