"""
This file contains utility classes and functions for logging to stdout, stderr,
and to tensorboard.
"""
import os
import sys
import numpy as np
from datetime import datetime
from contextlib import contextmanager
from tqdm import tqdm
import matplotlib.pyplot as plt


class PrintLogger(object):
    """
    This class redirects print statements to both console and a file.
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        print('STDOUT will be forked to %s' % log_file)
        self.log_file = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class DataLogger(object):
    """
    Logging class to log metrics to tensorboard and/or retrieve running statistics about logged data.
    """
    def __init__(self, log_dir, log_tb=True):
        """
        Args:
            log_dir (str): base path to store logs
            log_tb (bool): whether to use tensorboard logging
        """
        self._tb_logger = None
        self._data = dict() # store all the scalar data logged so far
        self.log_dir = log_dir

        if log_tb:
            from tensorboardX import SummaryWriter
            self._tb_logger = SummaryWriter(os.path.join(log_dir, 'tb'))

    def record(self, k, v, epoch, data_type='scalar', log_stats=False):
        """
        Record data with logger.

        Args:
            k (str): key string
            v (float or image): value to store
            epoch: current epoch number
            data_type (str): the type of data. either 'scalar' or 'image'
            log_stats (bool): whether to store the mean/max/min/std for all data logged so far with key k
        """

        assert data_type in ['scalar', 'image']

        if data_type == 'scalar':
            # maybe update internal cache if logging stats for this key
            if log_stats or k in self._data: # any key that we're logging or previously logged
                if k not in self._data:
                    self._data[k] = []
                self._data[k].append(v)

        # maybe log to tensorboard
        if self._tb_logger is not None:
            if data_type == 'scalar':
                self._tb_logger.add_scalar(k, v, epoch)
                if log_stats:
                    stats = self.get_stats(k)
                    for (stat_k, stat_v) in stats.items():
                        stat_k_name = '{}-{}'.format(k, stat_k)
                        self._tb_logger.add_scalar(stat_k_name, stat_v, epoch)
            elif data_type == 'image':
                self._tb_logger.add_images(k, img_tensor=v, global_step=epoch, dataformats="NHWC")

    def save_3d_affordance_grid(self, affordances, name_prefix, name_postfix):
        # TODO(VS) plot all affordances
        aff = affordances[-3].detach().cpu().numpy() # plotting one obs only

        import pylab

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colmap = pylab.cm.ScalarMappable(cmap=pylab.cm.viridis)
        colmap.set_array(aff[:,3])
        ax.scatter(aff[:,0], aff[:,1], aff[:,2], c=pylab.cm.viridis((aff[:,3]-min(aff[:,3]))/(max(aff[:,3])-min(aff[:,3]))), marker='o')
        fig.colorbar(colmap)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        os.makedirs(f'{self.log_dir}/affordances', exist_ok=True)
        plt.savefig(f'{self.log_dir}/affordances/{name_prefix}_affordances_{name_postfix}.png')
        plt.close()

    def save_2d_affordances_and_images(self, affordances, input_images, image_recons, name_prefix, name_postfix):
        for k in input_images: # all image keys in the input
            imgs = input_images[k]
            num_rows = 2

            if image_recons is not None:
                recons = image_recons[k]
                num_rows = 3

            num_imgs = len(imgs)
            fig, ax = plt.subplots(num_rows, num_imgs)
            if num_imgs == 1:
                ax = np.array([ax]).transpose()

            for col in range(num_imgs):
                aff = affordances[col, ..., -1]
                img = imgs[col].transpose([1,2,0])
                # aff = aff[..., -1] # others are just coordinates of the corresponding action
                if image_recons is not None:
                    recon = recons[col].transpose([1,2,0])
                    ax[2,col].imshow(recon)
                    ax[2,col].set_xlabel('X')
                    ax[2,col].set_ylabel('Y')
                    ax[2,col].set_title('recon')

                ax[0,col].imshow(img)
                plot = ax[1,col].imshow(aff, cmap='viridis'); plt.colorbar(plot, ax=ax[1,col])

                ax[0,col].set_xlabel('X'); ax[1,col].set_xlabel('X')
                ax[0,col].set_ylabel('Y'); ax[1,col].set_ylabel('Y')
                ax[0,col].set_title('input'); ax[1,col].set_title('afford')

            os.makedirs(f'{self.log_dir}/affordances', exist_ok=True)
            plt.savefig(f'{self.log_dir}/affordances/{name_prefix}_affordances_{k}_{name_postfix}.png')
            plt.close()

    def get_stats(self, k):
        """
        Computes running statistics for a particular key.

        Args:
            k (str): key string
        Returns:
            stats (dict): dictionary of statistics
        """
        stats = dict()
        stats['mean'] = np.mean(self._data[k])
        stats['std'] = np.std(self._data[k])
        stats['min'] = np.min(self._data[k])
        stats['max'] = np.max(self._data[k])
        return stats

    def close(self):
        """
        Run before terminating to make sure all logs are flushed
        """
        if self._tb_logger is not None:
            self._tb_logger.close()


class custom_tqdm(tqdm):
    """
    Small extension to tqdm to make a few changes from default behavior.
    By default tqdm writes to stderr. Instead, we change it to write
    to stdout.
    """
    def __init__(self, *args, **kwargs):
        assert "file" not in kwargs
        super(custom_tqdm, self).__init__(*args, file=sys.stdout, **kwargs)


@contextmanager
def silence_stdout():
    """
    This contextmanager will redirect stdout so that nothing is printed
    to the terminal. Taken from the link below:

    https://stackoverflow.com/questions/6735917/redirecting-stdout-to-nothing-in-python
    """
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target
