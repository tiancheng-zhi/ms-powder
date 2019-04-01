import visdom
import numpy as np

class Visualizer():

    def __init__(self, server='http://localhost', port=8097, env='main'):
        self.vis = visdom.Visdom(server=server, port=port, env=env, use_incoming_socket=False)
        self.iteration = []
        self.nlogloss = []
        self.epoch = []
        self.acc = []

    def state_dict(self):
        return {'iteration': self.iteration, 'nlogloss': self.nlogloss, 'epoch': self.epoch, 'acc': self.acc}


    def load_state_dict(self, state_dict):
        self.iteration = state_dict['iteration']
        self.nlogloss = state_dict['nlogloss']
        self.epoch = state_dict['epoch']
        self.acc = state_dict['acc']

    def plot_loss(self):
        self.vis.line(
            X=np.array(self.iteration),
            Y=np.array(self.nlogloss),
            opts={
                'title': '-LogLoss',
                'legend': ['-LogLoss'],
                'xlabel': 'epoch',
                'ylabel': '-logloss'},
            win=0)

    def plot_acc(self):
        self.vis.line(
            X=np.array(self.epoch),
            Y=np.array(self.acc),
            opts={
                'title': 'Performance',
                'legend': ['mIoUval', 'mIoUtest'],
                'xlabel': 'epoch',
                'ylabel': 'performance'},
            win=1)

    def plot_image(self, im, idx):
        self.vis.image(im, win=idx + 2)
