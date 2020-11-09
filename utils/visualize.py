# coding:utf8
import os
import time

import visdom
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn


vis = None


class Visualizer(object):
    """ A wrapper for Visdom
    """
    def __init__(self, env, offline, server, port, log_file, raise_exceptions):
        self.vdm = visdom.Visdom(env=env,
                                 offline=offline, server=server, port=port,
                                 log_to_filename=log_file,
                                 raise_exceptions=raise_exceptions)

        self.index = {}
        self.log_text = ''

    # def reinit(self, env='default', **kwargs):
    #     """ modify visdom configuration
    #     """
    #     self.vdm = visdom.Visdom(env=env, **kwargs)
    #     return self

    def plot_many(self, d):
        """
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vdm.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vdm.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def matplot(self, name, img_, colored=True, **kwargs):
        if colored:
            cmap = plt.cm.jet
        else:
            cmap = plt.cm.gray
        plt.imshow(img_, cmap=cmap)
        if kwargs.get('resizable') is not None:
            self.vdm.matplot(plt, win=name,
                             opts=dict(title=name, resizable=kwargs['resizable'], height=kwargs['height'], width=kwargs['width']))
        else:
            self.vdm.matplot(plt, win=name, opts=dict(title=name))

    def matplot_many(self, d, **kwargs):
        for k, v in d.items():
            self.matplot(k, v, **kwargs)

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vdm.text(self.log_text, win)

    def properties(self, d, win, ordered=False):
        data = []
        keys = d.keys()
        if ordered:
            keys = sorted(keys)
        for k in keys:
            v = d[k]
            if isinstance(v, (int, float, complex)):
                data.append({
                    'type': 'number',
                    'name': k,
                    'value': v
                })
            else:
                data.append({
                    'type': 'text',
                    'name': k,
                    'value': str(v)
                })
        self.vdm.properties(data, win=win, opts=dict(title=win))

    def save(self):
        self.vdm.save([self.vdm.env])

    def __getattr__(self, name):
        return getattr(self.vdm, name)


def show_options(opt, prefix=''):
    global vis
    if vis is None:
        raise Exception('Visualizer is undefined!')

    title = prefix
    excluded = ['model', 'use_gpu'] #['gpu_devices', 'model', 'use_gpu']

    show = {}
    optimizers = None
    for attr in dir(opt):
        if not attr.startswith('_'):
            o = getattr(opt, attr)
            if hasattr(o, '__name__'):
                continue
            if attr in excluded:
                continue
            if attr == 'optimizers':
                optimizers = o
                continue
            show[attr] = o

    vis.properties(show, title)

    if optimizers is not None:
        for key in optimizers:
            optim = optimizers[key]
            show_options(vis, optim, prefix+'_%s_optimizer' % key)


def get_criterion_weights(crit, prefix=''):
    excluded = ['dump_patches', 'training', 'bias', 'dilation', 'groups', 'in_channels',
                'kernel_size', 'out_channels', 'output_padding', 'padding', 'stride', 'transposed', 'weight',
                'use_mask']

    st = ['w', 'sample_num', 'split_x', 'split_y', 'thrld']
    end = ['factor', ]

    show = {}
    sub_criterions = []
    for attr in dir(crit):
        if not attr.startswith('_'):
            o = getattr(crit, attr)
            name = prefix + '/' + attr
            if hasattr(o, '__name__'):
                continue
            if attr in excluded:
                continue
            if isinstance(o, nn.Module):
                sub_criterions.append({
                    'name': name,
                    'crit': o
                })
                continue

            flag = False
            for v in st:
                if attr.startswith(v):
                    flag = True
                    break
            for v in end:
                if attr.endswith(v):
                    flag = True
                    break
            if flag:
                show[prefix + '/' + attr] = o
    for v in sub_criterions:
        show_sub = get_criterion_weights(v['crit'], v['name'])
        show = {**show, **show_sub}
    return show


def show_model_setting(model, prefix=''):
    global vis
    if vis is None:
        raise Exception('Visualizer is undefined!')

    show = get_criterion_weights(model.GM_criterion, '')
    vis.properties(show, prefix + 'GM_criterion', ordered=True)
    show = get_criterion_weights(model.IID_criterion, '')
    vis.properties(show, prefix + 'IID_criterion', ordered=True)
    # show = get_criterion_weights(model.DE_refine_criterion, '')
    # vis.properties(show, prefix + 'DE_refine_criterion', ordered=True)


def show_learning_rate(model):
    global vis
    if vis is None:
        raise Exception('Visualizer is undefined!')

    optimizer = model.train_state.optimizer
    for param in optimizer.param_groups:
        vis.plot('lr for %s ' % (param['name']), param['lr'])


def create_a_visualizer(opt):
    global vis
    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)
    vis = Visualizer(opt.env, opt.offline, opt.server, opt.port, os.path.join(opt.checkpoints_dir, 'visdom.log'), False)


def save():
    global vis
    if vis is None:
        raise Exception('Visualizer is undefined!')
    vis.save()


