import logging

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import torch

LINE_COLORS = ['w', 'r', 'y', 'cyan', 'm', 'b', 'lime']

MATPLOTLIB_FLAG = False

def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    return fig


def spec_f0_to_figure(spec, f0s, figsize=None):
    max_y = spec.shape[1]
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
        f0s = {k: f0.detach().cpu().numpy() for k, f0 in f0s.items()}
    f0s = {k: f0 / 10 for k, f0 in f0s.items()}
    fig = plt.figure(figsize=(12, 6) if figsize is None else figsize)
    plt.pcolor(spec.T)
    for i, (k, f0) in enumerate(f0s.items()):
        plt.plot(f0.clip(0, max_y), label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.8)
    plt.legend()
    return fig


# def dur_to_figure(dur_gt, dur_pred, txt):
#     dur_gt = dur_gt.long().cpu().numpy()
#     dur_pred = dur_pred.long().cpu().numpy()
#     dur_gt = np.cumsum(dur_gt)
#     dur_pred = np.cumsum(dur_pred)
#     fig = plt.figure(figsize=(12, 6))
#     for i in range(len(dur_gt)):
#         shift = (i % 8) + 1
#         plt.text(dur_gt[i], shift, txt[i])
#         plt.text(dur_pred[i], 10 + shift, txt[i])
#         plt.vlines(dur_gt[i], 0, 10, colors='b')  # blue is gt
#         plt.vlines(dur_pred[i], 10, 20, colors='r')  # red is pred
#     return fig

def dur_to_figure(dur_gt, dur_pred, txt):
    if isinstance(dur_gt, torch.Tensor):
        dur_gt = dur_gt.cpu().numpy()
    if isinstance(dur_pred, torch.Tensor):
        dur_pred = dur_pred.cpu().numpy()
    dur_gt = dur_gt.astype(np.int64)
    dur_pred = dur_pred.astype(np.int64)
    dur_gt = np.cumsum(dur_gt)
    dur_pred = np.cumsum(dur_pred)
    width = max(12, min(48, len(txt) // 2))
    fig = plt.figure(figsize=(width, 8))
    plt.vlines(dur_pred, 12, 22, colors='r', label='pred')
    plt.vlines(dur_gt, 0, 10, colors='b', label='gt')
    for i in range(len(txt)):
        shift = (i % 8) + 1
        plt.text((dur_pred[i-1] + dur_pred[i]) / 2 if i > 0 else dur_pred[i] / 2, 12 + shift, txt[i],
                 size=16, horizontalalignment='center')
        plt.text((dur_gt[i-1] + dur_gt[i]) / 2 if i > 0 else dur_gt[i] / 2, shift, txt[i],
                 size=16, horizontalalignment='center')
        plt.plot([dur_pred[i], dur_gt[i]], [12, 10], color='black', linewidth=2, linestyle=':')
    plt.yticks([])
    plt.xlim(0, max(dur_pred[-1], dur_gt[-1]))
    fig.legend()
    fig.tight_layout()
    return fig



def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure()
    f0_gt = f0_gt.cpu().numpy()
    plt.plot(f0_gt, color='r', label='gt')
    if f0_cwt is not None:
        f0_cwt = f0_cwt.cpu().numpy()
        plt.plot(f0_cwt, color='b', label='cwt')
    if f0_pred is not None:
        f0_pred = f0_pred.cpu().numpy()
        plt.plot(f0_pred, color='green', label='pred')
    plt.legend()
    return fig



def curve_to_figure(curve_gt, curve_pred=None, curve_base=None, grid=None):
    if isinstance(curve_gt, torch.Tensor):
        curve_gt = curve_gt.cpu().numpy()
    if isinstance(curve_pred, torch.Tensor):
        curve_pred = curve_pred.cpu().numpy()
    if isinstance(curve_base, torch.Tensor):
        curve_base = curve_base.cpu().numpy()
    fig = plt.figure()
    if curve_base is not None:
        plt.plot(curve_base, color='g', label='base')
    plt.plot(curve_gt, color='b', label='gt')
    if curve_pred is not None:
        plt.plot(curve_pred, color='r', label='pred')
    if grid is not None:
        plt.gca().yaxis.set_major_locator(MultipleLocator(grid))
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    return fig


def distribution_to_figure(title, x_label, y_label, items: list, values: list, zoom=0.8):
    fig = plt.figure(figsize=(int(len(items) * zoom), 10))
    plt.bar(x=items, height=values)
    plt.tick_params(labelsize=15)
    plt.xlim(-1, len(items))
    for a, b in zip(items, values):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=15)
    plt.grid()
    plt.title(title, fontsize=30)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    return fig


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


