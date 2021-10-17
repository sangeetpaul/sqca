import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.gridspec import GridSpec
from colorsys import hls_to_rgb
from tqdm import tqdm
import argparse
import sqca

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1234, type=int, help='seed')
parser.add_argument('--n', default=100, type=int, help='cells per dimension')
parser.add_argument('--surf', default='T', type=str, help='surface type')
parser.add_argument('--kernel', default='M1', type=str, help='convolution kernel')
parser.add_argument('--frames', default=100, type=int, help='# of frames')
parser.add_argument('--interval', default=100, type=int, help='interval bw frames')
parser.add_argument('--stats', action='store_true', help='statistics')
parser.add_argument('--soup', default='bloch', type=str, help='random soup')
parser.add_argument('--ext', default='mp4', type=str, help='file extension')
parser.add_argument('--fps', default=None, type=int, help='frames per second')
parser.add_argument('--bitrate', default=None, type=int, help='bitrate in kbps')

args = parser.parse_args()
n = args.n
filename = f"sqca_{n}_{args.surf}_{args.frames}_{args.kernel}_{args.soup}{'_stats' if args.stats else ''}.{args.ext}"
print(f'File will be saved as {filename}')

np.random.seed(args.seed)

U = sqca.Automaton(n, args.surf, args.kernel)
if args.soup == 'bloch':
    rand = np.random.random((3, n, n))
    a, b = np.sqrt([rand[0], 1. - rand[0]])
    U.arr = np.stack([a * np.exp(2j * np.pi * rand[1]), b * np.exp(2j * np.pi * rand[2])], axis=2)
elif args.soup == 'pure':
    rand = np.random.random((3, n, n))
    a = rand[0] < 0.5
    U.arr = np.stack([a * np.exp(2j * np.pi * rand[1]), (1 - a) * np.exp(2j * np.pi * rand[2])], axis=2)
elif args.soup == 'roots':
    rand = np.random.random((3, n, n))
    a = rand[0] < 0.5
    b = np.array([1, 1j, -1, -1j])
    U.arr = np.stack([a * np.exp(2j * np.pi * rand[1]), (1 - a) * b[(rand[2] // 0.25).astype(int)]], axis=2)
elif args.soup == 'real':
    rand = np.random.random((3, n, n))
    a = rand[0] < 0.5
    b = np.array([1, -1])
    U.arr = np.stack([a * np.exp(2j * np.pi * rand[1]), (1 - a) * b[(rand[2] // 0.5).astype(int)]], axis=2)
elif args.soup == 'gol':
    rand = np.random.random((2, n, n))
    a = rand[0] < 0.5
    U.arr = np.stack([a * np.exp(2j * np.pi * rand[1]), (1 - a) * 1.], axis=2)


def init():
    return im,


if args.stats:
    def evolve(i):
        im.set_array(sqca.colorize(U.life()))
        life = np.abs(U.life().flatten()) ** 2
        y, _ = np.histogram(life, 10, range=(0, 1), density=True)
        for count, rect in zip(y, mod.patches):
            rect.set_height(count)
        y, _ = np.histogram(np.angle(U.life().flatten()), 100, range=(-np.pi, np.pi), weights=life, density=True)
        for count, rect in zip(y, arg.patches):
            rect.set_height(count)
        ent.set_data(np.r_[ent.get_xdata(), i], np.r_[ent.get_ydata(), U.mean_weighted_phase_diff()])
        U.tick()
        return im,


    def colorize_(z):
        h = (np.angle(z) + np.pi) / (2 * np.pi) + 0.5
        l = 2 / np.pi * np.arctan(np.abs(z))
        s = 1.
        c = hls_to_rgb(h, l, s)
        return c


    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(3, 3, figure=fig)
    ax0 = fig.add_subplot(gs[0:2, 0:2])
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[1, 2], projection='polar')
    ax3 = fig.add_subplot(gs[2, 0:3], yscale='log')
    im = ax0.imshow(sqca.colorize(U.life()), animated=True)
    life = np.abs(U.life().flatten()) ** 2
    _, _, mod = ax1.hist(life, 10, range=(0, 1), density=True)
    _, _, arg = ax2.hist(np.angle(U.life().flatten()), 100, range=(-np.pi, np.pi), weights=life, density=True)
    ent, = ax3.plot([], [])
    for rect in arg.patches:
        rect.set_facecolor(colorize_(np.exp(1j * rect.xy[0])))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 10)
    ax2.set_ylim(0, 0.5)
    ax3.set_xlim(0, args.frames - 1)
    ax3.set_ylim(0.01, np.pi)
    ax0.axis('off')
    ax1.set_yticks(ax1.get_yticks())
    ax1.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xticklabels([])
    ax2.set_yticks(ax2.get_yticks())
    ax2.set_yticklabels(['', '', '', '', r'$\frac{\pi}{100}$'])
    ax1.set_xlabel(r'$|\langle 1 | \psi \rangle|^2$')
    ax2.set_ylabel(r'$\arg(\langle 1 | \psi \rangle) \times |\langle 1 | \psi \rangle|^2$')
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\Delta\phi \times |\langle 1 | \psi \rangle|^2$')
    ax0.set_title(r'$\langle 1 | \psi \rangle$')
else:
    def evolve(i):
        im.set_array(sqca.colorize(U.life()))
        U.tick()
        return im,


    fig = plt.figure(figsize=(10, 10))
    im = plt.imshow(sqca.colorize(U.life()), animated=True)
    plt.axis('off')

fig.tight_layout()
ani = anim.FuncAnimation(fig, evolve, init_func=init, blit=True, frames=tqdm(range(args.frames), initial=1),
                         interval=args.interval)
ani.save(filename, fps=args.fps, bitrate=args.bitrate)
plt.close()
