import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.gridspec import GridSpec
from colorsys import hls_to_rgb
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1234, type=int, help='seed')
parser.add_argument('--n', default=100, type=int, help='cells per dimension')
parser.add_argument('--surf', default='E', type=str, help='surface type')
parser.add_argument('--n_frames', default=100, type=int, help='# of frames')
parser.add_argument('--interval', default=100, type=int, help='interval bw frames')
parser.add_argument('--stats', action='store_true', help='statistics')

args = parser.parse_args()
n = args.n

np.random.seed(args.seed)

def norm(q):  # sqrt(|α|^2+|β|^2)
    return np.sqrt(np.sum(np.abs(q)**2))

def normalize(q):
    return q/norm(q)

def colorize(z):
    h = (np.angle(z)+np.pi)/(2*np.pi)+0.5
    l = 2/np.pi*np.arctan(np.abs(z))
    s = 1.
    c = np.vectorize(hls_to_rgb) (h,l,s) # tuple
    c = np.array(c)  # array of (3,n,m) shape, but need (n,m,3)
    c = c.transpose(1,2,0)
    return c

c = np.sqrt(2)+1
# qubit is measured and set to |1>
B = lambda q,ϕ: normalize(np.array([0., np.abs(q[0])*np.exp(1j*ϕ)+q[1]], complex))
# Identity
S = lambda q: np.dot(np.array([[1,0],[0,1]], complex), q)
# qubit is measured and set to |0>
D = lambda q,ϕ: normalize(np.array([q[0]+np.abs(q[1])*np.exp(1j*ϕ), 0.], complex))
# Pauli-Z
Z = lambda q: np.dot(np.array([[0,1],[1,0]], complex), q)

class Universe():
    def __init__(self, n=5, surface='E'):
        self.arr = np.stack([np.ones((n,n), complex), np.zeros((n,n), complex)], axis=2)
        self.n = n
        self.surface=surface
    def life(self):
        life = self.arr[:,:,1]
        return life
    def measure(self):
        life = np.abs(self.arr[:,:,1])**2
        return life
    def show(self):
        life = colorize(self.life())
        im = plt.imshow(life)
        return im
    def Aij(self, i, j):  # liveness
        a = 0.
        for x,y in [(i,j+1),(i-1,j+1),(i-1,j),(i-1,j-1),(i,j-1),(i+1,j-1),(i+1,j),(i+1,j+1)]:
            if 0<=x<self.n and 0<=y<self.n:
                a += self.arr[x,y,1]
        return a
    def Gij(self, i, j):  # next generation
        a = self.Aij(i, j)
        A, ϕ = np.abs(a), np.angle(a)
        q = self.arr[i,j]
        if 0<=A<=1 or 4<=A:
            g = D(q,ϕ)
        elif 1<A<=2:
            g = c*(2-A)*D(q,ϕ) + (A-1)*S(q)
        elif 2<A<=3:
            g = c*(3-A)*S(q) + (A-2)*B(q,ϕ)
        elif 3<A<4:
            g = c*(4-A)*B(q,ϕ) + (A-3)*D(q,ϕ)
        else:
            raise ValueError(f'Liveness cannot be {a}')
        return normalize(g)
    def A(self):
        if self.surface=='E':  # plane
            pad = np.pad(U.arr[:,:,1], 1)
        elif self.surface=='S':  # sphere
            pad = np.pad(U.arr[:,:,1], 1, 'edge')
            pad[0,1:-1] = U.arr[::-1,-1,1]
            pad[1:-1,-1] = U.arr[0,::-1,1]
            pad[-1,1:-1] = U.arr[::-1,0,1]
            pad[1:-1,0] = U.arr[-1,::-1,1]
        elif self.surface=='T':  # torus
            pad = np.pad(U.arr[:,:,1], 1, 'wrap')
        elif self.surface=='K':  # klein bottle
            pad = np.pad(U.arr[:,:,1], 1, 'wrap')
            pad[0] = pad[0,::-1]
            pad[-1] = pad[-1,::-1]
        elif self.surface=='P':  # projective plane
            pad = np.pad(U.arr[:,:,1], 1, 'wrap')
            pad[0] = pad[0,::-1]
            pad[-1] = pad[-1,::-1]
            pad[:,0] = pad[::-1,0]
            pad[:,-1] = pad[::-1,-1]
        a = np.zeros((self.n,self.n), complex)
        for x,y in [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]:
            a += pad[x+1:x+1+self.n, y+1:y+1+self.n]
        return a
    def G_(self):
        g = np.empty_like(self.arr)
        for i in range(self.n):
            for j in range(self.n):
                g[i,j] = self.Gij(i, j)
        return g
    def G(self):
        a = self.A()
        A, ϕ = np.abs(a)[...,None], np.angle(a)
        fA = A%1
        fA = np.where(fA==0, 1, fA)
        mA = 1-fA
        b = np.vectorize(B, signature='(2),()->(2)') (self.arr, ϕ)
        s = self.arr
        d = np.vectorize(D, signature='(2),()->(2)') (self.arr, ϕ)
        sel = np.stack([np.logical_or(np.logical_and(0<=A, A<=1), 4<=A),
                        np.logical_and(1<A, A<=2),
                        np.logical_and(2<A, A<=3),
                        np.logical_and(3<A, A<4)])
        g0 = sel[0]*d
        g1 = sel[1]*(c*mA*d + fA*s)
        g2 = sel[2]*(c*mA*s + fA*b)
        g3 = sel[3]*(c*mA*b + fA*d)
        g = np.vectorize(normalize, signature='(2)->(2)') (g0+g1+g2+g3)
        return g
    def tick(self):
        g = self.G()
        self.arr = g
    def flip(self, i, j):
        self.arr[i,j] = Z(self.arr[i,j])

def init():
    return im,

if args.stats:
    def evolve(i):
        im.set_array(colorize(U.life()))
        life = np.abs(U.life().flatten())**2
        y, _ = np.histogram(life, 10, density=True)
        for count, rect in zip(y, mod.patches):
            rect.set_height(count)
        y, _ = np.histogram(np.angle(U.life().flatten())[life>0.5], 100, density=True)
        for count, rect in zip(y, arg.patches):
            rect.set_height(count)
        U.tick()
        return im,

    def colorize_(z):
        h = (np.angle(z)+np.pi)/(2*np.pi)+0.5
        l = 2/np.pi*np.arctan(np.abs(z))
        s = 1.
        c = np.vectorize(hls_to_rgb) (h,l,s)
        c = np.array(c).T
        return c
    palette = colorize_([np.exp(1j*angle) for angle in np.arange(-np.pi,np.pi,2*np.pi/100.)])

    fig = plt.figure(figsize=(9,6))
    gs = GridSpec(2, 3, figure=fig)
    ax0 = fig.add_subplot(gs[:,:2])
    ax1 = fig.add_subplot(gs[0,2])
    ax2 = fig.add_subplot(gs[1,2], projection='polar')
    U = Universe(n, args.surf)
    rand = np.random.random((3,n,n))
    a, b = np.sqrt([rand[0], 1.-rand[0]])
    U.arr = np.stack([a*np.exp(2j*np.pi*rand[1]), b*np.exp(2j*np.pi*rand[2])], axis=2)
    im = ax0.imshow(colorize(U.life()), animated=True)
    life = np.abs(U.life().flatten())**2
    _,_, mod = ax1.hist(life, 10, density=True)
    _,_, arg = ax2.hist(np.angle(U.life().flatten()), 100, density=True, weights=life)
    for i,rect in enumerate(arg.patches):
        rect.set_facecolor(palette[i])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 10)
    ax2.set_ylim(0, 0.5)
    ax0.axis('off')
    ax1.set_yticks(ax1.get_yticks())
    ax1.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xticklabels([])
    ax2.set_yticks(ax2.get_yticks())
    ax2.set_yticklabels(['', '', '', '', 0.5])
    ax1.set_xlabel(r'$|\langle 1 | \psi \rangle|^2$')
    ax2.set_ylabel(r'$\arg(\langle 1 | \psi \rangle) \times |\langle 1 | \psi \rangle|^2$')
    ax0.set_title(r'$\langle 1 | \psi \rangle$')
    fig.tight_layout()
    ani = anim.FuncAnimation(fig, evolve, init_func=init, blit=True, frames=tqdm(range(args.n_frames),initial=1), interval=args.interval)
    ani.save(f'sqgol_{U.surface}_stats.gif')
    plt.close()
else:
    def evolve(i):
        im.set_array(colorize(U.life()))
        U.tick()
        return im,

    fig = plt.figure(figsize=(5,5))
    U = Universe(n, args.surf)
    rand = np.random.random((3,n,n))
    a, b = np.sqrt([rand[0], 1.-rand[0]])
    U.arr = np.stack([a*np.exp(2j*np.pi*rand[1]), b*np.exp(2j*np.pi*rand[2])], axis=2)
    im = plt.imshow(colorize(U.life()), animated=True)
    plt.axis('off')
    fig.tight_layout()
    ani = anim.FuncAnimation(fig, evolve, init_func=init, blit=True, frames=tqdm(range(args.n_frames),initial=1), interval=args.interval)
    ani.save(f'sqgol_{U.surface}.gif')
    plt.close()
