import numpy as np
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
from scipy.signal import convolve2d

const = np.sqrt(2) + 1


def norm(q):  # sqrt(|α|^2+|β|^2)
    return np.sqrt(np.sum(np.abs(q) ** 2))


def normalize(q):
    return q / norm(q)


def colorize(z):
    hue = (np.angle(z) + np.pi) / (2 * np.pi) + 0.5
    lig = 2 / np.pi * np.arctan(np.abs(z))
    sat = 1.
    c = np.vectorize(hls_to_rgb)(hue, lig, sat)  # tuple
    c = np.array(c)  # array of (3,n,m) shape, but need (n,m,3)
    c = c.transpose((1, 2, 0))
    return c


def birth(q, phi):  # qubit is measured and set to |1>
    return normalize(np.array([0., np.abs(q[0]) * np.exp(1j * phi) + q[1]], complex))


def survival(q):  # identity
    return np.dot(np.array([[1, 0], [0, 1]], complex), q)


def death(q, phi):  # qubit is measured and set to |0>
    return normalize(np.array([q[0] + np.abs(q[1]) * np.exp(1j * phi), 0.], complex))


def pauli_z(q):  # Pauli-Z
    return np.dot(np.array([[0, 1], [1, 0]], complex), q)


class Automaton:
    def __init__(self, n=5, surface='T', kernel='M', radius=1):
        self.arr = np.stack([np.ones((n, n), complex), np.zeros((n, n), complex)], axis=2)
        self.n = n
        if surface in ['E', 'S', 'T', 'K', 'P']:
            self.surface = surface
        else:
            raise ValueError(f'Surface cannot be {surface}')
        if kernel == 'M':  # Moore
            self.kernel = np.ones((2*radius+1, 2*radius+1))
            self.kernel[radius, radius] = 0
        elif kernel == 'vN':  # von Neumann
            self.kernel = np.zeros((2*radius+1, 2*radius+1))
            self.kernel[radius, :] = 1
            self.kernel[:, radius] = 1
            self.kernel[radius, radius] = 0
        elif kernel == 'd':  # diamond
            self.kernel = np.zeros((2*radius+1, 2*radius+1))
            for i in range(2*radius+1):
                self.kernel[i, abs(radius-i):2*radius+1-abs(radius-i)] = 1
            self.kernel[radius, radius] = 0
        elif kernel == 'c':  # circle (disc)
            self.kernel = np.zeros((2*radius+1, 2*radius+1))
            for i in range(2*radius+1):
                for j in range(2*radius+1):
                    if (radius-i)**2+(radius-j)**2 <= radius**2+1:
                        self.kernel[i, j] = 1
            self.kernel[radius, radius] = 0
        else:
            raise ValueError(f'Kernel cannot be {kernel}')
        self.radius = radius

    def life(self):  # aliveness of the cell
        life = self.arr[:, :, 1]
        return life

    def measure(self):  # probability of being alive
        life = np.abs(self.life()) ** 2
        return life

    def show(self):
        life = colorize(self.life())
        im = plt.imshow(life)
        return im

    def alpha(self):  # aliveness of the neighborhood
        if self.surface == 'E':  # plane
            return convolve2d(self.life(), self.kernel, 'same')
        elif self.surface == 'S':  # sphere
            pad = np.pad(self.life(), self.radius, 'edge')
            pad[0, 1:-1] = self.arr[::-1, -1, 1]
            pad[1:-1, -1] = self.arr[0, ::-1, 1]
            pad[-1, 1:-1] = self.arr[::-1, 0, 1]
            pad[1:-1, 0] = self.arr[-1, ::-1, 1]
            return convolve2d(pad, self.kernel, 'valid')
        elif self.surface == 'T':  # torus
            return convolve2d(self.life(), self.kernel, 'same', 'wrap')
        elif self.surface == 'K':  # klein bottle
            pad = np.pad(self.life(), self.radius, 'wrap')
            pad[0] = pad[0, ::-1]
            pad[-1] = pad[-1, ::-1]
            return convolve2d(pad, self.kernel, 'valid')
        elif self.surface == 'P':  # projective plane
            pad = np.pad(self.life(), self.radius, 'wrap')
            pad[0] = pad[0, ::-1]
            pad[-1] = pad[-1, ::-1]
            pad[:, 0] = pad[::-1, 0]
            pad[:, -1] = pad[::-1, -1]
            return convolve2d(pad, self.kernel, 'valid')

    def next_gen(self):  # next generation
        a = self.alpha()
        A, phi = np.abs(a)[..., None], np.angle(a)
        fA = A % 1
        fA = np.where(fA == 0, 1, fA)
        mA = 1 - fA
        b = np.vectorize(birth, signature='(2),()->(2)')(self.arr, phi)
        s = self.arr
        d = np.vectorize(death, signature='(2),()->(2)')(self.arr, phi)
        sel = np.stack([np.logical_or(np.logical_and(0 <= A, A <= 1), 4 <= A),
                        np.logical_and(1 < A, A <= 2),
                        np.logical_and(2 < A, A <= 3),
                        np.logical_and(3 < A, A < 4)])
        g0 = sel[0] * d
        g1 = sel[1] * (const * mA * d + fA * s)
        g2 = sel[2] * (const * mA * s + fA * b)
        g3 = sel[3] * (const * mA * b + fA * d)
        g = np.vectorize(normalize, signature='(2)->(2)')(g0 + g1 + g2 + g3)
        return g

    def tick(self):
        g = self.next_gen()
        self.arr = g

    def flip(self, i, j):
        self.arr[i, j] = pauli_z(self.arr[i, j])

    def same(self, auto):
        for k in range(4):
            if np.allclose(np.rot90(self.life(), k), auto.life()):
                return True
        for k in range(4):
            if np.allclose(np.rot90(self.life()[::-1], k), auto.life()):
                return True
        return False

    def mean_phase_diff(self):
        diff = (np.angle(self.life()) - np.angle(self.alpha()) + np.pi) % (2 * np.pi) - np.pi
        return np.mean(np.abs(diff))

    def mean_weighted_phase_diff(self):
        diff = (np.angle(self.life()) - np.angle(self.alpha()) + np.pi) % (2 * np.pi) - np.pi
        return np.average(np.abs(diff), weights=self.measure())
