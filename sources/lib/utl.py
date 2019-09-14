#
# Copyright (c) 2017, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

import math
import sys
import random
import string
import numpy as np

# ----------------------------

def polar(x, y):
    """
    Cartesian coordinates to polar coordinates; adapted to fit compass view.
    """
    r = math.sqrt(x**2 + y**2)
    p = math.atan2(y, x) * (-1) + math.radians(90)
    return p, r


def eigenvectors(covariance):
    """
    Returns Eigenvectors for given covariance matrix.
    """
    val, vec = np.linalg.eigh(covariance)
    order = val.argsort()[::-1]
    return val[order], vec[:,order]


def ellipse_covariance_2d(covariance, sigma=1):
    """
    Returns elliptical properties for given corvariance (in 2d space).
    """
    val, vec = eigenvectors(covariance)
    theta = np.degrees(np.arctan2(*vec[:, 0][::-1]))  # 2d
    w, h = 2 * sigma * np.sqrt(val)
    return w, h, theta


def euclidean_distance(a, b=None):
    """
    Euclidean distance between two n-dimensional points.
    """
    if b is None: return euclidean_distance(a, (0,) * len(a))
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(a, b)))
    #np.linalg.norm(a - b)
    #np.sqrt(np.sum((a-b)**2)))


def radians_squeeze(rad):
    rad = (rad + 2 * math.pi) % (2 * math.pi)
    rad = (rad % math.pi) - (math.pi * int(rad / math.pi))
    return rad

# ----------------------------

def tupleize(value):
    if type(value) not in (list, tuple):
        value = (value,)
    return tuple(value)


def tupleize_nx1(value):
    try:
        return tuple(tupleize_nx1(i) for i in value)
    except TypeError:
        return value


def tupleize_nxn(value):
    mat = tupleize_nx1(value)
    l = len(mat)
    return tuple([mat[i][i] for i in range(l)])

# ----------------------------

def gaussian_mixture(observations):
    """
    Accepts several sequences of (((value, .., ..) , (noise, .., ..)), ((..),(..)), ((..),(..)), ..),
    where for each element values and noises are tuples of float of equal length
     
    >>> print(gaussian_mixture([((10, 10), (10, 10)), ((20, 20), (20, 20))]))
    ((12.0, 12.0), (8.94427190999916, 8.94427190999916))
    >>> print(gaussian_mixture([((10, 10), (10, 20)), ((20, 20), (10, 20))]))
    ((15.0, 15.0), (7.0710678118654755, 14.142135623730951))
    >>> print(gaussian_mixture([((5, 5), (20, 20)), ((5, 5), (20, 20)), ((5, 5), (20, 20)), ((5, 5), (20, 20))]))
    ((5.0, 5.0), (10.000000000000002, 10.000000000000002))
    >>> print(gaussian_mixture([((5, 5), (10, 10)), ((5, 5), (15, 15)), ((5, 5), (20, 20)), ((5, 5), (25, 25))]))
    ((5.0, 5.0), (7.343330694720969, 7.343330694720969))
    >>> print(gaussian_mixture([((5, 5), (25, 25)), ((5, 5), (20, 20)), ((5, 5), (15, 15)), ((5, 5), (10, 10))]))
    ((5.0, 5.0), (7.3433306947209696, 7.3433306947209696))
    """
    val, noi = observations[0]
    values = list(val)
    noises = list(noi)
    for data in observations[1:]:
        val, noi = data
        values_ = list(val)
        noises_ = list(noi)
        for i in range(len(values)):
            var1 = noises[i] ** 2
            var2 = noises_[i] ** 2
            values[i] = (values[i] * var2 + values_[i] * var1) / (var1 + var2)
            noises[i] = math.sqrt((var1 * var2) / (var1 + var2))
    return tuple(values), tuple(noises)


def differentiate(values, order=1):
    """
    Creates n-th order differentiation of given values with order given by sign of argument order
    
    >>> print(differentiate([[2, 2], [4, 4], [7, 7], [8, 8]], 0))
    [[2, 2], [4, 4], [7, 7], [8, 8]]
    >>> print(differentiate([[2, 2], [4, 4], [7, 7], [8, 8]], 1))
    [[-2, -2], [-3, -3], [-1, -1]]
    >>> print(differentiate([[2, 2], [4, 4], [7, 7], [8, 8]], 2))
    [[1, 1], [-2, -2]]
    >>> print(differentiate([[2, 2], [4, 4], [7, 7], [8, 8]], 3))
    [[3, 3]]
    """
    if order == 0: return values
    sig = int(order / abs(order))
    return differentiate([[tupleize(p)[i] - tupleize(n)[i]
                           for i in range(len(tupleize(p)))]
                          for p, n in zip(values[:-1], values[1:])], order - sig)

# ----------------------------

def state_prediction(F, x, G, u):
    """
    F: state transition
    x: state estimate
    G: control transition
    u: control
    """
    return np.dot(F, x) + np.dot(G, u)


def state_covariance_prediction(F, P, Q):
    """
    F: state transition
    P: state estimate covariance
    Q: process noise distribution
    """
    return np.dot(F, np.dot(P, F.T)) + Q


def measurement_prediction(H, x):
    """
    H: observation transition
    x: state estimate
    s: predicted measurement
    """
    s = np.dot(H, x)
    return s


def measurement_covariance_prediction(H, P, R):
    """
    P: state estimate covariance
    H: observation transition
    R: observation noise
    S: innovation covariance
    """
    S = np.dot(H, np.dot(P, H.T)) + R
    S = np.linalg.inv(S)
    return S


def measurement_innovation(z, s):
    """
    z: actual measurement
    s: predicted measurement
    y: innovation
    """
    y = z - s
    return y


def measurement_gain(H, S, P):
    """
    H: observation transition
    S: innovation covariance
    P: state estimate covariance
    K: gain
    """
    K = np.dot(P, np.dot(H.T, S))
    return K


def state_correction(y, K, x):
    """
    y: innovation
    K: gain
    x: state estimate
    """
    return x + np.dot(K, y)


def state_covariance_correction(H, K, P):
    """
    H: observation transition
    y: innovation
    K: gain
    x: state estimate
    P: state estimate covariance
    """
    return np.dot((np.eye(P.shape[0]) - np.dot(K, H)), P)


def unscented_transform(w, xi, x, yi, y):
    """
    Basic unscented transform
    """
    m, n, o = len(xi), np.size(xi[0]), np.size(yi[0])
    res = np.zeros((n, o))
    for i in range(m):
        s = xi[i] - x
        t = yi[i] - y
        res += s * w[i] * t.T
    return res


def unscented_transform_mean(xi, wm):
    """
    Recovers mean by transformed sigma points.
    xi: sigma points
    wm: mean weights
    """
    return np.sum([xi[i] * wm[i] for i in range(len(xi))], axis=0)


def unscented_transform_sigma(xi, wc, mean, noise):
    """
    Recovers covariance by given sigma points and mean.
    xi:    sigma states
    wc:    covariance weights
    mean:  mean
    noise: noise
    """
    return unscented_transform(wc, xi, mean, xi, mean) + noise


def sigma_points(mu, sigma, alpha=1e-3, kappa=0, beta=2):
    """
    Creates sigma points given mean and covariance.
    mu:    mean
    sigma: covariance
    alpha: primary scaling factor; spead of sigma points
    kappa: secondary scaling factor
    beta:  incorporation of prior knowledge about distribution
    """
    n = np.size(mu)
    lmbda = alpha**2 * (n + kappa) - n
    wm = np.full((2 * n + 1, 1), 1 / (2 * (n + lmbda)))
    wc = np.copy(wm)
    xi = [np.zeros((n, 1)) for _ in range(2 * n + 1)]
    xi[0] = mu
    wm[0] = lmbda / (n + lmbda)
    wc[0] = wm[0] + (1 - alpha**2 + beta)
    Ps = np.linalg.cholesky((n + lmbda) * sigma).T  # yields matrix square root; n x n
    for k in range(n):
        ofs = Ps[k][:, np.newaxis]
        xi[k + 1] = mu + ofs
        xi[n + k + 1] = mu - ofs
    return xi, wm, wc

# ----------------------------

def oval(x0, y0, x1, y1, rotation=0, steps=12):
    """
    Returns an oval as list of coordinates (e.g. for canvas.create_poly).
    """
    # rotation is in degrees anti-clockwise, convert to radians
    rotation = -rotation * math.pi / 180.0
    # major and minor axes
    a = (x1 - x0) / 2.0
    b = (y1 - y0) / 2.0
    # center
    xc = x0 + a
    yc = y0 + b
    point_list = []
    # create the oval as a list of points
    for i in range(steps):
        # Calculate the angle for this step
        theta = (math.pi * 2) * (float(i) / steps)
        x1 = a * math.cos(theta)
        y1 = b * math.sin(theta)
        # rotate x, y
        x = (x1 * math.cos(rotation)) + (y1 * math.sin(rotation))
        y = (y1 * math.cos(rotation)) - (x1 * math.sin(rotation))
        point_list.append(round(x + xc))
        point_list.append(round(y + yc))
    return point_list

# ----------------------------

class Events:

    def __init__(self):
        self.__listener = {}

    def register(self, listener, events=None):
        if listener not in self.__listener:
            self.__listener[listener] = []
        if events is not None:
            if type(events) not in (list, tuple):
                events = (events,)
            [self.__listener[listener].append(event) for event in events]

    def unregister(self, listener=None):
        if listener is None:
            for listener, _ in self.__listener.items():
                del self.__listener[listener][:]
            self.__listener.clear()
        else:
            del self.__listener[listener][:]
            del self.__listener[listener]

    def notify(self, sender=None, event=None, msg=None):

        for listener, events in self.__listener.items():
            if event is None or len(events) == 0 or event in events:
                try:
                    listener(sender, event, msg)
                except TypeError as e:  # on error just delete this listener
                    eprint('exception: ', e, 'sender:', type(sender), 'event:', event, 'object:', msg)
                    pass
                    try:
                        self.__listener[listener].remove(event)
                    except ValueError as e:
                        eprint('exception: ', e, 'sender:', type(sender), 'event:', event, 'object:', msg)
                        pass
