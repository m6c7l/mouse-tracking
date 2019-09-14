#
# Copyright (c) 2017, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

import lib.utl as utl_
import random
import math
import enum
import numpy as np

# ----------------------------

events = utl_.Events()  # event system for sensor management and network traffic

# ----------------------------

class Net:
    """
    segmentation of emulated network traffic to avoid interference of emulated network traffic in event system
    """

    def __init__(self, origin=None):
        self.__name = ''
        if origin is not None:
            self.__name = str(id(origin))

    def id(self, value):
        if len(self.__name) > 0:
            return self.__name + '@' + str(value)
        return str(value)


class Clock:
    """
    time keeper
    """

    def __init__(self, net):

        self._net = net
        self._time = 0
        self._step = 0
        self._deltatime = 0
        self._running = False

    def time(self):
        return self._time

    def step(self):
        return self._step

    def running(self):
        return self._running

    def stop(self):
        self._running = False
        events.notify(self, event=self._net.id('clock:stop'), msg=(self._time, self._deltatime, self._step))

    def start(self):
        self._running = True
        events.notify(self, event=self._net.id('clock:start'), msg=(self._time, self._deltatime, self._step))

    def reset(self):
        events.notify(self, event=self._net.id('clock:reset'), msg=(self._time, self._deltatime, self._step))
        self._time = 0
        self._step = 0
        self._deltatime = 0

    def tick(self, dt_millis=None):
        if not self._running: return
        if dt_millis is not None:
            self._deltatime = dt_millis
        self._time += self._deltatime
        self._step += 1
        events.notify(self, event=self._net.id('clock:tick'), msg=(self._time, self._deltatime, self._step))

# ----------------------------

class Quantity(enum.Enum):
    """
    quantities
    """

    class POSITION: pass
    class VELOCITY: pass
    class ACCELERATION: pass
    class JERK: pass
    class SNAP: pass
    class YAW: pass
    class YAWRATE: pass
    class PITCH: pass
    class PITCHRATE: pass
    class CURVATURE: pass


class Utilization(enum.Enum):
    """
    either control or process
    """

    class PROCESS: pass
    class CONTROL: pass


class Progress(enum.Enum):
    """
    flags to give some loosely coupled methods a clue about the current progress
    """

    class PRIOR_PREDICT: pass
    class AFTER_PREDICT: pass
    class PRIOR_CORRECT: pass
    class AFTER_CORRECT: pass


class Step(enum.Enum):
    """
    flag to get some information about what is going on during estimation
    """

    class INITIALIZATION: pass
    class ADAPTION: pass
    class PREDICTION: pass
    class CORRECTION: pass

# ----------------------------

class Source:
    """
    interface for any source/sensor providing data
    """

    def __init__(self, net, name, quantity):

        self._net = net

        self._name = name
        if self._name is None: self._name = ''

        self._quantity = quantity

        self.__f_start = lambda ori, evt, msg: self._freeze(False)
        events.register(self.__f_start, events=self._net.id('clock:start'))

        self.__f_stop = lambda ori, evt, msg: self._freeze(True)
        events.register(self.__f_stop, events=self._net.id('clock:stop'))

        self.__f_tick = lambda ori, evt, msg: self._tick(msg)
        events.register(self.__f_tick, events=self._net.id('clock:tick'))

    def _freeze(self, value):
        raise NotImplementedError

    def _tick(self, value):
        raise NotImplementedError

    def name(self):
        return self._name

    def quantity(self):
        return self._quantity

    def active(self):
        raise NotImplementedError

    def dimension(self):
        raise NotImplementedError

    def source(self):
        raise NotImplementedError

    def destroy(self):
        events.unregister(self.__f_start)
        events.unregister(self.__f_stop)
        events.unregister(self.__f_tick)

    def __str__(self):
        return '{}'.format(self._name)


class Sensor(Source):
    """
    dimension-independent and abstract multi-purpose sensor
    """

    def __init__(self, net, name, noise, bias=None, interval=200):

        Source.__init__(self, net, name, Quantity.POSITION)

        self._interval = interval
        self._noise = utl_.tupleize(noise)

        self._dynamic_noise = False
        if None in self._noise:
            self._dynamic_noise = True

        if bias is None:
            bias = (0, ) * self.dimension()
        self._bias = utl_.tupleize(bias)

        len_diff = len(self._noise) - len(self._bias)
        if len_diff > 0:
            self._bias = self._bias + (0, ) * len_diff
        elif len_diff < 0:
            self._bias = self._bias[:len(self._noise)]
        self._invalid = float('nan')

        self._time = float('nan')
        self._signal = ()
        self._active = self._dynamic_noise  # all sensor with dynamic noise are active initially
        self._paused = False

        self.__f_signal = lambda ori, evt, msg: self.__signal(msg)
        events.register(self.__f_signal, events='world:signal')

    def destroy(self):
        Source.destroy(self)
        events.unregister(self.__f_signal)
        self._active = False
        self._signal = ()

    def active(self, value=None):
        if value is not None:
            self._active = value
            events.notify(self, event=self._net.id('sensor:status'), msg=self._time)
        return self._active

    def interval(self):
        return self._interval

    def dimension(self):
        return len(self._noise)

    def source(self):
        return None

    def bias(self):
        return self._bias

    def noise(self, value=None):
        if value is not None:
            for val in value:
                if abs(val) <= 0.001: return
            self._noise = tuple(value)[:self.dimension()]
            events.notify(self, event=self._net.id('sensor:change'), msg=self._time)
        return self._noise

    def blip(self):
        return self._time % self._interval == 0

    def _freeze(self, value):
        self._paused = value

    def _tick(self, value):
        t, *_ = value
        self._time = t
        self.__send(self._signal)
        self._signal = ()

    def __signal(self, msg):
        if self._paused: return
        if self._active:
            if not self._dynamic_noise:
                quan, val = msg
                noise = self._noise
                error = tuple([random.gauss(0, noise[i]) for i in range(self.dimension())])
            else:
                sen, val, noise, error = msg
                if sen != self: return
                quan = sen.quantity()
                self.noise(noise)
            if self.blip():  # would this "whatever" be receivable by this sensor due to sample rate?
                if quan == self._quantity:  # is this sensor an approriate sensor for "whatever"?
                    value = utl_.tupleize(val)[:self.dimension()]
                    value = list(value)
                    for i in range(self.dimension()):
                        try:
                            value[i] = value[i] + self._bias[i] + error[i]
                        except IndexError:
                            value[i] = self._invalid
                    self._signal = self._signal + ((self._time, tuple(value), noise),)  # put sensor data into queue for transmission

    def __send(self, signals):
        if len(signals) == 0:
            return  # no new signals in queue
        for item in signals:
            t, z, r = item
            packet = Packet(self, t, z, r)
            events.notify(self, event=self._net.id('sensor:data'), msg=packet)  # transmit sensor data


class Port(Source):
    """
    interface for filters for listening to sensor data
    """

    def __init__(self, net, name, quantity, dimension, timeout=10000):

        Source.__init__(self, net, name, quantity)

        self._timeout_value = timeout
        self._timeout = {}
        self._source = []

        self._dimension = dimension

        self._time = float('nan')

        self._active = False
        self._paused = False

        self.__f_source_status = lambda ori, evt, msg: self.__source_status(ori)
        events.register(self.__f_source_status, events=(self._net.id('sensor:status'),
                                                        self._net.id('sensor:change')))

        self.__f_source_data = lambda ori, evt, msg: self.__source_data(ori, msg)
        events.register(self.__f_source_data, events=(self._net.id('sensor:data'),
                                                      self._net.id('port:data')))

    def destroy(self):
        Source.destroy(self)
        events.unregister(self.__f_source_status)
        events.unregister(self.__f_source_data)
        self._active = False
        self._source.clear()

    def active(self, value=None):
        if value is not None:
            self._active = value
            events.notify(self, event=self._net.id('port:status'), msg=self._active)
        return self._active

    def dimension(self):
        return self._dimension

    def source(self):
        return tuple(self._source)

    def _freeze(self, value):
        self._paused = value

    def _tick(self, value):
        if self._paused: return
        t, dt, _ = value
        self._time = t
        to_delete = []
        for source in self._timeout:
            self._timeout[source] += dt
            if self._timeout[source] > self._timeout_value:
                to_delete.append(source)
        for source in to_delete:
            self.__detach(source)
            del self._timeout[source]
        if self._active and len(self._timeout) == 0:
            self._active = False

    def __source_status(self, source):
        if self._paused: return
        if source in self._source:
            events.notify(self, event=self._net.id('source:status'), msg=source)

    def __attach(self, source):
        if (source.dimension() == self.dimension()) and\
            (source.quantity() == self.quantity()) and\
            (source != self) and\
            (source not in self._source):
            self._source.append(source)
            events.notify(self, event=self._net.id('source:attach'), msg=source)

    def __detach(self, source):
        if source in self._source:
            idx = self._source.index(source)
            del self._source[idx]
            events.notify(self, event=self._net.id('source:detach'), msg=source)

    def __source_data(self, source, packet):
        if self._paused: return
        if source in self.source():
            events.notify(self, event=self._net.id('source:data'), msg=packet)
            self._timeout[source] = 0
            self._active = True
        else:
            self._timeout[source] = 0
            self.__attach(source)
            if source in self.source():
                self.__source_data(source, packet)

# ----------------------------

class State:

    def __init__(self, value):
        self._value = None
        State.set(self, value)

    def set(self, value):
        self._value = value
        if type(value) != np.ndarray:
            self._value = np.array(utl_.tupleize(value))[:, np.newaxis]

    def value(self, flat=False):
        if flat:
            return utl_.tupleize_nx1(self._value.flatten())
        else:
            return self._value[:]

    def dimension(self):
        return np.size(self._value)

    def __str__(self):
        return '{}'.format(self.value(True))


class Particle(State):

    def __init__(self, value, weight):
        State.__init__(self, value)
        self._weight = float(weight)

    def weight(self):
        return self._weight

    def __str__(self):
        return '{} {}'.format(State.__str__(self), self.weight())


class SigmaPoint(Particle):

    @classmethod
    def create(cls, mean, covariance):
        xi, wm, wc = utl_.sigma_points(mean, covariance)
        return [SigmaPoint(xi[k], wm[k], wc[k]) for k in range(len(xi))]

    @classmethod
    def decompose(cls, sigmas):
        return [s.value() for s in sigmas], [s.weight() for s in sigmas], [s.weight_covariance() for s in sigmas]

    def __init__(self, value, weight_mean, weight_cov):
        Particle.__init__(self, value, weight_mean)
        self._weight_cov = float(weight_cov)

    def weight_covariance(self):
        return self._weight_cov

    def __str__(self):
        return '{} {}'.format(Particle.__str__(self), self.weight_covariance())


class NoisyState(State):

    def __init__(self, value, noise):
        self._noise = None, None
        NoisyState.set(self, value, noise)

    def set(self, value, noise):
        State.set(self, value)
        self._noise = noise, noise
        if type(noise) != np.ndarray:
            noi = np.zeros((self._value.shape[0], self._value.shape[0]))
            self._noise = noi[:], noi[:]
            for j in range(2):
                for i in range(noi.shape[0]):
                    self._noise[j][i, i] = noise[i] ** (j + 1)

    def noise(self, flat=False, squared=True):
        if flat:
            return utl_.tupleize_nxn(self._noise[int(squared)])
        else:
            return self._noise[int(squared)][:]

    def __str__(self):
        return '{} {}'.format(State.__str__(self), self.noise(True))


class TimeState(NoisyState):

    def __init__(self, time, value, noise):
        NoisyState.__init__(self, value, noise)
        self._time = time

    def set(self, time, value, noise):
        NoisyState.set(self, value, noise)
        self._time = time

    def time(self, time=None):
        if time is not None:
            self._time = time
        return self._time

    def __eq__(self, other):
        if isinstance(other, TimeState): return self._time == other._time
        else: return NotImplemented

    def __lt__(self, other):
        if isinstance(other, TimeState): return self._time < other._time
        else: return NotImplemented

    def __gt__(self, other):
        if isinstance(other, TimeState): return self._time > other._time
        else: return NotImplemented

    def __str__(self):
        return '{} {}'.format(self.time(), NoisyState.__str__(self))


class Packet(TimeState):

    def __init__(self, source, time, value, noise):
        TimeState.__init__(self, time, value, noise)
        self._source = source
        events.notify(self, event='source:packet')

    def set(self, source, time, value, noise):
        return NotImplemented

    def source(self):
        return self._source

    def __str__(self):
        return '{} {} {}'.format(self.source(), self.time(), NoisyState.__str__(self))


class Estimate(TimeState):

    def __init__(self, process):
        self._quantities = process.quantities()
        self._dimensions = process.dimension()
        dim = sum(process.dimension())
        value = np.zeros((dim, 1)) * np.NaN
        noise = np.zeros((dim, dim)) + process.noise()
        TimeState.__init__(self, float('nan'), value, noise)

    def set(self, time, value, noise, originator=None, step=None):
        TimeState.set(self, time, value, noise)
        events.notify(self, event='estimate:change', msg=(originator, step))

    def replace(self, quantity, value, originator=None):
        _from, _to = self.__find(quantity)
        if _from != _to:
            value = utl_.tupleize(value)
            for i in range(len(value)):
                self._value[_from + i] = value[i]
            events.notify(self, event='estimate:modify', msg=(originator, quantity))
            return True
        return False

    def extract(self, quantity, flat=False):
        _from, _to = self.__find(quantity)
        if _from != _to:
            if flat:
                return self.value(True)[_from:_to], self.noise(True)[_from:_to]
            else:
                return self.value(False)[_from:_to, _from:_to], self.noise(False)[_from:_to, _from:_to]
        return None, None

    def __find(self, quantity):
        _from, _to = 0, 0
        for i, q in enumerate(self._quantities):
            _to += self._dimensions[i]
            if q == quantity:
                return _from, _to
            _from = _to
        return None, None


class Measurement(TimeState):

    def __init__(self, filter_, port_, packet_):

        TimeState.__init__(self, packet_.time(), packet_.value(), packet_.noise())

        dim_filter, dim_source = sum(filter_.process().dimension()), packet_.source().dimension()

        self._trans = np.zeros((dim_source, dim_filter))

        col = 0
        for p in filter_.ports():
            if packet_.source() in p.source():
                for i in range(packet_.source().dimension()):
                    self._trans[i, i + col] = 1
            col += p.dimension()

    def transition(self):
        return self._trans

    def mahalanobis(self, estimate):
        """
        calculates Mahalanobis distance by innovation;
        innovation is the difference of this measurement and a given estimate
        """
        x = estimate.value()
        P = estimate.noise()
        H = self._trans
        R = self.noise()
        z = self.value()
        S = np.linalg.inv(np.dot(H, np.dot(P, H.T)) + R)
        s = np.dot(H, x)
        y = z - s
        gate = np.sqrt(np.dot(np.dot(y.T, S), y))
        return gate

# ----------------------------

class Process:
    """
    abstract process model
    """

    def __init__(self, quantities, utilization, dimensions, dt_millis):

        self._dt, self._dt_millis = dt_millis / 1000.0, dt_millis

        self._dimensions = dimensions
        self._utilization = utilization
        self._quantities = quantities

        n = sum(self._dimensions)

        self._F = np.zeros((n, n))
        self._G = np.zeros((n, n))
        self._Q = np.zeros((n, n))

    def noise(self):
        return self._Q

    def deltatime(self):
        return self._dt, self._dt_millis

    def quantities(self):
        return self._quantities

    def dimension(self, quantity=None):
        if quantity is None:
            return self._dimensions
        else:
            try:
                idx = self._quantities.index(quantity)
                return self._dimensions[idx]
            except ValueError:
                return None

    def progress(self, step):
        """
        to be performed at various steps of process
        """
        raise NotImplementedError

    def reset(self, estimate, states):
        """
        reset process; (re)initialization
        """
        raise NotImplementedError

    def transition(self, utilization, value=None):
        """
        returns input/output state for prediction and correction or control.
        """
        if utilization == Utilization.PROCESS:
            return self.__build(Utilization.PROCESS, value), self._F
        elif utilization == Utilization.CONTROL:
            return self.__build(Utilization.CONTROL, value), self._G
        return None, None

    def __build(self, utilization, value):
        vec = None
        if value is not None:
            vec = np.zeros((value.shape[0], 1))
            from_, to_ = 0, 0
            for i, usage in enumerate(self._utilization):
                to_ = to_ + self._dimensions[i]
                if usage == utilization:
                    for j in range(from_, to_):
                        vec[j][0] = value[j][0]
                from_ = to_
        return vec

# ----------------------------

class Filter:
    """
    abstract Filter
    """

    def __init__(self, net, name, process):

        self._net = net

        self._ports = []
        for i, quantity in enumerate(process.quantities()):
            port = Port(net, name + ':' + str(len(self._ports)), quantity, process.dimension()[i])
            self._ports.append(port)

        self._process = process
        self._estimate = Estimate(process)

        self._initialized = False

        self._gate = None
        self._timeout = None
        self._adaptive = True

        self.__f_source_changed = lambda ori, evt, msg: self.__source_changed(ori, msg)
        events.register(self.__f_source_changed, events=(self._net.id('source:attach'),
                                                         self._net.id('source:detach'),
                                                         self._net.id('source:status')))

        self.__f_source_data = lambda ori, evt, msg: self.__source_data(ori, msg)
        events.register(self.__f_source_data, events=self._net.id('source:data'))

        self.__f_tick = lambda ori, evt, msg: self._tick(msg)
        events.register(self.__f_tick, events=self._net.id('clock:tick'))

    def destroy(self):
        events.unregister(self.__f_source_changed)
        events.unregister(self.__f_source_data)
        events.unregister(self.__f_tick)
        for port in self._ports: port.destroy()

    def _tick(self, value):
        t, _, _ = value
        _, dt = self._process.deltatime()
        if t % dt == 0:
            self.predict(dt, t)

    def initialize(self, packet=None, gate=None, timeout=None, adaptive=True):
        if packet is not None and not self._initialized:
            port = [p for p in self._ports if p.dimension() == packet.dimension() and p.quantity() == packet.source().quantity()]
            if len(port) > 0:
                self._estimate.replace(Quantity.POSITION, packet.value(True), self)
                self._gate = gate
                self._timeout = timeout
                self._adaptive = adaptive
                self._initialized = True
        return self._initialized

    def gate(self):
        return self._gate

    def timeout(self):
        return self._timeout

    def adaptive(self):
        return self._adaptive

    def predict(self, dt, t):
        raise NotImplementedError

    def correct(self):
        raise NotImplementedError

    def ports(self):
        return tuple(self._ports)

    def process(self):
        return self._process

    def estimate(self):
        return self._estimate

    def __source_data(self, port, packet):
        if port in self._ports and self._initialized:
            self._source_data(port, packet)

    def _source_data(self, port, packet):
        raise NotImplementedError

    def __source_changed(self, port, source):
        if port in self._ports and self._initialized:
            self._source_changed(port, source)

    def _source_changed(self, port, source):
        raise NotImplementedError


class KF(Filter):
    """
    Kalman filter
    """

    def __init__(self, net, process, origin_uncertain=False):

        self._origin_uncertain = origin_uncertain  # only for behaviourial and testing purpose

        Filter.__init__(self, net, str(id(self)) + ':' + self.__class__.__name__, process)

        self._data_cache = []  # used to initialize model
        self._data_queue = []  # used for correction

        self._data_accept = 0
        self._data_reject = 0

        self._gain = None

        self._running = False
        self._timer = 0

    def destroy(self):
        Filter.destroy(self)

    def __reset(self):
        self._initialized = False
        self._running = False
        self._timer = 0
        self._data_queue.clear()
        self._data_cache.clear()
        self._data_accept = 0
        self._data_reject = 0
        self._gain = None

    def predict(self, dt, t):
        """
        project state ahead; a priori (elder)
        """
        if not self._initialized: return
        self._timer += dt

        if self._timer > self._timeout:
            self.__reset()
            events.notify(self, event='filter:status', msg=(self._timeout, self._running))
            return

        if not self._running: return
        self.time_update(self._process, self._estimate)  # tbd in implementation

    def correct(self):
        """
        correct state; a posteriori (younger)
        """
        self.measurement_update(self._process, self._estimate, self._data_queue)  # tbd in implementation
        self._data_queue.clear()

    def time_update(self, process, estimate):
        raise NotImplementedError

    def measurement_update(self, process, estimate, measurements):
        raise NotImplementedError

    def _source_changed(self, port, source):
        pass

    def _source_data(self, port, packet):

        if not self._initialized: return

        if port.quantity() != Quantity.POSITION:  # filter accepts position measurements only
            events.notify(self, event='filter:invalid', msg=packet)
            return

        measurement = Measurement(self, port, packet)

        if not self._running:  # filter in initialization

            p1, _ = self._estimate.extract(Quantity.POSITION, True)
            p2 = measurement.value(True)

            d = utl_.euclidean_distance(p1, p2)

            rng = utl_.euclidean_distance(packet.source().noise()) * 3  # three times Euclidean deviation for range addon

            if d < rng + self._gate or not self._origin_uncertain:  # measurement is in vicinity of initial estimate, therefore accept measurement

                self._estimate.replace(Quantity.POSITION, p2)

                if self._add_cache(measurement):  # enough to initialize

                    init = []
                    segments = self.__div_cache(self._process.reset())
                    for segment in segments:
                        if len(segment) > 0:
                            t = sum([item.time() for item in segment]) / len(segment)
                            x = utl_.gaussian_mixture([(item.value(True), item.noise(True)) for item in segment])
                            init.append(TimeState(t, x[0], [math.sqrt(i) for i in x[1]]))

                    if self._process.reset() == len(init):
                        self._process.reset(self._estimate, init)  # kick it into being
                        self._running = True
                        events.notify(self, event='filter:status', msg=(self._timeout, self._running))

                self._timer = 0
                events.notify(self, event='filter:apply', msg=measurement)

            else:
                events.notify(self, event='filter:drop', msg=measurement)

        else:  # filter in estimation mode

            d = measurement.mahalanobis(self._estimate)

            dmax = 0  # chi2
            if measurement.dimension() == 2:  # df
                dmax = 5.99  # p = 0.05
            elif measurement.dimension() == 3:  # df
                dmax = 7.82  # p = 0.05

            if d < dmax or not self._origin_uncertain:  # gate size

                if self._adaptive and d > dmax * 0.33:  # possible (slight) manoeuver; add some noise to process
                    est = self._estimate
                    t, v, n = est.time(), est.value(), est.noise()
                    n = n * (1.33 + (d / dmax))
                    est.set(t, v, n, self, Step.ADAPTION)

                self._add_cache(measurement)
                self._data_queue.append(measurement)

                self.correct()

                self._data_accept += 1
                self._timer = 0
                events.notify(self, event='filter:accept', msg=measurement)

            else:
                self._data_reject += 1
                events.notify(self, event='filter:reject', msg=measurement)

    def path(self):
        result = []
        if not self._running:
            return result
        count = self.__min_cache()
        segments = self.__div_cache(count)
        for segment in segments:
            x = utl_.gaussian_mixture([(item.value(True), item.noise(True)) for item in segment])
            result.append(x[0])
        return result[::-1]

    def measurements(self):
        return self._data_cache

    def gain(self):
        return self._gain

    def _add_cache(self, m):
        initialized = False
        size = self._process.reset()
        sens = sum([len(p.source()) for p in self.ports()])
        n = (size * sens) * 2 + 1  # max cache size
        if len(self._data_cache) < n:
            self._data_cache.append(m)
        else:
            self._data_cache[-1] = m
        self._data_cache.sort(reverse=True)
        lengths = [len(i) for i in self.__div_cache(size)]
        if min(lengths) != 0:
            initialized = True
        return initialized

    def __get_cache(self, age_from, age_to):
        cache = self._data_cache
        t1 = cache[0].time() - age_from
        t2 = cache[0].time() - age_to
        result = []
        for elem in cache:
            if t2 < elem.time() <= t1:
                result.append(elem)
        return result

    def __div_cache(self, segments):
        cache = self._data_cache
        tmin, tmax = cache[-1].time(), cache[0].time()
        interval = (tmax - tmin) / segments
        res = []
        for i in range(segments):
            res.append(self.__get_cache(interval * i, interval * (i + 1)))
        return res

    def __min_cache(self):
        count = 1
        lengths = [len(i) for i in self.__div_cache(count + 1)]
        while min(lengths) > 0:
            count += 1
            lengths = [len(i) for i in self.__div_cache(count + 1)]
        return count


class EKF(KF):
    """
    Extended Kalman Filter
    """

    def __init__(self, net, process, origin_uncertain):
        KF.__init__(self, net, process, origin_uncertain)

    def time_update(self, process, estimate):
        process.progress(estimate, Progress.PRIOR_PREDICT)
        t, x, P = estimate.time(), estimate.value(), estimate.noise()
        _, dt_millis = process.deltatime()
        v, F = process.transition(Utilization.PROCESS, x)
        u, G = process.transition(Utilization.CONTROL, x)
        x = utl_.state_prediction(F, v, G, u)
        P = utl_.state_covariance_prediction(F, P, process.noise())
        estimate.set(t + dt_millis, x, P, self, Step.PREDICTION)
        process.progress(estimate, Progress.AFTER_PREDICT)

    def measurement_update(self, process, estimate, measurements):
        process.progress(estimate, Progress.PRIOR_CORRECT)
        x, P = estimate.value(), estimate.noise()
        for m in measurements:
            z, R, H = m.value(), m.noise(), m.transition()
            s = utl_.measurement_prediction(H, x)
            S = utl_.measurement_covariance_prediction(H, P, R)
            y = utl_.measurement_innovation(z, s)
            K = utl_.measurement_gain(H, S, P)
            x = utl_.state_correction(y, K, x)
            P = utl_.state_covariance_correction(H, K, P)
            estimate.set(m.time(), x, P, self, Step.CORRECTION)
            self._gain = K
        process.progress(estimate, Progress.AFTER_CORRECT)


class UKF(KF):
    """
    Unscented Kalman Filter (unscented transform in prediction and correction)
    """

    def __init__(self, net, process, origin_uncertain):
        KF.__init__(self, net, process, origin_uncertain)

    def time_update(self, process, estimate):
        _, dt_millis = process.deltatime()
        t, x, P = estimate.time(), estimate.value(), estimate.noise()
        X = SigmaPoint.create(x, P)  # xi, wm, wc
        for sigma in X:
            process.progress(sigma, Progress.PRIOR_PREDICT)
            v, F = process.transition(Utilization.PROCESS, sigma.value())
            u, G = process.transition(Utilization.CONTROL, sigma.value())
            sigma.set(utl_.state_prediction(F, v, G, u))
            process.progress(sigma, Progress.AFTER_PREDICT)
        xi, wm, wc = SigmaPoint.decompose(X)
        x = utl_.unscented_transform_mean(xi, wm)
        P = utl_.unscented_transform_sigma(xi, wc, x, process.noise())
        estimate.set(t + dt_millis, x, P, self, Step.PREDICTION)

    def measurement_update(self, process, estimate, measurements):
        process.progress(estimate, Progress.PRIOR_CORRECT)
        for m in measurements:
            x, P = estimate.value(), estimate.noise()
            z, R, H = m.value(), m.noise(), m.transition()
            X = SigmaPoint.create(x, P)  # xi, wm, wc
            zi = [None] * len(X)
            for i, sigma in enumerate(X):
                zi[i] = utl_.measurement_prediction(H, sigma.value())
            xi, wm, wc = SigmaPoint.decompose(X)
            s = utl_.unscented_transform_mean(zi, wm)
            y = utl_.measurement_innovation(z, s)
            S = utl_.unscented_transform_sigma(zi, wc, s, R)
            K = np.dot(utl_.unscented_transform(wc, xi, x, zi, s), np.linalg.inv(S))
            x = utl_.state_correction(y, K, x)
            P = P - np.dot(np.dot(K, S), K.T)
            estimate.set(m.time(), x, P, self, Step.CORRECTION)
            self._gain = K
        process.progress(estimate, Progress.AFTER_CORRECT)

# ----------------------------

class BM(Process):
    """
    Brownian Motion Model (linear)
    """

    def __init__(self, noise, dt_millis):

        Process.__init__(self,
                         (Quantity.POSITION,),
                         (Utilization.PROCESS,) * 1,
                         (2,) * 1,
                         dt_millis)
        order = (2,)

        dt, _ = self.deltatime()

        # process transition
        self._F = np.eye(self._F.shape[0]) * 1

        # process noise
        dim = self.dimension()
        q = []
        for i in range(len(order)):
            for d in range(dim[i]):
                q.append(order[i])
        n = dim[0]
        for i in range(len(q)):
            for j in range(self._Q.shape[0]):
                if abs(i - j) % n == 0:
                    self._Q[i, j] = (dt ** (q[i] + q[j])) / (q[i] * q[j])
        self._Q = self._Q * noise ** 2

    def reset(self, estimate=None, states=None):
        if states is None: return 1
        states = utl_.tupleize(states)
        if len(states) != 1: raise ValueError
        t = [s.time() for s in states]
        pos = [[v for v in s.value(True)] for s in states]
        pos = [p for p in pos[0]]
        record = pos
        noi = [[n for n in s.noise(True)] for s in states]
        noi = np.sum(noi) / len(states)
        noi = noi**2
        estimate.set(t[0], record, np.eye(self._Q.shape[0]) * noi, None, Step.INITIALIZATION)

    def progress(self, state, step):
        """
        linear model, no adjustment
        """
        pass


class CV(Process):
    """
    Constant Velocity Model (linear); 2D and 3D
    """

    def __init__(self, noise, dt_millis, dimension=2):

        Process.__init__(self,
                         (Quantity.POSITION, Quantity.VELOCITY,),
                         (Utilization.PROCESS,) * 2,
                         (dimension,) * 2,
                         dt_millis)
        order = (2, 1)

        dt, _ = self.deltatime()

        # time derivatives
        vdt = (dt**1) / 1

        # process transition
        self._F = np.eye(self._F.shape[0]) * 1
        for i in range(dimension):
            self._F[i, dimension + i] = vdt

        # process noise
        dim = self.dimension()
        q = []
        for i in range(len(order)):
            for d in range(dim[i]):
                q.append(order[i])
        n = dim[0]
        for i in range(len(q)):
            for j in range(self._Q.shape[0]):
                if abs(i - j) % n == 0:
                    self._Q[i, j] = (dt ** (q[i] + q[j])) / (q[i] * q[j])
        self._Q = self._Q * noise ** 2

    def reset(self, estimate=None, states=None):
        if states is None: return 2
        states = utl_.tupleize(states)
        if len(states) != 2: raise ValueError
        t = [s.time() for s in states]
        dt = [i[0] for i in utl_.differentiate(t, 1)]
        dt = sum(dt) / len(dt) / 1000
        pos = [[v for v in s.value(True)] for s in states]
        vel = utl_.differentiate(pos, 1)
        pos = [p for p in pos[0]]
        vel = [v / dt for v in vel[0]]
        record = pos + vel
        noi = [[n for n in s.noise(True)] for s in states]
        noi = np.sum(noi) / len(states)
        noi = noi**2
        estimate.set(t[0], record, np.eye(self._Q.shape[0]) * noi, None, Step.INITIALIZATION)

    def progress(self, state, step):
        """
        linear model, no adjustment
        """
        pass


class CT(CV):
    """
    Coordinate Turn Model (linear), if w==0 -> CV; 2D only
    """

    def __init__(self, noise, angle, dt_millis):

        CV.__init__(self, noise, dt_millis, 2)

        dt, _ = self.deltatime()

        # time derivatives
        vdt = (dt**1) / 1

        # curvilinear process transition
        self._F = np.eye(self._F.shape[0]) * 1
        if abs(angle) > 0.0001:

            w = math.radians(angle)
            sinwdt = math.sin(w * vdt)
            coswdt = math.cos(w * vdt)

            self._F[0, 2] = sinwdt / w
            self._F[0, 3] = (-coswdt + 1) / w
            self._F[1, 2] = -(-coswdt + 1) / w
            self._F[1, 3] = sinwdt / w
            self._F[2, 2] = coswdt
            self._F[2, 3] = sinwdt
            self._F[3, 2] = -sinwdt
            self._F[3, 3] = coswdt


class CA(Process):
    """
    Constant Acceleration Model (linear); 2D and 3D
    """

    def __init__(self, noise, dt_millis, dimension=2):

        Process.__init__(self,
                         (Quantity.POSITION, Quantity.VELOCITY, Quantity.ACCELERATION),
                         (Utilization.PROCESS,) * 3,
                         (dimension,) * 3,
                         dt_millis)

        order = (2, 1.5, 1)

        dt, _ = self.deltatime()

        # time derivatives
        vdt = (dt**1) / 1
        adt = (dt**2) / 2

        # process transition
        self._F = np.eye(self._F.shape[0]) * 1
        for i in range(dimension):
            self._F[i, dimension + i] = vdt
            self._F[dimension + i, 2 * dimension + i] = vdt
            self._F[i, 2 * dimension + i] = adt

        # process noise
        dim = self.dimension()
        q = []
        for i in range(len(order)):
            for d in range(dim[i]):
                q.append(order[i])
        n = dim[0]
        for i in range(len(q)):
            for j in range(self._Q.shape[0]):
                if abs(i - j) % n == 0:
                    self._Q[i, j] = (dt ** (q[i] + q[j])) / (q[i] * q[j])
        self._Q = self._Q * noise ** 2

    def reset(self, estimate=None, states=None):
        if states is None: return 3
        states = utl_.tupleize(states)
        if len(states) != 3: raise ValueError
        t = [s.time() for s in states]
        dt = [i[0] for i in utl_.differentiate(t, 1)]
        dt = sum(dt) / len(dt) / 1000
        pos = [[v for v in s.value(True)] for s in states]
        vel = utl_.differentiate(pos, 1)
        acc = utl_.differentiate(vel, 1)
        pos = [p for p in pos[0]]
        vel = [v / dt for v in vel[0]]
        acc = [0 for a in acc[0]]
        record = pos + vel + acc
        noi = [[n for n in s.noise(True)] for s in states]
        noi = np.sum(noi) / len(states)
        noi = noi**2
        estimate.set(t[0], record, np.eye(self._Q.shape[0]) * noi, None, Step.INITIALIZATION)

    def progress(self, state, step):
        """
        linear model, no adjustment
        """
        pass


class CJ(Process):
    """
    Constant Jerk Model (linear); 2D and 3D
    """

    def __init__(self, noise, dt_millis, dimension=2):

        Process.__init__(self,
                         (Quantity.POSITION, Quantity.VELOCITY, Quantity.ACCELERATION, Quantity.JERK),
                         (Utilization.PROCESS,) * 4,
                         (dimension,) * 4,
                         dt_millis)

        order = (2, 1.67, 1.33, 1)

        dt, _ = self.deltatime()

        # time derivatives
        vdt = (dt**1) / 1
        adt = (dt**2) / 2
        jdt = (dt**3) / 6

        # process transition
        self._F = np.eye(self._F.shape[0]) * 1
        for i in range(dimension):
            self._F[i, dimension + i] = vdt
            self._F[dimension + i, 2 * dimension + i] = vdt
            self._F[2 * dimension + i, 3 * dimension + i] = vdt
            self._F[i, 2 * dimension + i] = adt
            self._F[dimension + i, 3 * dimension + i] = adt
            self._F[i, 3 * dimension + i] = jdt

        # process noise
        dim = self.dimension()
        q = []
        for i in range(len(order)):
            for d in range(dim[i]):
                q.append(order[i])
        n = dim[0]
        for i in range(len(q)):
            for j in range(self._Q.shape[0]):
                if abs(i - j) % n == 0:
                    self._Q[i, j] = (dt ** (q[i] + q[j])) / (q[i] * q[j])
        self._Q = self._Q * noise ** 2

    def reset(self, estimate=None, states=None):
        if states is None: return 4
        states = utl_.tupleize(states)
        if len(states) != 4: raise ValueError
        t = [s.time() for s in states]
        dt = [i[0] for i in utl_.differentiate(t, 1)]
        dt = sum(dt) / len(dt) / 1000
        pos = [[v for v in s.value(True)] for s in states]
        vel = utl_.differentiate(pos, 1)
        acc = utl_.differentiate(vel, 1)
        jer = utl_.differentiate(acc, 1)
        pos = [p for p in pos[0]]
        vel = [v / dt for v in vel[0]]
        acc = [a / dt for a in acc[0]]
        jer = [0 for j in jer[0]]
        record = pos + vel + acc + jer
        noi = [[n for n in s.noise(True)] for s in states]
        noi = np.sum(noi) / len(states)
        noi = noi**2
        estimate.set(t[0], record, np.eye(self._Q.shape[0]) * noi, None, Step.INITIALIZATION)

    def progress(self, state, step):
        """
        linear model, no adjustment
        """
        pass


class CTRV(Process):
    """
    Augmented Coordinate Turn Model, polar with velocity (non-linear); 2D only
    """

    def __init__(self, noise, dt_millis):

        Process.__init__(self,
                         (Quantity.POSITION, Quantity.VELOCITY, Quantity.YAW, Quantity.YAWRATE),
                         (Utilization.PROCESS, Utilization.PROCESS, Utilization.CONTROL, Utilization.CONTROL,),
                         (2, 1, 1, 1),
                         dt_millis)

        dt, _ = self.deltatime()

        order = (2, 1, 2, 1)

        # process noise
        dim = self.dimension()
        q = []
        for i in range(len(order)):
            for d in range(dim[i]):
                q.append(order[i])
        n = dim[0]
        for i in range(len(q)):
            for j in range(self._Q.shape[0]):
                if abs(i - j) % n == 0:
                    self._Q[i, j] = (dt ** (q[i] + q[j])) / (q[i] * q[j])
        self._Q = self._Q * noise ** 2

        # control input
        self._G[3, 4] = dt
        self._G[3, 3] = 1
        self._G[4, 4] = 1

    def reset(self, estimate=None, states=None):
        if states is None: return 2
        states = utl_.tupleize(states)
        if len(states) != 2: raise ValueError
        t = [s.time() for s in states]
        dt = [i[0] for i in utl_.differentiate(t, 1)]
        dt = sum(dt) / len(dt) / 1000
        pos = [[v for v in s.value(True)] for s in states]
        vel = utl_.differentiate(pos, 1)
        pos = [p for p in pos[0]]
        vel = [v / dt for v in vel[0]]
        yaw, vel = utl_.polar(*vel)
        record = pos + [vel, yaw, 0,]
        # noi = [[n for n in s.noise(True)] for s in states]
        # noi = np.sum(noi) / len(states)
        # noi = noi**2
        estimate.set(t[0], record, np.eye(self._Q.shape[0]) * dt, None, Step.INITIALIZATION)

    def progress(self, state, step):
        if step == Progress.PRIOR_PREDICT:
            x, y, v, h, w = state.value(True)
            State.set(state, (x, y, v, utl_.radians_squeeze(h), 0))
            self.__create_transition(state, type(state) != SigmaPoint)
        elif step == Progress.AFTER_PREDICT:
            x, y, v, h, w = state.value(True)
            State.set(state, (x, y, v, utl_.radians_squeeze(h), utl_.radians_squeeze(w)))

    def __create_transition(self, state, linearize=True):  # jacobian

        x, y, v, h, w = state.value(True)
        dt, _ = self.deltatime()

        if abs(w) < 0.0001 or not linearize:  # w == 0 or sigma point

            cosh = np.cos(h)
            sinh = np.sin(h)

            self._F = np.eye(self._F.shape[0])
            self._F[0, 2] = float(dt * sinh)
            self._F[0, 3] = float(dt * v * cosh)
            self._F[1, 2] = float(dt * cosh)
            self._F[1, 3] = float(-dt * v * sinh)

        else:  # w != 0

            # variante 1
            sindtw = np.sin((dt * w) / 2)
            sindtwh = np.sin((dt * w) / 2 + h)
            cosdtw = np.cos((dt * w) / 2)
            cosdtwh = np.cos((dt * w) / 2 + h)

            self._F = np.eye(self._F.shape[0])
            self._F[0, 2] = float(2 * sindtw * sindtwh / w)
            self._F[0, 3] = float(2 * v * sindtw * cosdtwh / w)
            self._F[0, 4] = float((dt * v * sindtw * cosdtwh / w) + (dt * v * sindtwh * cosdtw / w) - (2 * v * sindtw * sindtwh / w ** 2))
            self._F[1, 2] = float(2 * sindtw * cosdtwh / w)
            self._F[1, 3] = float(-2 * v * sindtw * sindtwh / w)
            self._F[1, 4] = float(-(dt * v * sindtw * sindtwh / w) + (dt * v * cosdtw * cosdtwh / w) - (2 * v * sindtw * cosdtwh / w ** 2))
            self._F[3, 4] = float(dt)


class CTRA(Process):
    """
    Augmented Coordinate Turn Model, polar with velocity and acceleration (non-linear); 2D only
    """

    def __init__(self, noise, dt_millis):

        Process.__init__(self,
                         (Quantity.POSITION, Quantity.VELOCITY, Quantity.ACCELERATION, Quantity.YAW, Quantity.YAWRATE),
                         (Utilization.PROCESS, Utilization.PROCESS, Utilization.PROCESS, Utilization.CONTROL, Utilization.CONTROL,),
                         (2, 1, 1, 1, 1),
                         dt_millis)

        dt, _ = self.deltatime()

        order = (2, 1.5, 1, 2, 1)

        # process noise
        dim = self.dimension()
        q = []
        for i in range(len(order)):
            for d in range(dim[i]):
                q.append(order[i])
        n = dim[0]
        for i in range(len(q)):
            for j in range(self._Q.shape[0]):
                if abs(i - j) % n == 0:
                    self._Q[i, j] = (dt ** (q[i] + q[j])) / (q[i] * q[j])
        self._Q = self._Q * noise ** 2

        # control input
        self._G[4, 5] = dt
        self._G[4, 4] = 1
        self._G[5, 5] = 1

    def reset(self, estimate=None, states=None):
        if states is None: return 3
        states = utl_.tupleize(states)
        if len(states) != 3: raise ValueError
        t = [s.time() for s in states]
        dt = [i[0] for i in utl_.differentiate(t, 1)]
        dt = sum(dt) / len(dt) / 1000
        pos = [[v for v in s.value(True)] for s in states]
        vel = utl_.differentiate(pos, 1)
        acc = utl_.differentiate(vel, 1)
        pos = [p for p in pos[0]]
        vel = [v / dt for v in vel[0]]
        acc = [a / dt for a in acc[0]]
        yaw, vel = utl_.polar(*vel)
        acc = utl_.euclidean_distance(acc)
        record = pos + [vel, acc, yaw, 0,]
        # noi = [[n for n in s.noise(True)] for s in states]
        # noi = np.sum(noi) / len(states)
        # noi = noi**2
        estimate.set(t[0], record, np.eye(self._Q.shape[0]) * dt, None, Step.INITIALIZATION)

    def progress(self, state, step):
        if step == Progress.PRIOR_PREDICT:
            self.__create_transition(state, type(state) != SigmaPoint)
            x, y, v, a, h, w = state.value(True)
            State.set(state, (x, y, v, a, utl_.radians_squeeze(h), 0))  # no yaw rate impact (not measured)
        elif step == Progress.AFTER_PREDICT:
            x, y, v, a, h, w = state.value(True)
            State.set(state, (x, y, v, a, utl_.radians_squeeze(h), utl_.radians_squeeze(w)))

    def __create_transition(self, state, linearize=True):  # jacobian

        x, y, v, a, h, w = state.value(True)
        dt, _ = self.deltatime()

        cosh = np.cos(h)
        sinh = np.sin(h)

        if abs(w) < 0.0001 or not linearize:  # w == 0 or sigma point

            self._F = np.eye(self._F.shape[0])
            self._F[0, 2] = float(dt * sinh)
            self._F[0, 3] = float((dt**2 * sinh) / 2)
            self._F[0, 4] = float(dt * ((a * dt) / 2 + v) * cosh)
            self._F[1, 2] = float(dt * cosh)
            self._F[1, 3] = float((dt**2 * cosh) / 2)
            self._F[1, 4] = float(-dt * ((a * dt) / 2 + v) * sinh)
            self._F[2, 3] = dt

        else:  # w != 0

            dtwh = dt * w + h
            sindtwh = np.sin(dtwh)
            cosdtwh = np.cos(dtwh)
            adtw = a * dt * w
            vw = v * w

            self._F = np.eye(self._F.shape[0])
            self._F[0, 2] = float((w * cosh - w * cosdtwh) / w**2)
            self._F[0, 3] = float((-dt * w * cosdtwh - sinh + sindtwh) / w**2)
            self._F[0, 4] = float((-a * cosh + a * cosdtwh - vw * sinh - (-adtw - vw) * sindtwh) / w**2)
            self._F[0, 5] = float((a * dt * cosdtwh - dt * (-adtw - vw) * sindtwh + v * cosh + (-a * dt - v) * cosdtwh) / w**2 - (2 * (-a * sinh + a * sindtwh + vw * cosh + (-adtw - vw) * cosdtwh)) / w**3)
            self._F[1, 2] = float((-w * sinh + w * sindtwh) / w**2)
            self._F[1, 3] = float((dt * w * sindtwh - cosh + cosdtwh) / w**2)
            self._F[1, 4] = float((a * sinh - a * sindtwh - vw * cosh + (adtw + vw) * cosdtwh) / w**2)
            self._F[1, 5] = float((-a * dt * sindtwh + dt * (adtw + vw) * cosdtwh - v * sinh + (a * dt + v) * sindtwh) / w**2 - (2 * (-a * cosh + a * cosdtwh - vw * sinh + (adtw + vw) * sindtwh)) / w**3)
            self._F[2, 3] = dt
            self._F[4, 5] = dt

# ----------------------------

class CYRPRV(Process):
    """
    Constant Yaw Rate Pitch Rate Velocity Model (custom model); 3D only
    """

    def __init__(self, noise, dt_millis):

        Process.__init__(self,
                         (Quantity.POSITION, Quantity.VELOCITY,
                          Quantity.YAW, Quantity.PITCH,
                          Quantity.YAWRATE, Quantity.PITCHRATE,),
                         (Utilization.PROCESS, Utilization.PROCESS,
                          Utilization.CONTROL, Utilization.CONTROL,
                          Utilization.CONTROL, Utilization.CONTROL,),
                         (3, 1,
                          1, 1,
                          1, 1),
                         dt_millis)

        dt, _ = self.deltatime()

        order = (2, 1, 2, 2, 1, 1)

        # process noise
        dim = self.dimension()
        q = []
        for i in range(len(order)):
            for d in range(dim[i]):
                q.append(order[i])
        n = dim[0]
        for i in range(len(q)):
            for j in range(self._Q.shape[0]):
                if abs(i - j) % n == 0:
                    self._Q[i, j] = (dt ** (q[i] + q[j])) / (q[i] * q[j])
        self._Q = self._Q * noise ** 2

        # control input
        self._G[4, 6] = dt
        self._G[5, 7] = dt
        for i in range(4, 8):
            self._G[i, i] = 1

    def reset(self, estimate=None, states=None):
        if states is None: return 2
        states = utl_.tupleize(states)
        if len(states) != 2: raise ValueError
        t = [s.time() for s in states]
        dt = [i[0] for i in utl_.differentiate(t, 1)]
        dt = sum(dt) / len(dt) / 1000
        pos = [[v for v in s.value(True)] for s in states]
        vel = utl_.differentiate(pos, 1)
        pos = [p for p in pos[0]]
        vel = [v / dt for v in vel[0]]
        yaw, v = utl_.polar(*vel[:2])
        pit = (vel[2] / abs(sum(vel[:2]))) * (math.pi / 2)
        record = pos + [v, yaw, pit, 0, 0]
        # noi = [[n for n in s.noise(True)] for s in states]
        # noi = np.sum(noi) / len(states)
        # noi = noi**2
        estimate.set(t[0], record, np.eye(self._Q.shape[0]) * dt, None, Step.INITIALIZATION)

    def progress(self, state, step):
        if step == Progress.PRIOR_PREDICT:
            self.__create_transition(state, type(state) != SigmaPoint)
            x, y, z, v, h, p, wh, wp  = state.value(True)
            State.set(state, (x, y, z, v, utl_.radians_squeeze(h), utl_.radians_squeeze(p), 0, 0))
        elif step == Progress.AFTER_PREDICT:
            x, y, z, v, h, p, wh, wp  = state.value(True)
            State.set(state, (x, y, z, v, utl_.radians_squeeze(h), utl_.radians_squeeze(p), utl_.radians_squeeze(wh), utl_.radians_squeeze(wp)))

    def __create_transition(self, state, linearize=True):

        x, y, z, v, h, p, wh, wp = state.value(True)
        dt, _ = self.deltatime()

        sy = math.sin(h)
        cy = math.cos(h)
        sp = math.sin(p)
        cp = math.cos(p)

        self._F = np.eye(self._F.shape[0])
        self._F[0, 3] = float(dt * sy * cp)
        self._F[0, 4] = float(dt * v * cp * cy)
        self._F[0, 5] = float(-dt * v * sp * sy)
        self._F[1, 3] = float(dt * cp * cy)
        self._F[1, 4] = float(-dt * v * sy * cp)
        self._F[1, 5] = float(-dt * v * sp * cy)
        self._F[2, 3] = float(dt * sp)
        self._F[2, 5] = float(dt * v * cp)
        self._F[4, 6] = dt
        self._F[5, 7] = dt


class CYRPRA(Process):
    """
    Constant Yaw Rate Pitch Rate Acceleration Model (custom model); 3D only
    """

    def __init__(self, noise, dt_millis):

        Process.__init__(self,
                         (Quantity.POSITION, Quantity.VELOCITY, Quantity.ACCELERATION,
                          Quantity.YAW, Quantity.PITCH,
                          Quantity.YAWRATE, Quantity.PITCHRATE,),
                         (Utilization.PROCESS, Utilization.PROCESS, Utilization.PROCESS,
                          Utilization.CONTROL, Utilization.CONTROL,
                          Utilization.CONTROL, Utilization.CONTROL,),
                         (3, 1, 1,
                          1, 1,
                          1, 1),
                         dt_millis)

        dt, _ = self.deltatime()

        order = (2, 1.5, 1, 2, 2, 1, 1)

        # process noise
        dim = self.dimension()
        q = []
        for i in range(len(order)):
            for d in range(dim[i]):
                q.append(order[i])
        n = dim[0]
        for i in range(len(q)):
            for j in range(self._Q.shape[0]):
                if abs(i - j) % n == 0:
                    self._Q[i, j] = (dt ** (q[i] + q[j])) / (q[i] * q[j])
        self._Q = self._Q * noise ** 2

        # control input
        self._G[3, 4] = dt
        self._G[5, 7] = dt
        self._G[6, 8] = dt
        for i in range(3, 9):
            self._G[i, i] = 1

    def reset(self, estimate=None, states=None):
        if states is None: return 3
        states = utl_.tupleize(states)
        if len(states) != 3: raise ValueError
        t = [s.time() for s in states]
        dt = [i[0] for i in utl_.differentiate(t, 1)]
        dt = sum(dt) / len(dt) / 1000
        pos = [[v for v in s.value(True)] for s in states]
        vel = utl_.differentiate(pos, 1)
        acc = utl_.differentiate(vel, 1)
        pos = [p for p in pos[0]]
        vel = [v / dt for v in vel[0]]
        acc = [a * dt**2 / 2 for a in acc[0]]
        a = utl_.euclidean_distance((0,) * len(acc), acc)
        yaw, v = utl_.polar(*vel[:2])
        pit = (vel[2] / abs(sum(vel[:2]))) * (math.pi / 2)
        record = pos + [v, a, yaw, pit, 0, 0]
        # noi = [[n for n in s.noise(True)] for s in states]
        # noi = np.sum(noi) / len(states)
        # noi = noi**2
        estimate.set(t[0], record, np.eye(self._Q.shape[0]) * dt, None, Step.INITIALIZATION)

    def progress(self, state, step):
        if step == Progress.PRIOR_PREDICT:
            self.__create_transition(state, type(state) != SigmaPoint)
            x, y, z, v, a, h, p, wh, wp  = state.value(True)
            State.set(state, (x, y, z, v, a, utl_.radians_squeeze(h), utl_.radians_squeeze(p), 0, 0))
        elif step == Progress.AFTER_PREDICT:
            x, y, z, v, a, h, p, wh, wp  = state.value(True)
            State.set(state, (x, y, z, v, a, utl_.radians_squeeze(h), utl_.radians_squeeze(p), utl_.radians_squeeze(wh), utl_.radians_squeeze(wp)))

    def __create_transition(self, state, linearize=True):

        x, y, z, v, a, yaw, pit, yr, pr = state.value(True)
        dt, _ = self.deltatime()

        sy = math.sin(yaw)
        cy = math.cos(yaw)
        sp = math.sin(pit)
        cp = math.cos(pit)

        self._F = np.eye(self._F.shape[0])
        self._F[0, 3] = float(dt * sy * cp)
        self._F[0, 4] = float((dt**2 * sy * cp) / 2)
        self._F[0, 5] = float(dt * ((a * dt) / 2 + v) * cp * cy)
        self._F[0, 6] = float(-dt * ((a * dt) / 2 + v) * sp * sy)
        self._F[1, 3] = float(dt * cp * cy)
        self._F[1, 4] = float((dt**2 * cp * cy) / 2)
        self._F[1, 5] = float(-dt * ((a * dt) / 2 + v) * sy * cp)
        self._F[1, 6] = float(-dt * ((a * dt) / 2 + v) * sp * cy)
        self._F[2, 3] = float(dt * sp)
        self._F[2, 4] = float((dt**2 * sp) / 2)
        self._F[2, 6] = float(dt * ((a * dt) / 2 + v) * cp)
        self._F[3, 4] = dt
        self._F[5, 7] = dt
        self._F[6, 8] = dt
