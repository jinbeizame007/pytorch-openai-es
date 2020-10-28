import numpy as np


# https://github.com/openai/evolution-strategies-starter/blob/951f19986921135739633fb23e55b2075f66c2e6/es_distributed/es.py#L25

class RunningStat(object):
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def set_from_init(self, init_mean, init_std, init_count):
        self.sum[:] = init_mean * init_count
        self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
        self.count = init_count


def compute_centered_ranks(x):
    ranks = np.empty(x.shape[0], dtype=np.float32)
    ranks[x.argsort()] = np.linspace(-0.5, 0.5, x.shape[0], dtype=np.float32)
    return ranks


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


# https://github.com/openai/evolution-strategies-starter/blob/08b9aeec3691fcf768eed3d9b5f2245236ba58f9/es_distributed/optimizers.py

class Optimizer(object):
    def __init__(self, num_params):
        self.dim = num_params
        self.t = 0

    def update(self, pi, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = pi.get_params()
        pi.set_params(theta + step)
        return pi

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, pi, stepsize, momentum=0.9):
        Optimizer.__init__(self, pi)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    #def __init__(self, pi, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
    def __init__(self, num_params, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        #Optimizer.__init__(self, pi)
        Optimizer.__init__(self, num_params)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
