import numpy as np
from scipy import linalg
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.metrics.pairwise import pairwise_kernels


def S(p, A, gamma=0):
    k, d = A.shape
    ret = np.zeros((d, d))
    for i in range(k):
        ret += p[i] * A[[i], :].T @ A[[i], :]
    return ret + gamma * np.eye(d)


# Faster matrix inverse for positive semi-definite matrix using Cholesky decomposition
def s_inv(V):
    L = np.linalg.inv(np.linalg.cholesky(V))
    return L.T @ L


def optimal_design(A, p0=None):
    k, d = A.shape
    p = p0
    if p is None:
        p = np.ones(k) / k # initialize with uniform distribution
    for _ in range(200):
        V = S(p, A)
        V_inv = s_inv(V)
        vs = [None] * k
        for i in range(k):
            a = A[[i], :]
            vs[i] = (a @ V_inv @ a.T).item()
        i_star = np.argmax(vs)
        v_star = vs[i_star]
        gamma = (v_star / d - 1) / (v_star - 1)
        p = (1 - gamma) * p
        p[i_star] += gamma

    V = S(p, A)
    V_inv = s_inv(V)
    for a in range(k):
        arm = A[[a], :]
        assert arm @ V_inv @ arm.T < 3 * d / 2
    return p


def glm_fit(X, y, link='logistic'):
    # TODO: better link function management
    assert link in ['logistic']
    mean_function = lambda x: 1 / (1 + np.exp(-x))
    var_function = lambda x: np.exp(-x) / np.power(1 + np.exp(-x), 2)

    n, d = X.shape

    theta = np.zeros(d)
    S = np.sum(X * y.reshape((n, 1)), axis=0)
    for _ in range(30):
        scores = X.dot(theta)
        g = S - np.sum(X * mean_function(scores).reshape((n, 1)), axis=0)
        if np.allclose(g, np.zeros(d)):
            break
        # TODO: optimize hessian computation
        H = np.zeros((d, d))
        variances = var_function(scores)
        for i in np.arange(n):
            x = X[i, :]
            H -= variances[i] * np.outer(x, x)
        H_inv = np.linalg.pinv(H)
        theta = theta - H_inv @ g
    return theta


def G(x, A, gamma):
    k, d = A.shape
    return (
        S(x, A) + gamma * np.eye(d)
    )


def phi(c, t, x, A, gamma):
    g = G(x, A, gamma)
    _, logdet = np.linalg.slogdet(g)
    return (
        t * (np.sum(c * x) - logdet)
        - np.sum(np.log(x))
        - np.log(1 - np.sum(x))
    )


def J(c, x, A, gamma):
    g = G(x, A, gamma)
    _, logdet = np.linalg.slogdet(g)
    return (
        np.sum(c * x) - logdet
    )


def newton_direction(c, t, x, A, gamma):
    n, d = A.shape
    V = S(x, A) + gamma * np.eye(d)
    V_inv = np.linalg.inv(V)
    vs = np.zeros(n)
    for i in range(n):
        a = A[i, :]
        vs[i] = V_inv.dot(a).dot(a)
    d1 = (
        t * (c - vs)
        - 1 / x
        + 1 / (1 - np.sum(x))
    )
    d2 = (
        t * np.power(A @ V_inv @ A.T, 2)
        + 1 / ((1 - np.sum(x)) ** 2)
        + np.diag(1 / np.power(x, 2))
    )
    direction = -np.linalg.inv(d2) @ d1
    v = direction.reshape((n, 1))
    l = np.sqrt(v.T @ d2 @ v).item()
    return direction, l


def generalized_eigenvalues(x, direction, A, gamma):
    k, d = A.shape
    left = S(direction, A)
    right = S(x, A) + gamma * np.eye(d)
    eigen1 = np.real(linalg.eigvals(left, right))
    eigen2 = np.append(direction / x, -np.sum(direction) / (1 - np.sum(x)))
    return (eigen1, eigen2)


def line_search(c, t, x, direction, A, gamma):
    eigen1, eigen2 = generalized_eigenvalues(x, direction, A, gamma)
    h = 0
    for _ in range(30):
        d1 = (
            t * np.sum(c * direction)
            - np.sum(t * eigen1 / (1 + h * eigen1))
            - np.sum(eigen2 / (1 + h * eigen2))
        )
        d2 = (
            np.sum(t / ((1 / eigen1 + h) ** 2))
            + np.sum(1 / ((1 / eigen2 + h) ** 2))
        )
        increment = -d1 / d2
        if np.any(1 + (h + increment) * eigen1 <= 1e-8) or np.any(1 + (h + increment) * eigen2 <= 1e-8):
            min_eigenvalue = min(np.min(eigen1), np.min(eigen2))
            h = (h - 1 / min_eigenvalue) / 2
        else:
            h = h - d1 / d2
    return h


def newton_optimize(c, t, A, gamma, x=None):
    n, d = A.shape
    if x is None:
        x = np.ones(n) / (2 * n)
    l = 1
    J = np.Inf
    counter = 0

    save_x = x.copy()
    # TODO: check the while condition
    #while l > max(1e-5, 1e-4 / gamma * t):
    while l > 1e-6 / gamma * t:
        direction, l = newton_direction(c, t, x, A, gamma)
        h = 1
        if l > 0.5:
            h = line_search(c, t, x, direction, A, gamma)
        x = x + h * direction
        new_J = phi(c, t, x, A, gamma)
        if not (new_J < J or np.allclose(new_J, J, atol=1e-03)):
            print(new_J, J)
            raise
        J = new_J
        counter += 1
        if counter > 100:
            #print(c, t, A, gamma, save_x)
            #print(x)
            print(t, h, l)
        if counter > 200:
            raise
    return x


def central_path(c, A, gamma):
    n, d = A.shape
    x = np.ones(n) / (2 * n)
    t = 1
    while t < 1e7 or (t < 1e10 and np.sum(x) < 0.8):
        x = newton_optimize(c, t, A, gamma, x=x)
        t = t * 1.1
    return x


def OP(A, delta, beta, gamma):
    n, d = A.shape
    c = delta * beta / 2
    p = central_path(c, A, gamma)

    # sanity check: `p` is nearly a probability measure
    assert np.abs(1 - np.sum(p)) < 0.8

    # sanity check
    s = S(p, A) + gamma * np.eye(d)
    s_inv = np.linalg.inv(s)
    k, d = A.shape
    J = np.sum(p * delta)
    assert J < 2 * d  / beta
    for a in range(k):
        arm = A[[a], :]
        norm = arm @ s_inv @ arm.T
        assert norm < beta * delta[a] + 2 * n or np.allclose(norm, beta * delta[a] + 2 * n)

    return p / np.sum(p)


def kernel_matrix(A, config):
    name = config['name']
    C = config.copy()
    del C['name']
    if name == 'matern':
        kernel = Matern(**C)
        return pairwise_kernels(A, metric=kernel)
    elif name == 'rbf':
        kernel = RBF(**C)
        return pairwise_kernels(A, metric=kernel)
    raise Exception(f'Invalid kernel: {name}')


class InformationGain:
    def __init__(self, Phi, T, gamma):
        self.Phi = Phi
        self.T = T
        self.gamma = gamma
        self.cache = {}


    def get_or_compute(self, t):
        if t in self.cache:
            return self.cache[t]
        n, d = self.Phi.shape
        delta = np.zeros(n)
        p = OP(self.Phi, delta, 1, self.gamma / t)
        s = S(p, self.Phi) * t / self.gamma + np.eye(d)
        _, logdet = np.linalg.slogdet(s)
        information_gain = logdet / 2
        self.cache[t] = information_gain
        return information_gain

    def get(self, t):
        if t in self.cache:
            return self.cache[t]
        t_left = 2 ** int(np.log2(t))
        dimension_left = self.get_or_compute(t_left)
        if t_left == t:
            return dimension_left
        t_right = min(self.T, t_left * 2)
        dimension_right = self.get_or_compute(t_right)

        # interpolate
        p = (t - t_left) / (t_right - t_left)
        return dimension_left * (1 - p) + dimension_right * p

    def get_exact(self, t):
        if t in self.cache:
            return self.cache[t]
        return self.get_or_compute(t)

if __name__ == '__main__':
    A = np.array([
        [1, 2, 3],
        [1, 0, 3],
        [0, 1, 2],
    ])
    T = 16
    info_gain = InformationGain(A, 16, 1)
    for i in range(T):
        print(i + 1, info_gain.get(i + 1))
