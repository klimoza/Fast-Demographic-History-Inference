import numpy
import sys
from dadi import Integration, Numerics, tridiag, Spectrum, PhiManip

# sys.path.append('demographic_inference_data/1_AraTha_4_Hub')
# import demographic_model_dadi


def tridiag_inverse(A):
    n = A.shape[0]
    theta = numpy.zeros(n + 1)
    theta[0] = 1
    theta[1] = A[0][0]
    for i in range(2, n + 1):
        theta[i] = A[i - 1][i - 1] * theta[i - 1] - A[i - 2][i - 1] * A[i - 1][i - 2] * theta[i - 2]

    phi = numpy.zeros(n + 1)
    phi[n] = 1
    phi[n - 1] = A[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        phi[i] = A[i][i] * phi[i + 1] - A[i][i + 1] * A[i + 1][i] * phi[i + 2]

    B = numpy.zeros(A.shape)
    for i in range(0, n):
        B[i][i] = theta[i] * phi[i + 1] / theta[n]

        cur_sign = 1
        cur_prod = 1
        for j in range(i + 1, n):
            cur_sign *= -1
            cur_prod *= A[j - 1][j]
            B[i][j] = cur_sign * cur_prod * theta[i] * phi[j + 1] / theta[n]

        cur_sign = 1
        cur_prod = 1
        for j in range(i - 1, -1, -1):
            cur_sign *= -1
            cur_prod *= A[j + 1][j]
            B[i][j] = cur_sign * cur_prod * theta[j] * phi[i + 1] / theta[n]

    return B


def make_matrix(a, b, c):
    n = a.size
    A = numpy.zeros((n, n))
    for i in range(n):
        if i:
            A[i][i - 1] = a[i]
        A[i][i] = b[i]
        if i + 1 < a.size:
            A[i][i + 1] = a[i]
    return A


def inverse(a, b, c):
    return tridiag_inverse(make_matrix(a, b, c))


def find_gradient_one_pop(phi, xx, dphi, T, nu=1, gamma=0, h=0.5, theta0=1, initial_t=0, beta=1):
    M = Integration._Mfunc1D(xx, gamma, h)
    MInt = Integration._Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
    V = Integration._Vfunc(xx, nu, beta=beta)
    VInt = Integration._Vfunc((xx[:-1] + xx[1:]) / 2, nu, beta=beta)

    dx = numpy.diff(xx)
    dfactor = Integration._compute_dfactor(dx)
    delj = Integration._compute_delj(dx, MInt, VInt)

    n = phi.size
    dA = numpy.zeros((n, n, 3))

    a = numpy.zeros(n)
    a[1:] += dfactor[1:] * (-MInt * delj - V[:-1] / (2 * dx))
    for i in range(1, n):
        dA[i][i - 1][0] += dfactor[i] * xx[i - 1] * (1 - xx[i - 1]) / (2 * dx[i - 1]) / (nu ** 2)
        dA[i][i - 1][1] += -dfactor[i] * (1 / 2) * ((xx[i - 1] + xx[i]) / 2) * (1 - (xx[i - 1] + xx[i]) / 2)

    c = numpy.zeros(n)
    c[:-1] += -dfactor[:-1] * (-MInt * (1 - delj) + V[1:] / (2 * dx))
    for i in range(0, n - 1):
        dA[i][i + 1][0] += dfactor[i] * xx[i + 1] * (1 - xx[i + 1]) / (2 * dx[i]) / (nu ** 2)
        dA[i][i + 1][1] += dfactor[i] * (1 / 2) * ((xx[i] + xx[i + 1]) / 2) * (1 - (xx[i] + xx[i + 1]) / 2)

    b = numpy.zeros(n)
    b[:-1] += -dfactor[:-1] * (-MInt * delj - V[:-1] / (2 * dx))
    for i in range(0, n - 1):
        dA[i][i][0] += -dfactor[i] * xx[i] * (1 - xx[i]) / (2 * dx[i]) / (nu ** 2)
        dA[i][i][1] += dfactor[i] * (1 / 2) * ((xx[i] + xx[i + 1]) / 2) * (1 - (xx[i] + xx[i + 1]) / 2)

    b[1:] += dfactor[1:] * (-MInt * (1 - delj) + V[1:] / (2 * dx))
    for i in range(1, n):
        dA[i][i][0] += -dfactor[i] * xx[i] * (1 - xx[i]) / (2 * dx[i - 1]) / (nu ** 2)
        dA[i][i][1] += -dfactor[i] * (1 / 2) * ((xx[i - 1] + xx[i]) / 2) * (1 - (xx[i - 1] + xx[i]) / 2)

    if M[0] <= 0:
        b[0] += (0.5 / nu - M[0]) * 2 / dx[0]
        dA[0][0][0] -= dfactor[0] / (2 * nu * nu)
        dA[0][0][1] -= dfactor[0] * xx[0] * (1 - xx[0])

    if M[-1] >= 0:
        b[-1] += -(-0.5 / nu - M[-1]) * 2 / dx[-1]
        dA[-1][-1][0] -= dfactor[-1] / (2 * nu * nu)
        dA[-1][-1][1] += dfactor[-1] * xx[-1] * (1 - xx[-1])

    A = make_matrix(a, b, c)
    dt = Integration._compute_dt(dx, nu, [0], gamma, h)
    current_t = initial_t
    while current_t < T:
        this_dt = min(dt, T - current_t)

        inv_A = inverse(this_dt * a, 1 + this_dt * b, this_dt * c)
        if current_t + this_dt != T:
            dphi = inv_A * dphi
        else:
            phi_tmp = dphi
            Integration._inject_mutations_1D(phi_tmp[:][-1], 1, xx, theta0)
            dphi = inv_A * (dphi + phi_tmp)

        Integration._inject_mutations_1D(phi, this_dt, xx, theta0)

        if current_t + this_dt != T:
            dinv = - dt * inv_A * dA * inv_A
            dphi += dinv * phi
        else:
            dA = dt * dA
            dA[:][:][-1] = A
            dinv = - inv_A * dA * inv_A
            dphi += dinv * phi

        r = phi / this_dt
        phi = tridiag.tridiag(a, b + 1 / this_dt, c, r)
        current_t += this_dt
    return dphi


def findCellGradient(xx, i):
    n = xx.size()
    res = 0
    for k in range(n - 1):
        res += (xx[k + 1] - xx[k]) * (
                (xx[k] ** i) * (1 - xx[k]) ** (n - i) + (xx[k + 1] ** i) * (1 - xx[k + 1]) ** (n - i))
    return Numerics.multinomln([n, i]) * res


def findLikelihoodGradient(phi, dphi, xx, expctd, res):
    n = xx.size()
    dM = numpy.zeros(n)

    for i in range(n):
        dM[i] = findCellGradient(xx, i) * (expctd[i] / res[i] - 1)

    return dM.dot(dphi)


eps = 1e-8


def isId(A):
    n = A.shape[0]
    for i in range(0, n):
        for j in range(0, n):
            if i == j and abs(A[i][j] - 1) >= eps:
                return False
            if i != j and abs(A[i][j]) >= eps:
                return False
    return True


def testInverse():
    A = numpy.array([[1, 4, 0, 0],
                     [3, 4, 1, 0],
                     [0, 2, 3, 4],
                     [0, 0, 1, 3]])
    Ak = tridiag_inverse(A)
    assert (isId(A.dot(Ak)))

    B = numpy.array([[4, 3],
                     [3, 2]])
    Bk = tridiag_inverse(B)
    assert (isId(B.dot(Bk)))

    C = numpy.array([[3, 4],
                     [1, 2]])
    Ck = tridiag_inverse(C)
    assert (isId(C.dot(Ck)))

    D = numpy.array([[-1, 8, 0, 0],
                     [2, -2, -7, 0],
                     [0, 7, 3, -6],
                     [0, 0, 8, -7]])
    Dk = tridiag_inverse(D)
    assert (isId(D.dot(Dk)))

    E = numpy.array([[2, 3, 0, 0],
                     [6, 3, 9, 0],
                     [0, 2, 5, 2],
                     [0, 0, 3, 3]])
    Ek = tridiag_inverse(E)
    assert (isId(E.dot(Ek)))


# 1_AraTha_4_Hub/demographic_model_dadi.py
def model_func(params, expctd, ns, pts):
    """
    Three epoch model from Huber et al., 2018.
    First epoch is ancestral.

    :param N1: Size of population during second epoch.
    :param T1: Time of second epoch.
    :param N2: Size of population during third epoch.
    :param T2: Time of third epoch.
    """
    N1, T1, N2, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    dphi = find_gradient_one_pop(phi, xx, numpy.zeros((xx.size, 3)), T1, nu=N1)
    phi = Integration.one_pop(phi, xx, T1, nu=N1)

    dphi = find_gradient_one_pop(phi, xx, dphi, T2, nu=N2)
    phi = Integration.one_pop(phi, xx, T2, nu=N2)

    fs = Spectrum.from_phi(phi, ns, [xx])
    return fs, findLikelihoodGradient(phi, dphi, xx, expctd, fs)

def testGradient_AraTha(N1, T1, N2, T2):
    expctd = Spectrum.from_file("demographic_inference_data/1_AraTha_4_Hub/fs_data.fs")
    res, gradient = model_func((N1, T1, N2, T2), expctd, 16, [40, 50, 60])

if __name__ == "__main__":
    testInverse()
