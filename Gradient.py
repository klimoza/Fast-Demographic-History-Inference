import numpy
import sys
from dadi import Integration, Numerics, tridiag, Spectrum, PhiManip, Inference


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
        if i + 1 < n:
            A[i][i + 1] = c[i]
    return A


def inverse(a, b, c):
    return tridiag_inverse(make_matrix(a, b, c))


def tens_mult(A, B):
    n1 = A.shape[0]
    n2 = B.shape[0]
    n3 = B.shape[1]
    C = numpy.zeros((n1, n3, 3))
    for i in range(0, n1):
        for j in range(0, n2):
            for k in range(0, n3):
                C[i][k] += A[i][j] * B[j][k]
    return C


def tens_vec_mult(A, b):
    n1 = A.shape[0]
    n2 = b.shape[0]
    C = numpy.zeros((n1, 3))
    for i in range(0, n1):
        for j in range(0, n2):
            C[i] += A[i][j] * b[j]
    return C


def give_me_everything(phi, xx, T, nu=1, gamma=0, h=0.5, theta0=1, initial_t=0, beta=1):
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
        dA[i][i + 1][0] += dfactor[i] * xx[i + 1] * (1 - xx[i + 1]) / (2 * dx[i] * nu * nu)
        dA[i][i + 1][1] += dfactor[i] * (1 / 2) * ((xx[i] + xx[i + 1]) / 2) * (1 - (xx[i] + xx[i + 1]) / 2)

    b = numpy.zeros(n)
    b[:-1] += -dfactor[:-1] * (-MInt * delj - V[:-1] / (2 * dx))
    for i in range(0, n - 1):
        dA[i][i][0] += -dfactor[i] * xx[i] * (1 - xx[i]) / (2 * dx[i]) / (nu ** 2)
        dA[i][i][1] += dfactor[i] * (1 / 2) * ((xx[i] + xx[i + 1]) / 2) * (1 - (xx[i] + xx[i + 1]) / 2)

    b[1:] += dfactor[1:] * (-MInt * (1 - delj) + V[1:] / (2 * dx))
    for i in range(1, n):
        dA[i][i][0] += -dfactor[i] * xx[i] * (1 - xx[i]) / (2 * dx[i - 1] * nu * nu)
        dA[i][i][1] += -dfactor[i] * (1 / 2) * ((xx[i - 1] + xx[i]) / 2) * (1 - (xx[i - 1] + xx[i]) / 2)

    if M[0] <= 0:
        b[0] += (0.5 / nu - M[0]) * 2 / dx[0]
        dA[0][0][0] += -1 / (nu * nu * dx[0])
        dA[0][0][1] += -2 * xx[0] * (1 - xx[0]) / (dx[0])

    if M[-1] >= 0:
        b[-1] += -(-0.5 / nu - M[-1]) * 2 / dx[-1]
        dA[-1][-1][0] += -1 / (nu * nu * dx[-1])
        dA[-1][-1][1] += 2 * xx[-1] * (1 - xx[-1]) / dx[-1]

    A = make_matrix(a, b, c)
    dt = Integration._compute_dt(dx, nu, [0], gamma, h)
    # dt = 1
    return a, b, c, A, dA, dt


def find_gradient_one_pop(phi, xx, dphi, T, nu=1, gamma=0, h=0.5, theta0=1, initial_t=0, beta=1):
    a, b, c, A, dA, dt = give_me_everything(phi, xx, T, nu, gamma, h, theta0, initial_t, beta)
    current_t = initial_t
    inv_A = inverse(dt * a, 1 + dt * b, dt * c)
    while current_t < T:
        this_dt = min(dt, T - current_t)

        if current_t + this_dt != T:
            dphi = inv_A.dot(dphi)
        else:
            inv_A = inverse(this_dt * a, 1 + this_dt * b, this_dt * c)
            dphi[1][-1] += 1 / xx[1] * theta0 / 2 * 2 / (xx[2] - xx[0])
            # phi_tmp = dphi
            # phi_tmp[1][-1] += 1 / xx[1] * theta0 / 2 * 2 / (xx[2] - xx[0])
            # phi_tmp[:, -1] = Integration._inject_mutations_1D(phi_tmp[:, -1], 1, xx, theta0)
            dphi = inv_A.dot(dphi)

        phi[1] += this_dt / xx[1] * theta0 / 2 * 2 / (xx[2] - xx[0])
        # phi = Integration._inject_mutations_1D(phi, this_dt, xx, theta0)

        if current_t + this_dt != T:
            dinv = - this_dt * tens_mult(tens_mult(inv_A, dA), inv_A)
            dphi += tens_vec_mult(dinv, phi)
        else:
            dA = this_dt * dA
            dA[:, :, -1] = A
            dinv = -tens_mult(tens_mult(inv_A, dA), inv_A)
            dphi += tens_vec_mult(dinv, phi)

        r = phi / this_dt
        phi = tridiag.tridiag(a, b + 1 / this_dt, c, r)
        current_t += this_dt
    return dphi, phi


def findCellGradient(xx, i):
    n = len(xx)
    res = 0
    for k in range(n - 1):
        res += (xx[k + 1] - xx[k]) * (
                (xx[k] ** i) * (1 - xx[k]) ** (n - i - 1) + (xx[k + 1] ** i) * (1 - xx[k + 1]) ** (n - i - 1))
    return Numerics.comb(n, i) * res


def findLikelihoodGradient(phi, dphi, xx, S, M):
    n = len(M)
    dM = numpy.zeros(n)
    A = 0
    B = 0
    dB = 0

    for i in range(n):
        dM[i] = findCellGradient(xx, i)
        dB += dM[i]
        B += M.data[i]
        A += S.data[i]
    # print(dM)
    dH = numpy.zeros(len(xx))
    for i in range(n):
        # dH += -A * (B * dM[i] - M.data[i] * dB) / (B ** 2) + S.data[i] * dM[i] / M.data[i] - S.data[i] * B
        dH += dM[i] * (S[i] / M[i] - 1)

    # print("A: ", A)
    # print("dH: ", dH)
    return dH.dot(dphi)


eps = 1e-10


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
def model_func(params, ns, pts):
    N1, T1, G = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    # phi = Integration.one_pop(phi, xx, T1, nu=N1, gamma=G)
    dphi, phi = find_gradient_one_pop(phi, xx, numpy.zeros((xx.size, 3)), T1, nu=N1, gamma=G)
    # phi = Integration.one_pop(phi, xx, T2, nu=N2)
    # print(phi)
    fs = Spectrum.from_phi(phi, [ns], [xx], force_direct=True)
    return phi, fs


# 1_AraTha_4_Hub/demographic_model_dadi.py
def model_grad(params, S, ns, pts):
    N1, T1, G = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    dphi, phi = find_gradient_one_pop(phi, xx, numpy.zeros((xx.size, 3)), T1, nu=N1, gamma=G)
    # phi = Integration.one_pop(phi, xx, T1, nu=N1, gamma=G)

    # dphi = find_gradient_one_pop(phi, xx, dphi, T2, nu=N2)
    # phi = Integration.one_pop(phi, xx, T2, nu=N2)
    # print("dphi: ", dphi)
    # print("kek: ", phi)
    fs = Spectrum.from_phi(phi, [ns], [xx], force_direct=True)
    return phi, fs, findLikelihoodGradient(phi, dphi, xx, S, fs)


def testMatrixDerivative(params, pts):
    N, T, G = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    a, b, c, A, dA, dt = give_me_everything(phi, xx, T, nu=N, gamma=G)
    this_dt = T - T % dt
    dA *= this_dt
    dA[:, :, -1] = A
    A = make_matrix(a * this_dt, 1 + b * this_dt, c * this_dt)
    a, b, c, _, _, _ = give_me_everything(phi, xx, T + eps, N, G)
    tmp_dt = (T + eps - T % dt)
    Az = make_matrix(a * tmp_dt, 1 + b * tmp_dt, c * tmp_dt)
    a, b, c, _, _, _ = give_me_everything(phi, xx, T, N + eps, G)
    Ax = make_matrix(a * this_dt, 1 + b * this_dt, c * this_dt)
    a, b, c, _, _, _ = give_me_everything(phi, xx, T, N, G + eps)
    Ay = make_matrix(a * this_dt, 1 + b * this_dt, c * this_dt)

    gr = numpy.zeros((len(xx), len(xx), 3))
    for i in range(0, len(xx)):
        for j in range(0, len(xx)):
            gr[i][j][0] = (Ax[i, j] - A[i, j]) / eps
            gr[i][j][1] = (Ay[i, j] - A[i, j]) / eps
            gr[i][j][2] = (Az[i, j] - A[i, j]) / eps
    print(dA - gr)
    # print(gr)


def testGradient_AraTha(N1, T1):
    S = Spectrum.from_file("data/1_AraTha_4_Hub/fs_data.fs")
    phi, M, gradient = model_grad((N1, T1, 0), S, 16, 40)
    ll = Inference.ll(M, S)
    # print(S.data)
    # print(M.data)
    print(gradient)
    print(M.data)
    phix, Mx = model_func((N1 + eps, T1, 0), 16, 40)
    print(Mx.data)
    phiy, My = model_func((N1, T1, eps), 16, 40)
    print(My.data)
    phiz, Mz = model_func((N1, T1 + eps, 0), 16, 40)
    print(Mz.data)
    llx = Inference.ll(Mx, S)
    lly = Inference.ll(My, S)
    llz = Inference.ll(Mz, S)
    # print("d_gr: ", gr)
    print(ll, llx, lly, llz)
    grad = [(llx - ll) / eps, (lly - ll) / eps, (llz - ll) / eps]
    print(grad)


if __name__ == "__main__":
    testInverse()
    # testMatrixDerivative((0.149, 0.023, 0), 40)
    testGradient_AraTha(0.149, 0.023)
