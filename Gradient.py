import math

import numpy
from dadi import Integration, Numerics, tridiag, Spectrum, PhiManip, Inference


def tridiag_inverse(A):
    """
    Функция, обращающая невырожденную трехдиагональную матрицу за O(n^2).

    https://en.wikipedia.org/wiki/Tridiagonal_matrix#Inversion
    """
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
    """
    Функция, составляющая трехдиагональную матрицу, по данным коэффициентам:

    b c 0 . 0 0
    a b c . 0 0
    0 a b . 0 0
    . . . . . .
    0 0 0 . a b
    """
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
    """
    Функция, перемножающая тензоры размерностей N1 x N2 x 1 на N2 x N3 x M следующим образом:
    N1 x N2 и N2 x N3 перемножаются как обычные матрицы,
    а в клетке матрицы лежит вектор размера M, домноженный на константу.
    """
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
    """
    Аналогично tens_mult, только N3 = 1.
    """
    n1 = A.shape[0]
    n2 = b.shape[0]
    C = numpy.zeros((n1, 3))
    for i in range(0, n1):
        for j in range(0, n2):
            C[i] += A[i][j] * b[j]
    return C


# For debug purposes
eps = 1e-8
ns = 16
pts = 40
filename = "data/1_AraTha_4_Hub/fs_data.fs"


def give_me_everything(xx, nu=1, gamma=0, h=0.5, beta=1):
    """
    Функция, возвращающая:
    - коэффициенты a, b, c нашего трехдиагонального уравнения без коэффициентов Delta_t и единицы в b
    - матрицу A нашего трехдиагонального уравнения
    - производную нашей матрицы dA
    - шаг времени dt
    !! Помним, что для удобства мы не используем нигде время, т.к. его удобнее добавлять в переходах !!
    """
    M = Integration._Mfunc1D(xx, gamma, h)
    MInt = Integration._Mfunc1D((xx[:-1] + xx[1:]) / 2, gamma, h)
    V = Integration._Vfunc(xx, nu, beta=beta)
    VInt = Integration._Vfunc((xx[:-1] + xx[1:]) / 2, nu, beta=beta)

    dx = numpy.diff(xx)
    dfactor = Integration._compute_dfactor(dx)
    delj = Integration._compute_delj(dx, MInt, VInt)

    dA = numpy.zeros((pts, pts, 3))

    a = numpy.zeros(pts)
    a[1:] += dfactor[1:] * (-MInt * delj - V[:-1] / (2 * dx))
    for i in range(1, pts):
        dA[i][i - 1][0] += dfactor[i] * xx[i - 1] * (1 - xx[i - 1]) / (2 * dx[i - 1]) / (nu ** 2)
        dA[i][i - 1][1] += -dfactor[i] * (1 / 2) * ((xx[i - 1] + xx[i]) / 2) * (1 - (xx[i - 1] + xx[i]) / 2)

    c = numpy.zeros(pts)
    c[:-1] += -dfactor[:-1] * (-MInt * (1 - delj) + V[1:] / (2 * dx))
    for i in range(0, pts - 1):
        dA[i][i + 1][0] += dfactor[i] * xx[i + 1] * (1 - xx[i + 1]) / (2 * dx[i] * nu * nu)
        dA[i][i + 1][1] += dfactor[i] * (1 / 2) * ((xx[i] + xx[i + 1]) / 2) * (1 - (xx[i] + xx[i + 1]) / 2)

    b = numpy.zeros(pts)
    b[:-1] += -dfactor[:-1] * (-MInt * delj - V[:-1] / (2 * dx))
    for i in range(0, pts - 1):
        dA[i][i][0] += -dfactor[i] * xx[i] * (1 - xx[i]) / (2 * dx[i]) / (nu ** 2)
        dA[i][i][1] += dfactor[i] * (1 / 2) * ((xx[i] + xx[i + 1]) / 2) * (1 - (xx[i] + xx[i + 1]) / 2)

    b[1:] += dfactor[1:] * (-MInt * (1 - delj) + V[1:] / (2 * dx))
    for i in range(1, pts):
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
    return a, b, c, A, dA, dt


def find_gradient_one_pop(phi, xx, dphi, T, nu=1, gamma=0, h=0.5, theta0=1, initial_t=0, beta=1):
    """
    Функция, интегрирующая одну эпоху.
    Параллельно с phi, считает еще и производную phi по параметрам (nu, gamma, T).
    """
    a, b, c, A, dA, dt = give_me_everything(xx, nu, gamma, h, beta)

    current_t = initial_t
    inv_A = inverse(dt * a, 1 + dt * b, dt * c)
    while current_t < T:
        this_dt = min(dt, T - current_t)

        if current_t + this_dt != T:
            dphi = inv_A.dot(dphi)
        else:
            inv_A = inverse(this_dt * a, 1 + this_dt * b, this_dt * c)

            dphi[1][-1] += 1 / xx[1] * theta0 / 2 * 2 / (xx[2] - xx[0])
            dphi = inv_A.dot(dphi)

        phi = Integration._inject_mutations_1D(phi, this_dt, xx, theta0)

        if current_t + this_dt != T:
            dinv = - tens_mult(tens_mult(inv_A, this_dt * dA), inv_A)
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


def comb_fun(i, x):
    """
    Производная функции, по которой мы интегрируем в ячейке нашего АЧС.
    Сама функция -- это выражение ниже, домноженное на phi[x]
    """
    return Numerics.comb(ns, i) * x**i * (1 - x)**(ns - i)


def findCellGradient(xx):
    """
    Ищем производную dM/dPhi => размерность должна быть N x G

    Так как там сумма по phi на сетке, то в j-ой координате должен остаться коэффициент при это phi-шке.
    Коэффициенты берем из метода трапеций, относительно numpy.trapz погрешность около 1e-12.
    """
    res = numpy.zeros((ns + 1, pts))
    for i in range(ns + 1):
        res[i][0] = comb_fun(i, xx[0]) * (xx[1] - xx[0]) / 2
        for j in range(1, len(xx) - 1):
            res[i][j] = comb_fun(i, xx[j]) * (xx[j + 1] - xx[j - 1]) / 2
        res[i][-1] = comb_fun(i, xx[-1]) * (xx[-1] - xx[-2]) / 2
    return res


def findLLdPhi(xx, M, S, multinom=True):
    """
    Ищем производную d log H / d Phi => размерность должна быть 1 x G
    M := model
    S := data
    """
    M, S = Numerics.intersect_masks(M, S)
    dM = findCellGradient(xx)
    A = S.sum()
    B = M.sum()
    dB = dM.sum(axis=0)

    if multinom:
        f = numpy.array([(-A * (B * dM[:, i] - M * dB[i]) / (B**2) + S * dM[:, i] / M - S * dB[i] / B).sum() for i in range(pts)])
    else:
        f = numpy.array([(dM[:, i] * (S / M - 1)).sum() for i in range(pts)])

    dH = Spectrum(f)
    return dH


def findLikelihoodGradient(dphi, xx, M, S):
    """
    Считаем d log H / d Theta => размерность должна быть 1 x M
    """
    dH = findLLdPhi(xx, M, S, True)
    return dH.dot(dphi)


def isId(A):
    """
    Check if matrix is id
    """
    n = A.shape[0]
    for i in range(0, n):
        for j in range(0, n):
            if i == j and abs(A[i][j] - 1) >= eps:
                return False
            if i != j and abs(A[i][j]) >= eps:
                return False
    return True


def testInverse():
    """
    Тесты для проверки функции tridiag_inverse().
    """
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
def model_grad(params, S, ns, pts, grad=0):
    """
    Моделируем 1_AraTha_4_Hub, отбросив последнюю эпоху.
    Помимо полученного АЧС возвращаем еще и phi, так как удобно для отладки.
    Сразу возвращаем и градиент, потому что удобно.
    !! force_direct=True, так как это простой случай и заставить бы хотя бы его работать !!
    """
    N1, T1, G1 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    dphi = numpy.zeros((len(xx), 3))
    if grad:
        dphi, phi = find_gradient_one_pop(phi, xx, numpy.zeros((len(xx), 3)), T1, nu=N1, gamma=G1)
    else:
        phi = Integration.one_pop(phi, xx, T1, N1, G1)

    fs = Spectrum.from_phi(phi, [ns], [xx], force_direct=True)

    if grad == 1:
        return phi, fs, findLikelihoodGradient(dphi, xx, fs, S)
    elif grad == 2:
        return phi, fs, findLikelihoodGradient(dphi, xx, fs, S), dphi
    else:
        return phi, fs


def testPhiCorrect(params):
    """
    Функция проверяет правильно ли считается phi.
    """
    S = Spectrum.from_file(filename)
    myPhi, _, _ = model_grad(params, S, ns, pts, 1)
    phi, _ = model_grad(params, S, ns, pts)
    print(myPhi - phi)


def testMatrixDerivative(params):
    """
    Функция проверяет правильно ли мы посчитали производную трехдиагональной матрицы.
    Для полноценной работы нужно еще что-то дописать, но основная структура написана.
    """
    N, T, G = params

    xx = Numerics.default_grid(pts)

    a, b, c, A, dA, dt = give_me_everything(xx, N, G)
    k = math.floor(T / dt)
    if k * dt == T:
        this_dt = dt
    else:
        this_dt = T - dt
    dA *= this_dt
    dA[:, :, -1] = A
    A = make_matrix(a * this_dt, 1 + b * this_dt, c * this_dt)

    a, b, c, _, _, _ = give_me_everything(xx, N + eps, G)
    Ax = make_matrix(a * this_dt, 1 + b * this_dt, c * this_dt)

    a, b, c, _, _, _ = give_me_everything(xx, N, G + eps)
    Ay = make_matrix(a * this_dt, 1 + b * this_dt, c * this_dt)

    a, b, c, _, _, _ = give_me_everything(xx, N, G)
    k = math.floor((T + eps) / dt)
    if k * dt == T + eps:
        tmp_dt = dt
    else:
        tmp_dt = T + eps - dt
    Az = make_matrix(a * tmp_dt, 1 + b * tmp_dt, c * tmp_dt)

    gr = numpy.zeros((len(xx), len(xx), 3))
    for i in range(len(xx)):
        for j in range(len(xx)):
            gr[i][j][0] = (Ax[i, j] - A[i, j]) / eps
            gr[i][j][1] = (Ay[i, j] - A[i, j]) / eps
            gr[i][j][2] = (Az[i, j] - A[i, j]) / eps

    print(dA - gr)


def testInverseMatrixDerivative(params):
    """
    Функция проверяет правильно ли мы посчитали производную обратной трехдиагональной матрицы.
    Формула: dA^{-1} = - A^{-1} * dA * A^{-1}
    """
    N, T, G = params

    xx = Numerics.default_grid(pts)

    a, b, c, A, dA, dt = give_me_everything(xx, N, G)
    k = math.floor(T / dt)
    if k * dt == T:
        this_dt = dt
    else:
        this_dt = T - dt
    dA *= this_dt
    dA[:, :, -1] = A
    A = make_matrix(a * this_dt, 1 + b * this_dt, c * this_dt)
    inv_A = tridiag_inverse(A)
    assert (isId(inv_A.dot(A)))
    dInv = -tens_mult(tens_mult(inv_A, dA), inv_A)

    a, b, c, _, _, _ = give_me_everything(xx, N + eps, G)
    Ax = make_matrix(a * this_dt, 1 + b * this_dt, c * this_dt)
    invx = tridiag_inverse(Ax)
    assert (isId(invx.dot(Ax)))

    a, b, c, _, _, _ = give_me_everything(xx, N, G + eps)
    Ay = make_matrix(a * this_dt, 1 + b * this_dt, c * this_dt)
    invy = tridiag_inverse(Ay)
    assert (isId(invy.dot(Ay)))

    a, b, c, _, _, _ = give_me_everything(xx, N, G)
    k = math.floor((T + eps) / dt)
    if k * dt == T + eps:
        tmp_dt = dt
    else:
        tmp_dt = T + eps - dt
    Az = make_matrix(a * tmp_dt, 1 + b * tmp_dt, c * tmp_dt)
    invz = tridiag_inverse(Az)
    assert (isId(invz.dot(Az)))

    gr = numpy.zeros((len(xx), len(xx), 3))
    for i in range(0, len(xx)):
        for j in range(0, len(xx)):
            gr[i][j][0] = (invx[i, j] - inv_A[i, j]) / eps
            gr[i][j][1] = (invy[i, j] - inv_A[i, j]) / eps
            gr[i][j][2] = (invz[i, j] - inv_A[i, j]) / eps

    print(dInv - gr)


def testCell(params):
    """
    Функция проверяет правильно ли мы посчитали градиент АЧС по Phi.
    """
    N, T, G = params
    S = Spectrum.from_file(filename)
    xx = Numerics.default_grid(pts)
    phi, M = model_grad((N, T, G), S, ns, pts)
    cell = findCellGradient(xx)

    man = numpy.zeros((ns + 1, pts))
    for i in range(pts):
        phi[i] += eps
        man[:, i] = (Spectrum.from_phi(phi, [ns], [xx], force_direct=True).data - M.data) / eps
        phi[i] -= eps

    print(cell - man)


def testDmDphi(params):
    """
    Проверяем правильно ли мы посчитали d log H / d Phi
    (первый множитель в произведении, которое дает искомый градиент).
    Можем сравнивать как всю сумму(1 x G), так и каждое слагаемое по отдельности(N x G).
    """
    N, T, G = params
    S = Spectrum.from_file(filename)
    xx = Numerics.default_grid(pts)
    phi, M = model_grad((N, T, G), S, ns, pts)

    dHdPhi = findLLdPhi(xx, M, S, True)
    grDhDphi = numpy.zeros(pts)

    grM = Inference.ll_multinom_per_bin(M, S)

    for i in range(pts):
        phi[i] += eps

        tmpM = Spectrum.from_phi(phi, [ns], [xx], force_direct=True)
        tmpV = Inference.ll_multinom_per_bin(tmpM, S)

        grDhDphi[i] = (tmpV.sum() - grM.sum()) / eps

        phi[i] -= eps

    print(dHdPhi - grDhDphi)


def testDphiDtheta(params):
    """
    Проверяем правильно ли мы посчитали d log Phi / d Theta
    (второй множитель в произведении, которое дает искомый градиент).
    Реализация практически совпадает с тестом градиента.
    """
    N, T, G = params
    S = Spectrum.from_file(filename)
    phi, M, _, dphi = model_grad((N, T, G), S, ns, pts, 2)

    phix, Mx = model_grad((N + eps, T, G), S, ns, pts)
    phiy, My = model_grad((N, T, G + eps), S, ns, pts)
    phiz, Mz = model_grad((N, T + eps, G), S, ns, pts)
    gr = numpy.zeros((pts, 3))

    for i in range(pts):
        gr[i][0] = (phix[i] - phi[i]) / eps
        gr[i][1] = (phiy[i] - phi[i]) / eps
        gr[i][2] = (phiz[i] - phi[i]) / eps

    print(dphi - gr)


def testGradient(params):
    """
    Тестируем правильно ли посчитался градиент.
    """
    N, T, G = params
    S = Spectrum.from_file(filename)
    phi, M, gradient = model_grad((N, T, G), S, ns, pts, 1)
    ll = Inference.ll_multinom(M, S)

    phix, Mx = model_grad((N + eps, T, G), S, ns, pts)
    phiy, My = model_grad((N, T, G + eps), S, ns, pts)
    phiz, Mz = model_grad((N, T + eps, G), S, ns, pts)

    llx = Inference.ll_multinom(Mx, S)
    lly = Inference.ll_multinom(My, S)
    llz = Inference.ll_multinom(Mz, S)
    grad = [(llx - ll) / eps, (lly - ll) / eps, (llz - ll) / eps]

    print(gradient - grad)


if __name__ == "__main__":
    dt = 0.000596
    params = (0.149, 0.023, 0)
    testGradient(params)
