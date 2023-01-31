import numpy as np
import scipy
from scipy import integrate


# Function to generate the Fibonacci sequence of order s
def fibzp(s):
    t1 = [0]
    t2 = [1]
    while s >= 2:
        t = t2 + t1
        t1 = t2
        t2 = t
        s = s - 1
    return t


# Function to generate the Triadic Cantor sequence of order s
def cantorzp(s):
    t = [1, 0, 1]
    t1 = [1, 0, 1]
    if s != 1:
        while s >= 2:
            for y in range(len(t)):
                if t[y] == 1:
                    t1 = t1[0:3 * (y)] + [1, 0, 1] + t1[3 * (y) + 2:]
                else:
                    t1 = t1[0:3 * (y)] + [0, 0, 0] + t1[3 * (y) + 2:]
            t = t1
            s = s - 1
    return t


# Transmittance function in 1 dimension corresponding to a ZP with an arbitrary sequence t
def transmitancia(chi, t):
    def transmitancia(chi, t):
        N = len(t)
        transm = np.linspace(0, 1, len(chi))
        for k in range(N):
            mask = (chi > k / N) & (chi < (k + 1) / N)
            transm[mask] = t[k]

        return transm


# Transmittance function in 2 dimensions corresponding to a ZP with an arbitrary sequence t
def transmitancia2dgeneral(x0, y0, t):
    x0matrix, y0matrix = np.meshgrid(x0, y0)
    chimatrix = x0matrix ** 2 + y0matrix ** 2

    N = len(t)
    transm = np.zeros((len(x0), len(y0)))
    for k in range(N):
        mask = (chimatrix > k / N) & (chimatrix < (k + 1) / N)
        transm[mask] = t[k]

    return transm


# Function to calculate the irradiance along the optical axis using the Simpson method. t = sequence of the ZP  u=reduced axial coordinate M=number of sampling points
def Intensidadaxialnum1d(t, u, M):
    Intensidadnum = np.linspace(0, 1, len(u))
    sumsimp = 1j * np.linspace(0, 1, len(u))
    chi = np.linspace(0, 1, M)

    for i in range(len(u)):
        integrando = transmitancia(chi, t) * np.exp(-1j * 2 * np.pi * u[i] * chi)
        sumsimp[i] = 1 * (chi[1] - chi[0]) / 3 * (integrando[0] + integrando[len(integrando) - 1])
        sumsimp[i] = sumsimp[i] + 4 * (chi[1] - chi[0]) / 3 * sum(integrando[1:len(integrando) - 1:2])
        sumsimp[i] = sumsimp[i] + 2 * (chi[1] - chi[0]) / 3 * sum(integrando[2:len(integrando) - 1:2])
        Intensidadnum[i] = 4 * np.pi ** 2 * u[i] ** 2 * abs(sumsimp[i]) ** 2

    return Intensidadnum


# Function to calculate the irradiance along the optical axis using the 1D FFT method. t = sequence of the ZP  u=reduced axial coordinate M=number of sampling points
def Intensidadaxialfft1d(t, u, M):
    Intensidadfft = np.linspace(0, 1, len(u))

    chi = np.linspace(0, 1, M)

    transm = 0 * np.linspace(0, 1, int(M / (u[1] - u[0])))

    transm[:len(chi)] = transmitancia(chi, t)

    transm = np.fft.fftshift(transm)
    fourier = (chi[1] - chi[0]) * np.fft.fft(transm)
    for i in range(len(u)):
        Intensidadfft[i] = 4 * np.pi ** 2 * u[i] ** 2 * abs(fourier[i]) ** 2

    return Intensidadfft


# Function to calculate the irradiance in a volume (x,y,u) the 2D FFT method. t = sequence of the ZP  u=reduced axial coordinate M=number of sampling points
def Intensidadfft2dxyu(t, x, y, u, M):
    A = 1
    x0 = np.linspace(-1, 1, M)
    y0 = np.linspace(-1, 1, M)
    stepx = abs(x[1] - x[0])
    stepy = abs(y[1] - y[0])
    x0matrix, y0matrix = np.meshgrid(x0, y0)
    chimatrix = x0matrix ** 2 + y0matrix ** 2
    transm = transmitancia2dgeneral(x0, y0, t)
    Intensidadxyu = np.zeros((len(y), len(x), len(u)))
    for i in range(len(u)):
        Mt = round(M / abs(x[1] - x[0]) / 2 / u[i] / 2)
        Nt = round(M / abs(y[1] - y[0]) / 2 / u[i] / 2)
        integrando = 1j * np.zeros((Mt, Nt))
        integrando[:len(x0), :len(y0)] = transm * np.exp(1j * np.pi * 2 * u[i] * chimatrix)
        for s in range(len(y0)):
            for k in range(len(x0)):
                integrando[s, k] = integrando[s, k] * np.exp(1j * 2 * np.pi * x[len(x)] / stepx * k / Mt) * np.exp(
                    1j * 2 * np.pi * y[len(y)] / stepy * s / Nt)
        fourier2d = 4 * u[i] ** 2 * (x0[1] - x0[0]) ** 2 * np.fft.fft2(integrando)
        Intensidadxyu[:len(y), :len(x), i] = A ** 2 / 4 / (u[i] ** 2) * abs(fourier2d[:len(y), :len(x)]) ** 2
    return Intensidadxyu


# Function to calculate the irradiance along the optical axis using the 2D gaussian quadrature method. t = sequence of the ZP  u=reduced axial coordinate M=number of sampling points
def Intensidadgauss2daxial(t, u, M):
    A = 1
    Intensidadaxial = np.linspace(0, 1, len(u))
    for i in range(len(u)):
        def integrando(x0, y0):
            X0, Y0 = np.meshgrid(x0, y0)
            return transmitancia2dgeneral(x0, y0, t) * np.exp(1j * 2 * np.pi * u[i] * (X0 ** 2 + Y0 ** 2))

        def fint_fixed_quad(x0):
            return integrate.fixed_quad(integrando, -1, 1, args=(x0,), n=M)[0]

        Intensidadaxial[i] = A ** 2 * 4 * u[i] ** 2 * abs(integrate.fixed_quad(fint_fixed_quad, -1, 1, n=M)[0]) ** 2

    return Intensidadaxial