# -*- coding: utf-8 -*-
"""
Szybka, wektorowa wersja p_weierstrass_matrix z only_e22_real_400_gpu_debug.py.

Oryginał liczy każdy element macierzy osobno, w pętli Pythona, w mpmath, i dla
każdego elementu od nowa wyznacza pierwiastki, AGM i q. Tutaj:
  * stałe sieci (e3, w1, q) liczone są RAZ, tą samą procedurą co w oryginale,
  * ta sama formuła theta
        p(z) = e3 + (pi*th2(0)*th3(0)*th4(z/w1) / (pi*w1*th1(z/w1)))**2
    liczona jest naraz dla całej macierzy (NumPy na CPU albo CuPy na GPU),
  * szeregi theta w konwencji mpmath.jtheta, obcięte do NTERMS wyrazów
    (dla g2=189.073, g3=0 jest |q| = e^-pi ~ 0.043, więc q^(n^2) < 1e-16 już dla n ~ 4).

Działa z GPU (CuPy) i bez GPU (NumPy). Wymuszenie CPU: zmienna środowiskowa PFAST_CPU=1.
"""
import os
import types
import numpy as np
from mpmath import mp, polyroots, agm, sqrt as msqrt, exp, pi as mpi, jtheta

NTERMS = 12

# --- Backend: CuPy jeśli jest (i działa), inaczej NumPy -------------------------------
cp = None
if os.environ.get('PFAST_CPU', '') != '1':
    try:
        import cupy as _cupy
        _cupy.zeros(1)            # sprawdza, czy jest GPU i sterownik
        cp = _cupy
    except Exception:
        cp = None
NA_GPU = cp is not None
if cp is None:
    # moduł zachowujący się jak cupy, ale liczący w NumPy (cp.asarray, cp.sum, ..., cp.asnumpy)
    cp = types.ModuleType('cp_numpy')
    cp.__dict__.update(np.__dict__)
    cp.asnumpy = np.asarray
print(f"[pfast] backend: {'GPU (CuPy)' if NA_GPU else 'CPU (NumPy)'}")


def _stale_sieci(g2, g3):
    """e3, w1, q i stała C — dokładnie jak w p_weierstrass_from_g2_g3 z oryginału."""
    mp.dps = 15
    r1, r2, r3 = polyroots([4, 0, -g2, -g3])
    e3 = r3
    a, b, c = msqrt(r1 - r3), msqrt(r1 - r2), msqrt(r2 - r3)
    if abs(a + b) < abs(a - b):
        b *= -1
    if abs(a + c) < abs(a - c):
        c *= -1
    if abs(c + 1j * b) < abs(c - 1j * b):
        e3 = r1
        a, b, c = msqrt(r3 - r1), msqrt(r3 - r2), msqrt(r2 - r1)
        w1 = 1 / agm(1j * b, c)
    else:
        w1 = 1 / agm(a, b)
    w3 = 1j / agm(a, c)
    q = exp(1j * mpi * w3 / w1)
    C = mpi * jtheta(2, 0, q) * jtheta(3, 0, q) / (mpi * w1)
    return complex(e3), complex(w1), complex(q), complex(C)


def p_weierstrass_matrix_fast(z, g2=189.073, g3=0):
    """Jak p_weierstrass_matrix z oryginału: p(z) elementowo, diagonala (z == 0) = 0.
    z może być tablicą NumPy albo CuPy — wynik jest tego samego typu."""
    xp = np
    if NA_GPU:
        xp = cp.get_array_module(z)
    e3, w1, q, C = _stale_sieci(g2, g3)
    u = z / w1
    # th1 bez czynnika q^(1/4); ten czynnik skraca się z th2(0) w C, stąd C / q^(1/4)
    th1 = xp.zeros_like(u)
    th4 = xp.ones_like(u)
    for n in range(NTERMS):
        th1 += 2 * (-1) ** n * q ** (n * (n + 1)) * xp.sin((2 * n + 1) * u)
        if n >= 1:
            th4 += 2 * (-1) ** n * q ** (n * n) * xp.cos(2 * n * u)
    Cq = C / q ** 0.25
    nz = z != 0
    th1 = xp.where(nz, th1, 1)
    wynik = e3 + (Cq * th4 / th1) ** 2
    return xp.where(nz, wynik, 0).astype(xp.complex128)
