# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np
from typing import Callable
from inspect import isfunction


def func(x: int | float | np.ndarray) -> int | float | np.ndarray:
    """Funkcja wyliczająca wartości funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return np.exp(-2 * x) + x**2 - 1


def dfunc(x: np.ndarray) -> np.ndarray:
    """Funkcja wyliczająca wartości pierwszej pochodnej (df(x)) funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1
    df(x) = -2 * e^(-2x) + 2x

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return -2 * np.exp(-2 * x) + 2 * x


def ddfunc(x: np.ndarray) -> np.ndarray:
    """Funkcja wyliczająca wartości drugiej pochodnej (ddf(x)) funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1
    ddf(x) = 4 * e^(-2x) + 2

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return 4 * np.exp(-2 * x) + 2


def bisection(
    a: int | float,
    b: int | float,
    f: Callable[[float], float],
    epsilon: float,
    max_iter: int,
) -> tuple[float, int] | None:
    """Funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] 
    metodą bisekcji.

    Args:
        a (int | float): Początek przedziału.
        b (int | float): Koniec przedziału.
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        epsilon (float): Tolerancja zera maszynowego (warunek stopu).
        max_iter (int): Maksymalna liczba iteracji.

    Returns:
        (tuple[float, int]):
            - Aproksymowane rozwiązanie,
            - Liczba wykonanych iteracji.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if a >= b or epsilon <= 0 or max_iter <= 0:
        return None

    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        return None

    iter_count = 0

    while iter_count < max_iter:
        c = (a + b) / 2.0
        fc = f(c)
        iter_count += 1

        if abs(fc) < epsilon or abs(a - b) / 2.0 < epsilon:
            return c, iter_count

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, iter_count


def secant(
    a: int | float,
    b: int | float,
    f: Callable[[float], float],
    epsilon: float,
    max_iters: int,
) -> tuple[float, int] | None:
    """funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] 
    metodą siecznych.

    Args:
        a (int | float): Początek przedziału.
        b (int | float): Koniec przedziału.
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        epsilon (float): Tolerancja zera maszynowego (warunek stopu).
        max_iters (int): Maksymalna liczba iteracji.

    Returns:
        (tuple[float, int]):
            - Aproksymowane rozwiązanie,
            - Liczba wykonanych iteracji.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """


    if a == b or epsilon <= 0 or max_iters <= 0:
        return None

    fa = f(a)
    fb = f(b)

    for iter_count in range(1, max_iters + 1):

        if fb - fa == 0:
            return None

        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)

        if abs(fc) < epsilon:
            return c, iter_count

        a, fa = b, fb
        b, fb = c, fc

    return b, max_iters


def difference_quotient(
    f: Callable[[float], float], x: int | float, h: int | float
) -> float | None:
    """Funkcja obliczająca wartość iloazu różnicowego w punkcie x dla zadanej 
    funkcji f(x).

    Args:
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        x (int | float): Argument funkcji.
        h (int | float): Krok różnicy wykorzystywanej do wyliczenia ilorazu 
            różnicowego.

    Returns:
        (float): Wartość ilorazu różnicowego.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not callable(f):
        return None
    if not isinstance(x, (int, float)):
        return None
    if not isinstance(h, (int, float)):
        return None
    if h == 0:
        return None

    return (f(x + h) - f(x)) / h
    


def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    ddf: Callable[[float], float],
    a: int | float,
    b: int | float,
    epsilon: float,
    max_iter: int,
) -> tuple[float, int] | None:
    """Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.

    Args:
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        df (Callable[[float], float]): Pierwsza pochodna funkcji, dla której 
            poszukiwane jest rozwiązanie.
        ddf (Callable[[float], float]): Druga pochodna funkcji, dla której 
            poszukiwane jest rozwiązanie.
        a (int | float): Początek przedziału.
        b (int | float): Koniec przedziału.
        epsilon (float): Tolerancja zera maszynowego (warunek stopu).
        max_iter (int): Maksymalna liczba iteracji.

    Returns:
        (tuple[float, int]):
            - Aproksymowane rozwiązanie,
            - Liczba wykonanych iteracji.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not (isfunction(f) and isfunction(df) and isfunction(ddf)):
        return None

    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return None
    if not isinstance(max_iter, int):
        return None
    if not isinstance(epsilon, (int, float)):
        return None

    a = float(a)
    b = float(b)
    epsilon = float(epsilon)

    if a >= b or epsilon <= 0 or max_iter <= 0:
        return None

    try:
        fa = f(a)
        fb = f(b)
        dfa = df(a)
        dfb = df(b)
    except Exception:
        # coś poszło źle przy obliczaniu funkcji / pochodnych
        return None

    # musi być zmiana znaku na końcach przedziału
    if fa * fb > 0:
        return None

    # dla zbieżności Newtona: pochodna nie może zmieniać znaku
    if dfa * dfb <= 0:
        return None

    # --- wybór punktu startowego -------------------------------------------
    # najpierw środek przedziału
    x = 0.5 * (a + b)

    # jeśli warunek f(x)*f''(x) <= 0, to spróbuj końce przedziału
    try:
        if f(x) * ddf(x) <= 0:
            if f(a) * ddf(a) > 0:
                x = a
            elif f(b) * ddf(b) > 0:
                x = b
            else:
                # nie znaleziono dobrego punktu startowego
                return None
    except Exception:
        return None

    # --- właściwa iteracja Newtona -----------------------------------------
    iteration = 0
    while iteration < max_iter:
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            # nie wolno dzielić przez zero – przerwij
            return None

        x_new = x - fx / dfx
        iteration += 1

        # warunek stopu – dokładność w argumencie albo w wartości funkcji
        if abs(x_new - x) <= epsilon or abs(fx) <= epsilon:
            return (x_new, iteration)

        x = x_new

    # po max_iter zwróć ostatnią aproksymację
    return (x, iteration)
