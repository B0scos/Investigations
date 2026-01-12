import pywavelets as pwt
import numpy as np
import pandas as pd
from typing import List, Tuple


def wavelet_decompose(signal: np.ndarray, wavelet: str, level: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Decomposes a signal using discrete wavelet transform.

    Args:
        signal (np.ndarray): The input signal to decompose.
        wavelet (str): The type of wavelet to use for decomposition.
        level (int): The level of decomposition.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: A tuple containing two lists:
            - Approximation coefficients at each level.
            - Detail coefficients at each level.
    """
    coeffs = pwt.wavedec(signal, wavelet, level=level)
    approximation = [coeffs[0]]
    details = coeffs[1:]
    return approximation, details


if __name__ == "__main__":
    # Example usage
    signal = np.sin(np.linspace(0, 8 * np.pi, 512)) + 0.5 * np.random.randn(512)
    wavelet = 'haar'
    level = 4

    approximation, details = wavelet_decompose(signal, wavelet, level)

    print("Approximation Coefficients:", approximation) 
    print("Detail Coefficients:", details)