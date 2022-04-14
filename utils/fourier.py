from cmath import exp, pi


def fft(x):
    """
    Forward recursive fast Fourier transform.
    """
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [exp(-2j * pi * k / N) * odd[k] for k in range(N // 2)]     #TODO: or range(N // 2 - 1) ?
    return [even[k] + T[k] for k in range(N // 2)] + \
           [even[k] - T[k] for k in range(N // 2)]

def ifft(self, x):
    """
    TODO Inverse recursive fast Fourier transform.
    """
    pass
