from cmath import exp, pi


class Fourier:

    def fft(self, x):
        """Forward recursive fast Fourier transform."""
        N = len(x)
        if N <= 1: return x
        even = self.fft(x[0::2])
        odd = self.fft(x[1::2])
        T = [exp(-2j * pi * k / N) * odd[k] for k in range(N // 2)]
        return [even[k] + T[k] for k in range(N // 2)] + \
               [even[k] - T[k] for k in range(N // 2)]

    def ifft(self, x):
        """@TODO Inverse recursive fast Fourier transform."""
        pass
