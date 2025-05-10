#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <fftw3.h>
#include <math.h>

// Function to create an array with linearly spaced elements, similar to np.arange
void linspace(double* arr, int start, int stop, int num) {
    for (int i = 0; i < num; i++) {
        arr[i] = start + i * (stop - start) / (double)(num - 1);
    }
}

int main() {
    int K = 200;
    double freq = 2;
    double c = 343;
    double d = 20;
    double doa = 30;
    double t[K];
    linspace(t, 1, K, K);

    // Initialize s1 similar to the Python code
    double s1[K];
    for (int i = 0; i < K; i++) {
        s1[i] = sin(2 * M_PI * freq * t[i]);
    }

    // Prepare FFTW
    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * K);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * K);
    fftw_plan plan_forward = fftw_plan_dft_1d(K, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Fill in input for FFT
    for (int i = 0; i < K; i++) {
        in[i] = s1[i] + 0.0 * I; // Create complex number with imaginary part 0
    }

    // Execute FFT
    fftw_execute(plan_forward);

    // Apply the phase shift
    for (int i = 0; i < K; i++) {
        double w = (i <= K / 2) ? i : i - K;
        double phase_shift = -2 * M_PI * w * (d / c) * sin(doa * M_PI / 180.0);
        out[i] *= cexp(-I * phase_shift);
    }

    // Prepare for inverse FFT
    fftw_complex *y_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * K);
    fftw_plan plan_backward = fftw_plan_dft_1d(K, out, y_complex, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Execute inverse FFT
    fftw_execute(plan_backward);

    // Get the real part of the result
    double y[K];
    for (int i = 0; i < K; i++) {
        y[i] = creal(y_complex[i]) / K; // Normalize by dividing by K (FFT normalization)
    }

    // Print the first few results for verification
    for (int i = 0; i < 5; i++) {
        printf("y[%d] = %f\n", i, y[i]);
    }

    // Free resources
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_free(in);
    fftw_free(out);
    fftw_free(y_complex);

    return 0;
}