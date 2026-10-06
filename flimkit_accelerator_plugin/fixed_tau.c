/* Fixed-lifetime projection, one pixel per iteration. Same arithmetic as
 * FLIMKit's fixed-tau path, so the maps agree with it to rounding. */
#include <math.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int flimkit_c_abi_version(void) { return 1; }

EXPORT int flimkit_c_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* decay      n_pix x n_bins, float32, C order (raw counts)
 * bg         n_pix, float32
 * a_pinv     n_exp x n_bins, float64 (pinv of A)
 * a          n_bins x n_exp, float64 (convolved basis)
 * amps       n_pix x n_exp, float64, written
 * numerator, expected  n_pix, float64, written
 * valid      n_pix, uint8, written
 */
EXPORT void flimkit_c_fixed_tau(
    const float *decay, const float *bg, int64_t n_pix, int64_t n_bins,
    const double *a_pinv, const double *a, int64_t n_exp,
    double *amps, double *numerator, double *expected, uint8_t *valid)
{
    int64_t p;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (p = 0; p < n_pix; p++) {
        const float *d = decay + p * n_bins;
        double *amp = amps + p * n_exp;
        double b = bg[p];
        for (int64_t e = 0; e < n_exp; e++) {
            const double *row = a_pinv + e * n_bins;
            double s = 0.0;
            for (int64_t k = 0; k < n_bins; k++) {
                double c = d[k] - b;
                s += row[k] * (c > 0.0 ? c : 0.0);
            }
            amp[e] = s > 0.0 ? s : 0.0;
        }
        double num = 0.0, exp_sum = 0.0;
        uint8_t ok = 1;
        for (int64_t k = 0; k < n_bins; k++) {
            double m = b;
            for (int64_t e = 0; e < n_exp; e++)
                m += amp[e] * a[k * n_exp + e];
            double dk = d[k];
            if (!isfinite(dk) || dk < 0.0 || !isfinite(m) || m < 0.0)
                ok = 0;
            double r = dk - m;
            num += r * r / (m > 1.0 ? m : 1.0);
            exp_sum += m < 1.0 ? m : 1.0;
        }
        numerator[p] = num;
        expected[p] = exp_sum;
        valid[p] = ok;
    }
}
