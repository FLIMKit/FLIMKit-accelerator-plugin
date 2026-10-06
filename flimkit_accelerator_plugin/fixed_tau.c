/* Fixed-lifetime projection, one pixel per iteration. Same arithmetic as
 * FLIMKit's fixed-tau path, so the maps agree with it to rounding.
 *
 * Threads are pthreads, or CreateThread on Windows, rather than OpenMP. A
 * second OpenMP runtime in the process aborts it, and PyTorch carries its
 * own, so an OpenMP build of this kernel dies with OMP: Error #15 as soon as
 * either runtime initialises second. FLIMKIT_C_THREADS caps the pool. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int flimkit_c_abi_version(void) { return 2; }

static int cpu_count(void)
{
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return (int)info.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

EXPORT int flimkit_c_threads(void)
{
    const char *env = getenv("FLIMKIT_C_THREADS");
    int n = cpu_count();
    if (env != NULL) {
        int want = atoi(env);
        if (want > 0 && want < n)
            n = want;
    }
    return n < 1 ? 1 : n;
}

typedef struct {
    const float *decay;
    const float *bg;
    int64_t n_bins;
    const double *a_pinv;
    const double *a;
    int64_t n_exp;
    double *amps;
    double *numerator;
    double *expected;
    uint8_t *valid;
    int64_t lo;
    int64_t hi;
} range_t;

static void fit_range(const range_t *g)
{
    int64_t n_bins = g->n_bins, n_exp = g->n_exp;
    for (int64_t p = g->lo; p < g->hi; p++) {
        const float *d = g->decay + p * n_bins;
        double *amp = g->amps + p * n_exp;
        double b = g->bg[p];
        for (int64_t e = 0; e < n_exp; e++) {
            const double *row = g->a_pinv + e * n_bins;
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
                m += amp[e] * g->a[k * n_exp + e];
            double dk = d[k];
            if (!isfinite(dk) || dk < 0.0 || !isfinite(m) || m < 0.0)
                ok = 0;
            double r = dk - m;
            num += r * r / (m > 1.0 ? m : 1.0);
            exp_sum += m < 1.0 ? m : 1.0;
        }
        g->numerator[p] = num;
        g->expected[p] = exp_sum;
        g->valid[p] = ok;
    }
}

#if defined(_WIN32)
static DWORD WINAPI worker(LPVOID arg)
{
    fit_range((const range_t *)arg);
    return 0;
}
#else
static void *worker(void *arg)
{
    fit_range((const range_t *)arg);
    return NULL;
}
#endif

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
    if (n_pix < 1)
        return;
    range_t whole;
    whole.decay = decay;
    whole.bg = bg;
    whole.n_bins = n_bins;
    whole.a_pinv = a_pinv;
    whole.a = a;
    whole.n_exp = n_exp;
    whole.amps = amps;
    whole.numerator = numerator;
    whole.expected = expected;
    whole.valid = valid;
    whole.lo = 0;
    whole.hi = n_pix;
    int n_threads = flimkit_c_threads();
    if ((int64_t)n_threads > n_pix)
        n_threads = (int)n_pix;
    if (n_threads < 2) {
        fit_range(&whole);
        return;
    }
    range_t *parts = (range_t *)malloc((size_t)n_threads * sizeof(range_t));
#if defined(_WIN32)
    HANDLE *handles = (HANDLE *)malloc((size_t)n_threads * sizeof(HANDLE));
#else
    pthread_t *handles = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
#endif
    if (parts == NULL || handles == NULL) {
        free(parts);
        free(handles);
        fit_range(&whole);
        return;
    }
    int64_t base = n_pix / n_threads, rem = n_pix % n_threads, at = 0;
    for (int t = 0; t < n_threads; t++) {
        parts[t] = whole;
        parts[t].lo = at;
        at += base + (t < rem ? 1 : 0);
        parts[t].hi = at;
    }
    int started = 0;
    for (int t = 0; t + 1 < n_threads; t++) {
#if defined(_WIN32)
        handles[t] = CreateThread(NULL, 0, worker, &parts[t], 0, NULL);
        if (handles[t] == NULL)
            break;
#else
        if (pthread_create(&handles[t], NULL, worker, &parts[t]) != 0)
            break;
#endif
        started++;
    }
    fit_range(&parts[n_threads - 1]);
    for (int t = started; t + 1 < n_threads; t++)
        fit_range(&parts[t]);
    for (int t = 0; t < started; t++) {
#if defined(_WIN32)
        WaitForSingleObject(handles[t], INFINITE);
        CloseHandle(handles[t]);
#else
        pthread_join(handles[t], NULL);
#endif
    }
    free(parts);
    free(handles);
}
