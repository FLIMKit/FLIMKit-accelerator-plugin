/* FLIMKit add-on fit kernels in C, loaded through ctypes.
 *
 * flimkit_c_fixed_tau: fixed-lifetime projection, one pixel per iteration.
 * flimkit_c_free_tau:  per-pixel free-lifetime reconvolution fit, up to three
 *                      exponentials, the same projected Levenberg-Marquardt and
 *                      the same step sequence as FLIMKit's numba kernel.
 * Both reproduce FLIMKit's own arithmetic, so the maps agree to rounding.
 *
 * Threads are pthreads, or CreateThread on Windows, rather than OpenMP. A
 * second OpenMP runtime in the process aborts it, and PyTorch carries its own,
 * so an OpenMP build of these kernels dies with OMP: Error #15 as soon as
 * either runtime initialises second. Threads take chunks from a shared counter
 * rather than one block each, because slow pixels cluster in real images.
 * FLIMKIT_C_THREADS caps the pool. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
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

#define MAX_EXP 3
#define MAX_P (2 * MAX_EXP)
#define CHUNK 256

EXPORT int flimkit_c_abi_version(void) { return 3; }

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

typedef void (*chunk_fn)(void *work, void *local, int64_t lo, int64_t hi);
typedef void *(*local_new_fn)(void *work);

typedef struct {
    chunk_fn fn;
    local_new_fn local_new;
    void *work;
    int64_t n;
    int64_t next;
#if defined(_WIN32)
    CRITICAL_SECTION lock;
#else
    pthread_mutex_t lock;
#endif
    int failed;
} pool_t;

static int64_t pool_take(pool_t *p)
{
    int64_t lo;
#if defined(_WIN32)
    EnterCriticalSection(&p->lock);
    lo = p->next;
    p->next += CHUNK;
    LeaveCriticalSection(&p->lock);
#else
    pthread_mutex_lock(&p->lock);
    lo = p->next;
    p->next += CHUNK;
    pthread_mutex_unlock(&p->lock);
#endif
    return lo;
}

static void pool_run(pool_t *p)
{
    void *local = NULL;
    if (p->local_new != NULL) {
        local = p->local_new(p->work);
        if (local == NULL) {
            p->failed = 1;
            return;
        }
    }
    for (;;) {
        int64_t lo = pool_take(p);
        if (lo >= p->n)
            break;
        int64_t hi = lo + CHUNK < p->n ? lo + CHUNK : p->n;
        p->fn(p->work, local, lo, hi);
    }
    free(local);
}

#if defined(_WIN32)
static DWORD WINAPI pool_entry(LPVOID arg)
{
    pool_run((pool_t *)arg);
    return 0;
}
#else
static void *pool_entry(void *arg)
{
    pool_run((pool_t *)arg);
    return NULL;
}
#endif

static int parallel_for(chunk_fn fn, local_new_fn local_new, void *work, int64_t n)
{
    if (n < 1)
        return 0;
    pool_t p;
    p.fn = fn;
    p.local_new = local_new;
    p.work = work;
    p.n = n;
    p.next = 0;
    p.failed = 0;
#if defined(_WIN32)
    InitializeCriticalSection(&p.lock);
#else
    pthread_mutex_init(&p.lock, NULL);
#endif
    int n_threads = flimkit_c_threads();
    if ((int64_t)n_threads * CHUNK > n)
        n_threads = (int)((n + CHUNK - 1) / CHUNK);
    if (n_threads < 1)
        n_threads = 1;
#if defined(_WIN32)
    HANDLE *handles = (HANDLE *)malloc((size_t)n_threads * sizeof(HANDLE));
#else
    pthread_t *handles = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
#endif
    int started = 0;
    if (handles != NULL) {
        for (int t = 0; t + 1 < n_threads; t++) {
#if defined(_WIN32)
            handles[t] = CreateThread(NULL, 0, pool_entry, &p, 0, NULL);
            if (handles[t] == NULL)
                break;
#else
            if (pthread_create(&handles[t], NULL, pool_entry, &p) != 0)
                break;
#endif
            started++;
        }
    }
    pool_run(&p);
    for (int t = 0; t < started; t++) {
#if defined(_WIN32)
        WaitForSingleObject(handles[t], INFINITE);
        CloseHandle(handles[t]);
#else
        pthread_join(handles[t], NULL);
#endif
    }
    free(handles);
#if defined(_WIN32)
    DeleteCriticalSection(&p.lock);
#else
    pthread_mutex_destroy(&p.lock);
#endif
    return p.failed == 1 ? -2 : 0;
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
} fixed_t;

static void fixed_chunk(void *work, void *local, int64_t lo, int64_t hi)
{
    const fixed_t *g = (const fixed_t *)work;
    int64_t n_bins = g->n_bins, n_exp = g->n_exp;
    (void)local;
    for (int64_t p = lo; p < hi; p++) {
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
    fixed_t g;
    g.decay = decay;
    g.bg = bg;
    g.n_bins = n_bins;
    g.a_pinv = a_pinv;
    g.a = a;
    g.n_exp = n_exp;
    g.amps = amps;
    g.numerator = numerator;
    g.expected = expected;
    g.valid = valid;
    parallel_for(fixed_chunk, NULL, &g, n_pix);
}

static void exp_conv(const double *irf, int64_t n, double r, double *y, double *z)
{
    double rn = pow(r, (double)n), rn1 = pow(r, (double)(n - 1));
    double y0 = irf[0], z0 = 0.0, rk = 1.0, rk1 = 1.0;
    for (int64_t k = 1; k < n; k++) {
        rk1 = rk;
        rk *= r;
        y0 += rk * irf[n - k];
        z0 += (double)k * rk1 * irf[n - k];
    }
    y[0] = y0;
    z[0] = z0;
    for (int64_t i = 1; i < n; i++) {
        y[i] = r * y[i - 1] + (1.0 - rn) * irf[i];
        z[i] = y[i - 1] + r * z[i - 1] - (double)n * rn1 * irf[i];
    }
}

static int cholesky_solve(double *A, double *b, int64_t n)
{
    for (int64_t j = 0; j < n; j++) {
        double s = A[j * n + j];
        for (int64_t k = 0; k < j; k++)
            s -= A[j * n + k] * A[j * n + k];
        if ((s > 0.0) == 0)
            return -1;
        double d = sqrt(s);
        A[j * n + j] = d;
        for (int64_t i = j + 1; i < n; i++) {
            double t = A[i * n + j];
            for (int64_t k = 0; k < j; k++)
                t -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = t / d;
        }
    }
    for (int64_t i = 0; i < n; i++) {
        double t = b[i];
        for (int64_t k = 0; k < i; k++)
            t -= A[i * n + k] * b[k];
        b[i] = t / A[i * n + i];
    }
    for (int64_t i = n - 1; i >= 0; i--) {
        double t = b[i];
        for (int64_t k = i + 1; k < n; k++)
            t -= A[k * n + i] * b[k];
        b[i] = t / A[i * n + i];
    }
    return 0;
}

typedef struct {
    const double *data;
    const double *w_data;
    const double *bg;
    const double *irf;
    const int64_t *idx;
    int64_t n_bins;
    int64_t n_fit;
    int64_t n_exp;
    double irf_sum;
    double dt;
    const double *p0;
    const double *lo;
    const double *hi;
    int64_t max_iter;
    double tol;
    double *out_p;
    double *out_cost;
    int32_t *out_iter;
    double *numerator;
    double *expected;
    uint8_t *valid;
} free_t;

typedef struct {
    double *y;
    double *z;
    double *res;
    double *J;
    double *w;
} scratch_t;

static void *free_local_new(void *work)
{
    const free_t *g = (const free_t *)work;
    int64_t np = 2 * g->n_exp;
    size_t doubles = (size_t)(2 * g->n_exp * g->n_bins + g->n_fit * (np + 1) + g->n_bins);
    scratch_t *s = (scratch_t *)malloc(sizeof(scratch_t) + doubles * sizeof(double));
    if (s == NULL)
        return NULL;
    double *buf = (double *)(s + 1);
    s->y = buf;
    s->z = s->y + g->n_exp * g->n_bins;
    s->J = s->z + g->n_exp * g->n_bins;
    s->res = s->J + g->n_fit * np;
    s->w = s->res + g->n_fit;
    return s;
}

static double evaluate(const free_t *g, scratch_t *s, const double *p,
                       const double *d, double bg, int want_J)
{
    int64_t n = g->n_bins, ne = g->n_exp, np = 2 * ne;
    double drdt[MAX_EXP];
    for (int64_t j = 0; j < ne; j++) {
        double tau = p[j] > 1e-6 ? p[j] : 1e-6;
        double r = exp(-g->dt / tau);
        drdt[j] = r * g->dt / (tau * tau);
        exp_conv(g->irf, n, r, s->y + j * n, s->z + j * n);
    }
    double base = bg * g->irf_sum, cost = 0.0;
    for (int64_t q = 0; q < g->n_fit; q++) {
        int64_t i = g->idx[q];
        double m = base;
        for (int64_t j = 0; j < ne; j++)
            m += p[ne + j] * s->y[j * n + i];
        double ri = (m - d[i]) / s->w[i];
        cost += ri * ri;
        if (want_J == 1) {
            s->res[q] = ri;
            for (int64_t j = 0; j < ne; j++) {
                s->J[q * np + j] = p[ne + j] * s->z[j * n + i] * drdt[j] / s->w[i];
                s->J[q * np + ne + j] = s->y[j * n + i] / s->w[i];
            }
        }
    }
    return cost;
}

static void chi2_terms(const free_t *g, scratch_t *s, const double *p,
                       const double *d, double bg, int64_t b)
{
    int64_t n = g->n_bins, ne = g->n_exp;
    for (int64_t j = 0; j < ne; j++) {
        double tau = p[j] > 1e-6 ? p[j] : 1e-6;
        exp_conv(g->irf, n, exp(-g->dt / tau), s->y + j * n, s->z + j * n);
    }
    double base = bg * g->irf_sum, num = 0.0, exp_sum = 0.0;
    uint8_t ok = 1;
    for (int64_t q = 0; q < g->n_fit; q++) {
        int64_t i = g->idx[q];
        double m = base;
        for (int64_t j = 0; j < ne; j++)
            m += p[ne + j] * s->y[j * n + i];
        double dk = d[i];
        if (!isfinite(dk) || dk < 0.0 || !isfinite(m) || m < 0.0)
            ok = 0;
        double r = dk - m;
        num += r * r / (m > 1.0 ? m : 1.0);
        exp_sum += m < 1.0 ? m : 1.0;
    }
    g->numerator[b] = num;
    g->expected[b] = exp_sum;
    g->valid[b] = ok;
}

static void free_chunk(void *work, void *local, int64_t lo, int64_t hi)
{
    const free_t *g = (const free_t *)work;
    scratch_t *s = (scratch_t *)local;
    int64_t n = g->n_bins, ne = g->n_exp, np = 2 * ne;
    for (int64_t b = lo; b < hi; b++) {
        const double *d = g->data + b * n;
        const double *wd = g->w_data + b * n;
        for (int64_t i = 0; i < n; i++)
            s->w[i] = sqrt(wd[i] > 1.0 ? wd[i] : 1.0);
        double p[MAX_P], pt[MAX_P], A[MAX_P * MAX_P], M[MAX_P * MAX_P];
        double gr[MAX_P], step[MAX_P];
        memcpy(p, g->p0, sizeof(double) * (size_t)np);
        for (int64_t k = 0; k < np; k++) {
            if (p[k] < g->lo[k]) p[k] = g->lo[k];
            if (p[k] > g->hi[k]) p[k] = g->hi[k];
        }
        double lam = 1e-3;
        double cost = evaluate(g, s, p, d, g->bg[b], 1);
        int64_t n_it = g->max_iter, it;
        for (it = 0; it < g->max_iter; it++) {
            for (int64_t k = 0; k < np; k++) {
                gr[k] = 0.0;
                for (int64_t l = 0; l < np; l++)
                    A[k * np + l] = 0.0;
            }
            for (int64_t q = 0; q < g->n_fit; q++) {
                const double *Jq = s->J + q * np;
                for (int64_t k = 0; k < np; k++) {
                    gr[k] -= Jq[k] * s->res[q];
                    for (int64_t l = 0; l <= k; l++)
                        A[k * np + l] += Jq[k] * Jq[l];
                }
            }
            for (int64_t k = 0; k < np; k++)
                for (int64_t l = 0; l < k; l++)
                    A[l * np + k] = A[k * np + l];
            int accepted = 0;
            while (lam < 1e12) {
                memcpy(M, A, sizeof(double) * (size_t)(np * np));
                for (int64_t k = 0; k < np; k++) {
                    double dk = A[k * np + k];
                    M[k * np + k] += lam * (dk > 1e-30 ? dk : 1e-30);
                    step[k] = gr[k];
                }
                for (int64_t k = 0; k < np; k++) {
                    if ((p[k] <= g->lo[k] && gr[k] < 0.0)
                            || (p[k] >= g->hi[k] && gr[k] > 0.0)) {
                        for (int64_t l = 0; l < np; l++) {
                            M[k * np + l] = 0.0;
                            M[l * np + k] = 0.0;
                        }
                        M[k * np + k] = 1.0;
                        step[k] = 0.0;
                    }
                }
                if (cholesky_solve(M, step, np) == 0) {
                    for (int64_t k = 0; k < np; k++) {
                        double v = p[k] + step[k];
                        pt[k] = v < g->lo[k] ? g->lo[k] : (v > g->hi[k] ? g->hi[k] : v);
                    }
                    double ct = evaluate(g, s, pt, d, g->bg[b], 0);
                    if (ct < cost) {
                        double rel = (cost - ct) / (cost > 1e-300 ? cost : 1e-300);
                        double dp = 0.0, pn = 0.0;
                        for (int64_t k = 0; k < np; k++) {
                            dp += (pt[k] - p[k]) * (pt[k] - p[k]);
                            pn += p[k] * p[k];
                            p[k] = pt[k];
                        }
                        cost = evaluate(g, s, p, d, g->bg[b], 1);
                        lam = lam * 0.1 > 1e-12 ? lam * 0.1 : 1e-12;
                        accepted = (rel < g->tol
                                    || dp <= g->tol * g->tol * (pn + g->tol)) ? 2 : 1;
                        break;
                    }
                }
                lam *= 10.0;
            }
            if (accepted != 1) {
                n_it = it;
                break;
            }
        }
        memcpy(g->out_p + b * np, p, sizeof(double) * (size_t)np);
        g->out_cost[b] = cost;
        g->out_iter[b] = isfinite(cost) ? (int32_t)n_it : -1;
        chi2_terms(g, s, p, d, g->bg[b], b);
    }
}

/* data       n_pix x n_bins, float64, C order (the decay that is fitted, so
 *            pile-up corrected where that applies)
 * w_data     n_pix x n_bins, float64 (raw counts, weights are sqrt(max(w, 1)))
 * bg         n_pix, float64, one constant background per pixel
 * irf        n_bins, float64, already shifted, broadened and tailed
 * idx        n_fit bin indices, int64
 * dt         tcspc_res in the same units as p0, lo and hi
 * p0, lo, hi 2 * n_exp, lifetimes then amplitudes
 * out_p      n_pix x 2 * n_exp, float64, written
 * out_cost   n_pix, float64, written
 * out_iter   n_pix, int32, written, -1 when the cost is not finite
 * numerator, expected  n_pix, float64, written, FLIMKit's chi2 terms
 * valid      n_pix, uint8, written
 * Returns 0, -1 on a bad shape, -2 when a thread could not allocate.
 */
EXPORT int flimkit_c_free_tau(
    const double *data, const double *w_data, const double *bg,
    int64_t n_pix, int64_t n_bins, int64_t n_exp,
    const double *irf, const int64_t *idx, int64_t n_fit, double dt,
    const double *p0, const double *lo, const double *hi,
    int64_t max_iter, double tol,
    double *out_p, double *out_cost, int32_t *out_iter,
    double *numerator, double *expected, uint8_t *valid)
{
    if (n_exp < 1 || n_exp > MAX_EXP || n_bins < 2 || n_fit < 1)
        return -1;
    free_t g;
    g.data = data;
    g.w_data = w_data;
    g.bg = bg;
    g.irf = irf;
    g.idx = idx;
    g.n_bins = n_bins;
    g.n_fit = n_fit;
    g.n_exp = n_exp;
    g.irf_sum = 0.0;
    for (int64_t i = 0; i < n_bins; i++)
        g.irf_sum += irf[i];
    g.dt = dt;
    g.p0 = p0;
    g.lo = lo;
    g.hi = hi;
    g.max_iter = max_iter;
    g.tol = tol;
    g.out_p = out_p;
    g.out_cost = out_cost;
    g.out_iter = out_iter;
    g.numerator = numerator;
    g.expected = expected;
    g.valid = valid;
    return parallel_for(free_chunk, free_local_new, &g, n_pix);
}
