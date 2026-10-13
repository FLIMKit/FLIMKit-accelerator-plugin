/* The free-lifetime kernel from kernels.c on an NVIDIA GPU, in double precision.
 *
 * One thread per pixel. Each launch gives every unfinished pixel a bounded
 * number of Levenberg-Marquardt steps and saves its state, so no launch
 * outlasts the Windows display watchdog, and finished pixels drop out between
 * launches. The step sequence is the one in kernels.c, so the lifetimes match.
 *
 * flimkit_cu_fixed_tau_poisson is the Poisson maximum-likelihood fixed-lifetime
 * fit from kernels.c, one thread per pixel, run over pixels in bounded batches.
 *
 * Built only when nvcc is on PATH. Without it the add-on fits on the CPU. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define MAX_EXP 3
#define MAX_P 6
#define MAX_PP 36
#define BLOCK 128

extern "C" EXPORT int flimkit_cu_abi_version(void) { return 4; }

extern "C" EXPORT int flimkit_cu_available(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
        return 0;
    return n > 0 ? 1 : 0;
}

extern "C" EXPORT int flimkit_cu_threads(void) { return 0; }

__device__ static int cholesky_solve(double *A, double *b, int n)
{
    for (int j = 0; j < n; j++) {
        double s = A[j * n + j];
        for (int k = 0; k < j; k++)
            s -= A[j * n + k] * A[j * n + k];
        if (!(s > 0.0))
            return -1;
        double d = sqrt(s);
        A[j * n + j] = d;
        for (int i = j + 1; i < n; i++) {
            double t = A[i * n + j];
            for (int k = 0; k < j; k++)
                t -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = t / d;
        }
    }
    for (int i = 0; i < n; i++) {
        double t = b[i];
        for (int k = 0; k < i; k++)
            t -= A[i * n + k] * b[k];
        b[i] = t / A[i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double t = b[i];
        for (int k = i + 1; k < n; k++)
            t -= A[k * n + i] * b[k];
        b[i] = t / A[i * n + i];
    }
    return 0;
}

struct args_t {
    const double *data;
    const double *w_data;
    const double *bg;
    const double *irf;
    const int64_t *idx;
    int n_bins;
    int n_fit;
    int n_exp;
    double irf_sum;
    double dt;
    const double *bounds;
    int max_iter;
    double tol;
};

__device__ static double evaluate(const args_t a, const double *p, int b,
                                  double *A, double *g, int want_jac)
{
    int n = a.n_bins, ne = a.n_exp, np = 2 * ne;
    double r[MAX_EXP], drdt[MAX_EXP], rn[MAX_EXP], rn1[MAX_EXP];
    double y[MAX_EXP], z[MAX_EXP], jq[MAX_P];
    for (int j = 0; j < ne; j++) {
        double tau = p[j] > 1e-6 ? p[j] : 1e-6;
        r[j] = exp(-a.dt / tau);
        drdt[j] = r[j] * a.dt / (tau * tau);
        rn[j] = pow(r[j], (double)n);
        rn1[j] = pow(r[j], (double)(n - 1));
    }
    if (want_jac) {
        for (int k = 0; k < np; k++) {
            g[k] = 0.0;
            for (int l = 0; l < np; l++)
                A[k * np + l] = 0.0;
        }
    }
    double base = a.bg[b] * a.irf_sum, cost = 0.0;
    double y_prev[MAX_EXP], z_prev[MAX_EXP];
    for (int j = 0; j < ne; j++) {
        double y0 = a.irf[0], z0 = 0.0, rk = 1.0, rk1 = 1.0;
        for (int k = 1; k < n; k++) {
            rk1 = rk;
            rk *= r[j];
            y0 += rk * a.irf[n - k];
            z0 += (double)k * rk1 * a.irf[n - k];
        }
        y_prev[j] = y0;
        z_prev[j] = z0;
    }
    int q = 0;
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            for (int j = 0; j < ne; j++) {
                double yi = r[j] * y_prev[j] + (1.0 - rn[j]) * a.irf[i];
                double zi = y_prev[j] + r[j] * z_prev[j] - (double)n * rn1[j] * a.irf[i];
                y_prev[j] = yi;
                z_prev[j] = zi;
            }
        }
        if (q >= a.n_fit || a.idx[q] != (int64_t)i)
            continue;
        q++;
        double m = base;
        for (int j = 0; j < ne; j++) {
            y[j] = y_prev[j];
            z[j] = z_prev[j];
            m += p[ne + j] * y[j];
        }
        double w = sqrt(a.w_data[(size_t)b * n + i] > 1.0 ? a.w_data[(size_t)b * n + i] : 1.0);
        double ri = (m - a.data[(size_t)b * n + i]) / w;
        cost += ri * ri;
        if (want_jac) {
            for (int j = 0; j < ne; j++) {
                jq[j] = p[ne + j] * z[j] * drdt[j] / w;
                jq[ne + j] = y[j] / w;
            }
            for (int k = 0; k < np; k++) {
                g[k] -= jq[k] * ri;
                for (int l = 0; l <= k; l++)
                    A[k * np + l] += jq[k] * jq[l];
            }
        }
    }
    if (want_jac) {
        for (int k = 0; k < np; k++)
            for (int l = 0; l < k; l++)
                A[l * np + k] = A[k * np + l];
    }
    return cost;
}

__global__ static void fit_steps(args_t a, int max_steps, const int *active, int n_active,
                                 double *P, double *Ast, double *gst, double *cst,
                                 double *lamst, int *itst, int8_t *flags)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_active)
        return;
    int b = active[t];
    int ne = a.n_exp, np = 2 * ne;
    double p[MAX_P], pt[MAX_P], lo[MAX_P], hi[MAX_P];
    double A[MAX_PP], M[MAX_PP], g[MAX_P], step[MAX_P];
    for (int k = 0; k < np; k++) {
        lo[k] = a.bounds[MAX_P + k];
        hi[k] = a.bounds[2 * MAX_P + k];
    }
    int fresh = flags[b] == 0;
    double cost = 0.0, lam = 0.0;
    int it = 0;
    if (fresh) {
        for (int k = 0; k < np; k++) {
            double v = a.bounds[k];
            p[k] = v < lo[k] ? lo[k] : (v > hi[k] ? hi[k] : v);
            pt[k] = p[k];
        }
        cost = evaluate(a, p, b, A, g, 1);
        lam = 1e-3;
        fresh = 0;
    } else {
        for (int k = 0; k < np; k++) {
            p[k] = P[(size_t)b * MAX_P + k];
            g[k] = gst[(size_t)b * MAX_P + k];
        }
        for (int k = 0; k < np * np; k++)
            A[k] = Ast[(size_t)b * MAX_PP + k];
        cost = cst[b];
        lam = lamst[b];
        it = itst[b];
    }
    int done = 0;
    for (int steps = 0; steps < max_steps; steps++) {
        if (lam >= 1e12) {
            done = 1;
            break;
        }
        for (int k = 0; k < np * np; k++)
            M[k] = A[k];
        for (int k = 0; k < np; k++) {
            double dk = A[k * np + k];
            M[k * np + k] += lam * (dk > 1e-30 ? dk : 1e-30);
            step[k] = g[k];
        }
        for (int k = 0; k < np; k++) {
            if ((p[k] <= lo[k] && g[k] < 0.0) || (p[k] >= hi[k] && g[k] > 0.0)) {
                for (int l = 0; l < np; l++) {
                    M[k * np + l] = 0.0;
                    M[l * np + k] = 0.0;
                }
                M[k * np + k] = 1.0;
                step[k] = 0.0;
            }
        }
        if (cholesky_solve(M, step, np) != 0) {
            lam *= 10.0;
            continue;
        }
        for (int k = 0; k < np; k++) {
            double v = p[k] + step[k];
            pt[k] = v < lo[k] ? lo[k] : (v > hi[k] ? hi[k] : v);
        }
        double ct = evaluate(a, pt, b, NULL, NULL, 0);
        if (ct < cost) {
            double rel = (cost - ct) / (cost > 1e-300 ? cost : 1e-300);
            double dp = 0.0, pn = 0.0;
            for (int k = 0; k < np; k++) {
                dp += (pt[k] - p[k]) * (pt[k] - p[k]);
                pn += p[k] * p[k];
                p[k] = pt[k];
            }
            cost = evaluate(a, p, b, A, g, 1);
            lam = lam * 0.1 > 1e-12 ? lam * 0.1 : 1e-12;
            if (rel < a.tol || dp <= a.tol * a.tol * (pn + a.tol)) {
                done = 1;
                break;
            }
            it++;
            if (it >= a.max_iter) {
                done = 1;
                break;
            }
        } else {
            lam *= 10.0;
        }
    }
    for (int k = 0; k < np; k++)
        P[(size_t)b * MAX_P + k] = p[k];
    cst[b] = cost;
    itst[b] = it;
    if (done) {
        flags[b] = 2;
    } else {
        flags[b] = 1;
        lamst[b] = lam;
        for (int k = 0; k < np; k++)
            gst[(size_t)b * MAX_P + k] = g[k];
        for (int k = 0; k < np * np; k++)
            Ast[(size_t)b * MAX_PP + k] = A[k];
    }
}

__global__ static void chi2_kernel(args_t a, const double *P, int n_pix,
                                   double *numerator, double *expected, uint8_t *valid)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_pix)
        return;
    int n = a.n_bins, ne = a.n_exp;
    double p[MAX_P];
    for (int k = 0; k < 2 * ne; k++)
        p[k] = P[(size_t)b * MAX_P + k];
    double r[MAX_EXP], rn[MAX_EXP], y_prev[MAX_EXP];
    for (int j = 0; j < ne; j++) {
        double tau = p[j] > 1e-6 ? p[j] : 1e-6;
        r[j] = exp(-a.dt / tau);
        rn[j] = pow(r[j], (double)n);
        double y0 = a.irf[0], rk = 1.0;
        for (int k = 1; k < n; k++) {
            rk *= r[j];
            y0 += rk * a.irf[n - k];
        }
        y_prev[j] = y0;
    }
    double base = a.bg[b] * a.irf_sum, num = 0.0, exp_sum = 0.0;
    uint8_t ok = 1;
    int q = 0;
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            for (int j = 0; j < ne; j++)
                y_prev[j] = r[j] * y_prev[j] + (1.0 - rn[j]) * a.irf[i];
        }
        if (q >= a.n_fit || a.idx[q] != (int64_t)i)
            continue;
        q++;
        double m = base;
        for (int j = 0; j < ne; j++)
            m += p[ne + j] * y_prev[j];
        double dk = a.data[(size_t)b * n + i];
        if (!isfinite(dk) || dk < 0.0 || !isfinite(m) || m < 0.0)
            ok = 0;
        double rr = dk - m;
        num += rr * rr / (m > 1.0 ? m : 1.0);
        exp_sum += m < 1.0 ? m : 1.0;
    }
    numerator[b] = num;
    expected[b] = exp_sum;
    valid[b] = ok;
}

#define CHECK(call) do { if ((call) != cudaSuccess) { rc = -3; goto done; } } while (0)

/* Same arguments and the same meaning as flimkit_c_free_tau in kernels.c.
 * Returns 0, -1 on a bad shape, -3 on any CUDA failure, which hands the fit
 * back so the caller can use the CPU kernel. */
extern "C" EXPORT int flimkit_cu_free_tau(
    const double *data, const double *w_data, const double *bg,
    int64_t n_pix, int64_t n_bins, int64_t n_exp,
    const double *irf, const int64_t *idx, int64_t n_fit, double dt,
    const double *p0, const double *lo, const double *hi,
    int64_t max_iter, double tol,
    double *out_p, double *out_cost, int32_t *out_iter,
    double *numerator, double *expected, uint8_t *valid)
{
    if (n_exp < 1 || n_exp > MAX_EXP || n_bins < 2 || n_fit < 1 || n_pix < 1)
        return -1;
    int rc = 0;
    size_t cells = (size_t)n_pix * (size_t)n_bins;
    double *d_data = NULL, *d_w = NULL, *d_bg = NULL, *d_irf = NULL, *d_bounds = NULL;
    double *d_P = NULL, *d_A = NULL, *d_g = NULL, *d_cost = NULL, *d_lam = NULL;
    double *d_num = NULL, *d_exp = NULL;
    int64_t *d_idx = NULL;
    int32_t *d_it = NULL;
    int *d_active = NULL;
    int8_t *d_flags = NULL;
    uint8_t *d_valid = NULL;
    int8_t *flags = (int8_t *)calloc((size_t)n_pix, 1);
    int *active = (int *)malloc((size_t)n_pix * sizeof(int));
    double bounds[3 * MAX_P];
    if (flags == NULL || active == NULL) {
        rc = -2;
        goto done;
    }
    for (int k = 0; k < 3 * MAX_P; k++)
        bounds[k] = 0.0;
    for (int k = 0; k < 2 * (int)n_exp; k++) {
        bounds[k] = p0[k];
        bounds[MAX_P + k] = lo[k];
        bounds[2 * MAX_P + k] = hi[k];
    }
    CHECK(cudaMalloc(&d_data, cells * sizeof(double)));
    CHECK(cudaMalloc(&d_w, cells * sizeof(double)));
    CHECK(cudaMalloc(&d_bg, (size_t)n_pix * sizeof(double)));
    CHECK(cudaMalloc(&d_irf, (size_t)n_bins * sizeof(double)));
    CHECK(cudaMalloc(&d_idx, (size_t)n_fit * sizeof(int64_t)));
    CHECK(cudaMalloc(&d_bounds, 3 * MAX_P * sizeof(double)));
    CHECK(cudaMalloc(&d_P, (size_t)n_pix * MAX_P * sizeof(double)));
    CHECK(cudaMalloc(&d_A, (size_t)n_pix * MAX_PP * sizeof(double)));
    CHECK(cudaMalloc(&d_g, (size_t)n_pix * MAX_P * sizeof(double)));
    CHECK(cudaMalloc(&d_cost, (size_t)n_pix * sizeof(double)));
    CHECK(cudaMalloc(&d_lam, (size_t)n_pix * sizeof(double)));
    CHECK(cudaMalloc(&d_it, (size_t)n_pix * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_num, (size_t)n_pix * sizeof(double)));
    CHECK(cudaMalloc(&d_exp, (size_t)n_pix * sizeof(double)));
    CHECK(cudaMalloc(&d_valid, (size_t)n_pix));
    CHECK(cudaMalloc(&d_flags, (size_t)n_pix));
    CHECK(cudaMalloc(&d_active, (size_t)n_pix * sizeof(int)));
    CHECK(cudaMemcpy(d_data, data, cells * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, w_data, cells * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bg, bg, (size_t)n_pix * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_irf, irf, (size_t)n_bins * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_idx, idx, (size_t)n_fit * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bounds, bounds, 3 * MAX_P * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_flags, 0, (size_t)n_pix));

    args_t a;
    a.data = d_data;
    a.w_data = d_w;
    a.bg = d_bg;
    a.irf = d_irf;
    a.idx = d_idx;
    a.n_bins = (int)n_bins;
    a.n_fit = (int)n_fit;
    a.n_exp = (int)n_exp;
    a.irf_sum = 0.0;
    for (int64_t i = 0; i < n_bins; i++)
        a.irf_sum += irf[i];
    a.dt = dt;
    a.bounds = d_bounds;
    a.max_iter = (int)max_iter;
    a.tol = tol;

    {
        int64_t n_active = n_pix;
        for (int64_t i = 0; i < n_pix; i++)
            active[i] = (int)i;
        int max_steps = 4;
        int64_t max_px = n_pix;
        while (n_active > 0) {
            int64_t part = n_active < max_px ? n_active : max_px;
            CHECK(cudaMemcpy(d_active, active, (size_t)part * sizeof(int),
                             cudaMemcpyHostToDevice));
            int grid = (int)((part + BLOCK - 1) / BLOCK);
            cudaEvent_t t0, t1;
            CHECK(cudaEventCreate(&t0));
            CHECK(cudaEventCreate(&t1));
            CHECK(cudaEventRecord(t0));
            fit_steps<<<grid, BLOCK>>>(a, max_steps, d_active, (int)part, d_P, d_A, d_g,
                                       d_cost, d_lam, d_it, d_flags);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, t0, t1));
            cudaEventDestroy(t0);
            cudaEventDestroy(t1);
            if (cudaGetLastError() != cudaSuccess) {
                rc = -3;
                goto done;
            }
            CHECK(cudaMemcpy(flags, d_flags, (size_t)n_pix, cudaMemcpyDeviceToHost));
            int64_t before = n_active, kept = 0;
            for (int64_t i = 0; i < n_active; i++) {
                if (flags[active[i]] != 2)
                    active[kept++] = active[i];
            }
            n_active = kept;
            if (ms < 150.0f && part == before && max_steps < 256)
                max_steps *= 2;
            else if (ms > 300.0f) {
                if (max_steps > 1)
                    max_steps /= 2;
                else if (max_px > BLOCK)
                    max_px /= 2;
            }
        }
    }
    {
        int grid = (int)((n_pix + BLOCK - 1) / BLOCK);
        chi2_kernel<<<grid, BLOCK>>>(a, d_P, (int)n_pix, d_num, d_exp, d_valid);
        CHECK(cudaDeviceSynchronize());
        if (cudaGetLastError() != cudaSuccess) {
            rc = -3;
            goto done;
        }
    }
    {
        double *staged = (double *)malloc((size_t)n_pix * MAX_P * sizeof(double));
        if (staged == NULL) {
            rc = -2;
            goto done;
        }
        if (cudaMemcpy(staged, d_P, (size_t)n_pix * MAX_P * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            free(staged);
            rc = -3;
            goto done;
        }
        for (int64_t b = 0; b < n_pix; b++)
            for (int64_t k = 0; k < 2 * n_exp; k++)
                out_p[b * 2 * n_exp + k] = staged[b * MAX_P + k];
        free(staged);
    }
    CHECK(cudaMemcpy(out_cost, d_cost, (size_t)n_pix * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(out_iter, d_it, (size_t)n_pix * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(numerator, d_num, (size_t)n_pix * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(expected, d_exp, (size_t)n_pix * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(valid, d_valid, (size_t)n_pix, cudaMemcpyDeviceToHost));
    for (int64_t b = 0; b < n_pix; b++) {
        if (!isfinite(out_cost[b]))
            out_iter[b] = -1;
    }
done:
    free(flags);
    free(active);
    cudaFree(d_data);
    cudaFree(d_w);
    cudaFree(d_bg);
    cudaFree(d_irf);
    cudaFree(d_idx);
    cudaFree(d_bounds);
    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_g);
    cudaFree(d_cost);
    cudaFree(d_lam);
    cudaFree(d_it);
    cudaFree(d_num);
    cudaFree(d_exp);
    cudaFree(d_valid);
    cudaFree(d_flags);
    cudaFree(d_active);
    return rc;
}

struct poisson_args {
    const float *decay;
    int *idx;
    const double *a;
    const double *colsum;
    int n_bins;
    int n_exp;
    int iters;
    double tol;
};

__global__ static void poisson_kernel(poisson_args g, int n, double *amps, double *offset,
                                      double *numerator, double *expected, uint8_t *valid)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    int n_bins = g.n_bins, n_exp = g.n_exp, n_col = n_exp + 1;
    double inv_bins = 1.0 / (double)n_bins;
    const float *d = g.decay + (size_t)p * n_bins;
    int *idx = g.idx + (size_t)p * n_bins;
    double *amp = amps + (size_t)p * n_exp;
    double a[MAX_EXP + 1], num[MAX_EXP + 1];
    int m = 0;
    double tot = 0.0;
    for (int k = 0; k < n_bins; k++) {
        if (d[k] > 0.0f) {
            idx[m++] = k;
            tot += d[k];
        }
    }
    for (int e = 0; e < n_exp; e++)
        amp[e] = 0.0;
    offset[p] = 0.0;
    if (m == 0) {
        numerator[p] = 0.0;
        expected[p] = 0.0;
        valid[p] = 0;
        return;
    }
    for (int c = 0; c < n_col; c++)
        a[c] = tot / (double)n_col / g.colsum[c];
    for (int it = 0; it < g.iters; it++) {
        for (int c = 0; c < n_col; c++)
            num[c] = 0.0;
        for (int j = 0; j < m; j++) {
            int b = idx[j];
            const double *row = g.a + (size_t)b * n_exp;
            double mu = a[n_exp] * inv_bins;
            for (int e = 0; e < n_exp; e++)
                mu += a[e] * row[e];
            if (mu < 1e-12)
                mu = 1e-12;
            double r = (double)d[b] / mu;
            for (int e = 0; e < n_exp; e++)
                num[e] += r * row[e];
            num[n_exp] += r * inv_bins;
        }
        double dmax = 0.0;
        for (int c = 0; c < n_col; c++) {
            double next = a[c] * num[c] / g.colsum[c];
            double diff = fabs(next - a[c]) / (a[c] + 1e-12);
            if (diff > dmax)
                dmax = diff;
            a[c] = next;
        }
        if (dmax < g.tol)
            break;
    }
    double b0 = a[n_exp] * inv_bins;
    for (int e = 0; e < n_exp; e++)
        amp[e] = a[e];
    offset[p] = b0;
    double sq = 0.0, exp_sum = 0.0;
    uint8_t ok = 1;
    for (int k = 0; k < n_bins; k++) {
        double mm = b0;
        for (int e = 0; e < n_exp; e++)
            mm += amp[e] * g.a[(size_t)k * n_exp + e];
        double dk = d[k];
        if (!isfinite(dk) || dk < 0.0 || !isfinite(mm) || mm < 0.0)
            ok = 0;
        double r = dk - mm;
        sq += r * r / (mm > 1.0 ? mm : 1.0);
        exp_sum += mm < 1.0 ? mm : 1.0;
    }
    numerator[p] = sq;
    expected[p] = exp_sum;
    valid[p] = ok;
}

/* Same arguments and the same meaning as flimkit_c_fixed_tau_poisson in kernels.c.
 * Returns 0, -1 on a bad shape, -2 when host memory ran out, -3 on any CUDA
 * failure, which hands the fit back so the caller can use the CPU kernel. */
extern "C" EXPORT int flimkit_cu_fixed_tau_poisson(
    const float *decay, int64_t n_pix, int64_t n_bins, const double *a, int64_t n_exp,
    int64_t iters, double tol,
    double *amps, double *offset, double *numerator, double *expected, uint8_t *valid)
{
    if (n_exp < 1 || n_exp > MAX_EXP || n_bins < 1)
        return -1;
    int rc = 0;
    double colsum[MAX_EXP + 1];
    double *d_a = NULL, *d_colsum = NULL, *d_amps = NULL, *d_off = NULL;
    double *d_num = NULL, *d_exp = NULL;
    float *d_decay = NULL;
    int *d_idx = NULL;
    uint8_t *d_valid = NULL;
    int64_t chunk = (int64_t)(192u * 1024u * 1024u) / (8 * n_bins);
    if (chunk > 131072)
        chunk = 131072;
    if (chunk < BLOCK)
        chunk = BLOCK;
    if (chunk > n_pix)
        chunk = n_pix;
    for (int64_t e = 0; e < n_exp; e++) {
        colsum[e] = 0.0;
        for (int64_t k = 0; k < n_bins; k++)
            colsum[e] += a[k * n_exp + e];
    }
    colsum[n_exp] = 1.0;
    CHECK(cudaMalloc(&d_a, (size_t)(n_bins * n_exp) * sizeof(double)));
    CHECK(cudaMalloc(&d_colsum, (size_t)(n_exp + 1) * sizeof(double)));
    CHECK(cudaMalloc(&d_decay, (size_t)(chunk * n_bins) * sizeof(float)));
    CHECK(cudaMalloc(&d_idx, (size_t)(chunk * n_bins) * sizeof(int)));
    CHECK(cudaMalloc(&d_amps, (size_t)(chunk * n_exp) * sizeof(double)));
    CHECK(cudaMalloc(&d_off, (size_t)chunk * sizeof(double)));
    CHECK(cudaMalloc(&d_num, (size_t)chunk * sizeof(double)));
    CHECK(cudaMalloc(&d_exp, (size_t)chunk * sizeof(double)));
    CHECK(cudaMalloc(&d_valid, (size_t)chunk));
    CHECK(cudaMemcpy(d_a, a, (size_t)(n_bins * n_exp) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_colsum, colsum, (size_t)(n_exp + 1) * sizeof(double),
                     cudaMemcpyHostToDevice));
    for (int64_t first = 0; first < n_pix; first += chunk) {
        int64_t part = n_pix - first < chunk ? n_pix - first : chunk;
        poisson_args g;
        g.decay = d_decay;
        g.idx = d_idx;
        g.a = d_a;
        g.colsum = d_colsum;
        g.n_bins = (int)n_bins;
        g.n_exp = (int)n_exp;
        g.iters = (int)iters;
        g.tol = tol;
        CHECK(cudaMemcpy(d_decay, decay + first * n_bins, (size_t)(part * n_bins) * sizeof(float),
                         cudaMemcpyHostToDevice));
        int grid = (int)((part + BLOCK - 1) / BLOCK);
        poisson_kernel<<<grid, BLOCK>>>(g, (int)part, d_amps, d_off, d_num, d_exp, d_valid);
        CHECK(cudaDeviceSynchronize());
        if (cudaGetLastError() != cudaSuccess) {
            rc = -3;
            goto done;
        }
        CHECK(cudaMemcpy(amps + first * n_exp, d_amps, (size_t)(part * n_exp) * sizeof(double),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(offset + first, d_off, (size_t)part * sizeof(double),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(numerator + first, d_num, (size_t)part * sizeof(double),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(expected + first, d_exp, (size_t)part * sizeof(double),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(valid + first, d_valid, (size_t)part, cudaMemcpyDeviceToHost));
    }
done:
    cudaFree(d_a);
    cudaFree(d_colsum);
    cudaFree(d_decay);
    cudaFree(d_idx);
    cudaFree(d_amps);
    cudaFree(d_off);
    cudaFree(d_num);
    cudaFree(d_exp);
    cudaFree(d_valid);
    return rc;
}
