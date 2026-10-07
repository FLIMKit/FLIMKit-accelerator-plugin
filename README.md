# flimkit-accelerator-plugin

A [FLIMKit](https://github.com/FLIMKit/FLIMKit) fit backend written in C and loaded with `ctypes`, registered through the `@fit_backend` add-on hook. There is a CUDA build of the free-lifetime kernel as well, used when nvcc was on PATH at build time and a card is present.

It runs fixed-lifetime per-pixel fits: background subtraction, projection onto the fixed-lifetime basis, clamped amplitudes and the χ² terms, one pixel per loop iteration, on every core. The maps match FLIMKit's own CPU path to rounding (about 1e-15 relative). On a synthetic 512×512×256 stack with 4 threads it took 1–3 s against 50–56 s for FLIMKit's CPU path (19–51× across runs on a shared machine). It has not been compared against FLIMKit's GPU backends yet.

The threads are pthreads, or `CreateThread` on Windows, and not OpenMP. PyTorch carries its own OpenMP runtime, and a second one in the same process aborts it: an OpenMP build of this kernel died with `OMP: Error #15` whichever order the two were loaded in, measured on macOS with Homebrew's libomp. That is an abort, not an exception, so nothing can catch it and the FLIMKit window goes with it. Dropping OpenMP costs about 8 per cent on this kernel, 33.8 ms against 31.3 ms for 262,144 pixels at 256 bins on a 10-core M4, and the fitted amplitudes are bit-identical. `FLIMKIT_C_THREADS` caps the pool, and one thread takes 162.1 ms on the same data.

Requires a FLIMKit with the `@fit_backend` hook.

## What is in it

| File | What it does |
|---|---|
| `flimkit_accelerator_plugin/fixed_tau.c` | The kernel |
| `flimkit_accelerator_plugin/__init__.py` | Finds the shared library next to itself, declares its argument types, and wraps it as a backend whose `batch_fixed_tau` fills FLIMKit's maps |
| `build.py` | Compiles the library for this machine and, with `--install`, copies the plugin folder into `~/.flimkit/plugins/` |

## Installing

```bash
git clone https://github.com/FLIMKit/flimkit-accelerator-plugin
cd flimkit-accelerator-plugin
python build.py --install
```

That needs a C compiler:

- **Linux:** gcc or clang.
- **macOS:** the Xcode command line tools.
- **Windows:** `pip install ziglang`, which carries its own C compiler and Windows C runtime, so Visual Studio is not needed. MSVC works too if you have it. Zig's first Windows build takes a few minutes while it builds that runtime, and seconds after.
- **CUDA, optional:** nvcc from the CUDA toolkit. `build.py` builds the GPU kernel when it finds nvcc and skips it otherwise, so nothing breaks without it. `--no-cuda` skips it anyway. A local build uses `-arch=native`, so it compiles for the cards in that machine and nothing else, which is what you want on hardware the CI matrix does not cover, a Jetson for instance. Without a visible card it falls back to nvcc's default, which is an sm_52 binary plus PTX that every newer card has to compile for itself on first load.

No Python headers are needed, and nothing has to be installed for the threads.

Or skip the compiler. `.github/workflows/build.yml` builds the libraries for Linux x86_64 and aarch64, Windows x86_64 and arm64, and one universal2 for macOS, and attaches a drop-in folder per platform to the release. Download the one for your machine, unzip it into `~/.flimkit/plugins/`, and there is nothing to build. The CUDA library is built there too, for Pascal through Hopper with a PTX fallback for anything newer, so a card needs only its driver and no toolkit.

Then tick "Load from ~/.flimkit/plugins" in `File > Preferences > Plugins` and restart. The Progress log says `using add-on fit backend 'accelerator'` on the first fit.

It has to be installed as a folder, not as a wheel. Python cannot load compiled libraries from inside a zip ([zipimport](https://docs.python.org/3/library/zipimport.html)), and FLIMKit loads a folder from disk, so `ctypes` can open the library beside `__init__.py`. That also works in the compiled app, which has no `pip`.

## What it covers

**Fixed lifetime.** Only the plain case: no fit window, background profile or pile-up correction. Everything else returns `None`, which hands the fit back to FLIMKit: to the GPU backend this one displaced, or to the CPU.

**Free lifetime**, up to three exponentials, including a fit window and pile-up correction of the data. It declines a time-varying background, pile-up folded into the model, and more than three components. It is the same projected Levenberg-Marquardt as FLIMKit's numba kernel, step for step, so the two agree: on 16,384 pixels at 256 bins the lifetime, amplitude, fraction and χ² maps match to 1e-14 relative, and kernel against kernel on the same inputs the parameters sit 1.7e-12 apart with identical iteration counts.

One-exponential scans and distribution fits are left to FLIMKit.

| Fit | Where it runs | µs/px, 16,384 px at 256 bins, 10-core M4 |
|---|---|---|
| Free lifetime, this add-on | C, all cores | 5.7 |
| Free lifetime, FLIMKit | numba | 19.2 |
| Free lifetime, CUDA build | see below, slower than C on a 48-core box | |

That is the pattern for adding more kernels: write the C, add a method on `CFixedTau` with the signature in `flimkit/GPU/_base.py`, and decline the cases you have not covered.

To compare it against the built-in backends without uninstalling it, set `FLIMKIT_FIT_BACKEND` to `cuda`, `mlx`, `cpu` or `accelerator`.

The add-on's version is `__version__` in `__init__.py`, and it shows in the Progress log when the backend is picked, so a user can say which build they have without unzipping anything. Separately, `__init__.py` checks the library's ABI version and refuses one built by an older copy of `build.py`, so rerun `python build.py --install` after pulling. The ABI is 3 as of the free-lifetime kernel.

`FLIMKIT_C_THREADS` caps the CPU pool. Threads take 256-pixel chunks from a shared counter rather than one block each, because slow pixels cluster in real images.

## The CUDA kernel, and when it is worth building

`kernels.cu` is the free-lifetime kernel for an NVIDIA card, one thread per pixel in double precision, with the launch bounded to a few Levenberg-Marquardt steps at a time so no launch outlasts the Windows display watchdog. Finished pixels drop out between launches, and a CUDA failure at any point returns a non-zero code, which falls back to the C kernel.

It gives the same answer as the C kernel, inside floating point. Measured on an RTX A5000 with CUDA 12.1 against the C kernel on 48 cores, same synthetic data into both, 65,536 pixels per case:

| Case | Worst τ difference |
|---|---|
| 2-exp, 400 photons/px | 0.010 ps |
| 2-exp, 120 photons/px | 0.0002 ps |
| 3-exp, 4,000 photons/px | 0.740 ps |
| 2-exp, fit window from bin 30 | 2.911 ps |
| 1-exp, 2,000 photons/px | 0.000005 ps |
| 2-exp, 4,096 bins | 0.000000 ps |

Worst means one pixel out of 65,536. The median difference is below 1e-6 ps in every case, the reduced cost ratio is exactly 1.0, and the valid flags are identical. Shot noise on a 3 ns lifetime at 4,000 photons is around 60 to 70 ps, so none of this reaches a lifetime map.

What is left is nvcc contracting multiply-adds into fma, which the host compiler does differently. Building with `-fmad=false` drops the worst case to 0.0014 ps and makes the per-pixel iteration counts match everywhere, including the three-exponential case where they otherwise diverge, and costs 4 to 30 per cent. It is not the default, because the difference it removes is invisible next to the photon statistics. Reach for it when comparing CPU and GPU maps against each other directly, not for ordinary fitting.

Both kernels ran every case with no failures, and pixels that used all 200 iterations had their state saved and resumed across launches, which is the part the watchdog logic depends on.

It was slower than the C kernel on that machine, in every case:

| Pixels, 256 bins, 2-exp | C, 48 threads | CUDA, one A5000 |
|---|---|---|
| 4,096 | 6.21 µs/px | 69.44 µs/px |
| 65,536 | 2.81 µs/px | 9.99 µs/px |
| 262,144 | 2.67 µs/px | 5.11 µs/px |

Two things drive that. The fit is float64, and an A5000 runs double precision at a small fraction of its single-precision rate, so the card is working with one hand tied. And 48 cores is a lot of CPU to beat. The gap closes as the batch grows, from 11x down to 1.9x, which is the fixed per-launch cost being amortised.

Where it should win is the machine most people actually have: a few CPU cores next to a decent card. That has not been measured. Build it and compare before assuming either way, and if the GPU loses, `--no-cuda` or deleting the CUDA library leaves the C kernel in charge.

## Licence

MIT, same as FLIMKit.
