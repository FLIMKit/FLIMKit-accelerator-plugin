# flimkit-accelerator-plugin

A [FLIMKit](https://github.com/FLIMKit/FLIMKit) fit backend written in C and loaded with `ctypes`, registered through the `@fit_backend` add-on hook.

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
- **Windows:** MSVC, run from a Developer Command Prompt.

No Python headers are needed, and nothing has to be installed for the threads.

Then tick "Load from ~/.flimkit/plugins" in `File > Preferences > Plugins` and restart. The Progress log says `using add-on fit backend 'accelerator'` on the first fit.

It has to be installed as a folder, not as a wheel. Python cannot load compiled libraries from inside a zip ([zipimport](https://docs.python.org/3/library/zipimport.html)), and FLIMKit loads a folder from disk, so `ctypes` can open the library beside `__init__.py`. That also works in the compiled app, which has no `pip`.

## What it covers

Only the plain case: a fixed-lifetime fit with no fit window, background profile or pile-up correction. Everything else returns `None`, which hands the fit back to FLIMKit: to the GPU backend this one displaced, or to the CPU. One-exponential scans, free-lifetime fits and distribution fits are left to FLIMKit too.

That is the pattern for adding more kernels: write the C, add a method on `CFixedTau` with the signature in `flimkit/GPU/_base.py`, and decline the cases you have not covered.

To compare it against the built-in backends without uninstalling it, set `FLIMKIT_FIT_BACKEND` to `cuda`, `mlx`, `cpu` or `accelerator`.

`__init__.py` checks the library's ABI version and refuses one built by an older copy of `build.py`, so rerun `python build.py --install` after pulling. The ABI is 2 as of the move off OpenMP.

## Licence

MIT, same as FLIMKit.
