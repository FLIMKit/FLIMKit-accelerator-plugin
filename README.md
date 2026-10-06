# flimkit-accelerator-plugin

A [FLIMKit](https://github.com/FLIMKit/FLIMKit) fit backend written in C and loaded with `ctypes`, registered through the `@fit_backend` add-on hook.

It runs fixed-lifetime per-pixel fits: background subtraction, projection onto the fixed-lifetime basis, clamped amplitudes and the χ² terms, one pixel per loop iteration, on every core through OpenMP. The maps match FLIMKit's own CPU path to rounding (about 1e-15 relative). On a synthetic 512×512×256 stack with 4 threads it took 1–3 s against 50–56 s for FLIMKit's CPU path (19–51× across runs on a shared machine). It has not been compared against FLIMKit's GPU backends yet.

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
- **macOS:** the Xcode command line tools, plus `brew install libomp` for more than one thread.
- **Windows:** MSVC, run from a Developer Command Prompt.

No Python headers are needed.

Then tick "Load from ~/.flimkit/plugins" in `File > Preferences > Plugins` and restart. The Progress log says `using add-on fit backend 'accelerator'` on the first fit.

It has to be installed as a folder, not as a wheel. Python cannot load compiled libraries from inside a zip ([zipimport](https://docs.python.org/3/library/zipimport.html)), and FLIMKit loads a folder from disk, so `ctypes` can open the library beside `__init__.py`. That also works in the compiled app, which has no `pip`.

## What it covers

Only the plain case: a fixed-lifetime fit with no fit window, background profile or pile-up correction. Everything else returns `None`, which hands the fit back to FLIMKit: to the GPU backend this one displaced, or to the CPU. One-exponential scans, free-lifetime fits and distribution fits are left to FLIMKit too.

That is the pattern for adding more kernels: write the C, add a method on `CFixedTau` with the signature in `flimkit/GPU/_base.py`, and decline the cases you have not covered.

To compare it against the built-in backends without uninstalling it, set `FLIMKIT_FIT_BACKEND` to `cuda`, `mlx`, `cpu` or `accelerator`.

## Licence

MIT, same as FLIMKit.
