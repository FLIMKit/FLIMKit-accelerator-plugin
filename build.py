"""Compile the C fit backend and, optionally, install it as a FLIMKit add-on.

    python build.py              # build into flimkit_accelerator_plugin/
    python build.py --install    # build, then copy the folder into ~/.flimkit/plugins
    python build.py --no-cuda    # skip the CUDA kernel even when nvcc is present

The kernel threads itself with pthreads, so there is nothing to install first
and no OpenMP flag. FLIMKIT_C_THREADS caps the pool at run time.
"""
import argparse
import ctypes
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = HERE + os.sep + 'flimkit_accelerator_plugin'
SRC = PKG + os.sep + 'kernels.c'
CU_SRC = PKG + os.sep + 'kernels.cu'

def target():
    if sys.platform == 'win32':
        return 'flimkit_accelerator.dll'
    if sys.platform == 'darwin':
        return 'libflimkit_accelerator.dylib'
    return 'libflimkit_accelerator.so'

def cuda_target():
    if sys.platform == 'win32':
        return 'flimkit_accelerator_cuda.dll'
    return 'libflimkit_accelerator_cuda.so'

def compile_command(out):
    if sys.platform == 'win32':
        return ['cl', '/nologo', '/O2', '/LD', SRC, '/Fe:' + out]
    cc = os.environ.get('CC') or shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        sys.exit('No C compiler found. Install gcc or clang, or set CC.')
    shared = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared']
    cmd = [cc, '-O3', '-fPIC', *shared, SRC, '-o', out, '-lm']
    if sys.platform != 'darwin':
        cmd.insert(1, '-pthread')
    return cmd

def build():
    out = PKG + os.sep + target()
    cmd = compile_command(out)
    print('$ ' + ' '.join(cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit('Compilation failed.')
    lib = ctypes.CDLL(out)
    print('Built ' + out + ' (' + str(lib.flimkit_c_threads()) + ' thread(s))')
    return out

def build_cuda():
    out = PKG + os.sep + cuda_target()
    nvcc = os.environ.get('NVCC') or shutil.which('nvcc')
    if nvcc is None:
        print('No nvcc on PATH, skipping the CUDA kernel. The add-on fits on the CPU.')
        return None
    base = [nvcc, '-O3', '-shared', '-o', out]
    if sys.platform != 'win32':
        base += ['-Xcompiler', '-fPIC']
    attempts = [base[:1] + ['-arch=native'] + base[1:] + [CU_SRC], base + [CU_SRC]]
    for i, cmd in enumerate(attempts):
        print('$ ' + ' '.join(cmd))
        if subprocess.run(cmd).returncode == 0:
            break
        if i + 1 < len(attempts):
            print('  -arch=native needs CUDA 11.5 and a visible card, '
                  'building for the default architecture with a PTX fallback')
    else:
        print('CUDA build failed, leaving the add-on on the CPU kernel.')
        return None
    lib = ctypes.CDLL(out)
    print('Built ' + out + ' (device present: '
          + str(lib.flimkit_cu_available() == 1) + ')')
    return out

def install():
    dest = os.path.expanduser('~') + os.sep + '.flimkit' + os.sep + 'plugins' + os.sep + 'flimkit_accelerator_plugin'
    if os.path.isdir(dest) == True:
        shutil.rmtree(dest)
    shutil.copytree(PKG, dest, ignore=shutil.ignore_patterns('__pycache__'))
    print('Installed to ' + dest)
    print('Tick "Load from ~/.flimkit/plugins" in File > Preferences > Plugins and restart.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--install', action='store_true')
    ap.add_argument('--no-cuda', action='store_true')
    args = ap.parse_args()
    build()
    if args.no_cuda == False:
        build_cuda()
    if args.install == True:
        install()
