"""Compile the C fit backend and, optionally, install it as a FLIMKit add-on.

    python build.py              # build into flimkit_accelerator_plugin/
    python build.py --install    # build, then copy the folder into ~/.flimkit/plugins
    python build.py --no-openmp  # one thread, for a compiler without OpenMP
"""
import argparse
import ctypes
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = HERE + os.sep + 'flimkit_accelerator_plugin'
SRC = PKG + os.sep + 'fixed_tau.c'

def target():
    if sys.platform == 'win32':
        return 'flimkit_accelerator.dll'
    if sys.platform == 'darwin':
        return 'libflimkit_accelerator.dylib'
    return 'libflimkit_accelerator.so'

def compile_commands(out, openmp):
    if sys.platform == 'win32':
        cmd = ['cl', '/nologo', '/O2', '/LD', SRC, '/Fe:' + out]
        if openmp == True:
            return [cmd[:3] + ['/openmp'] + cmd[3:], cmd]
        return [cmd]
    cc = os.environ.get('CC') or shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        sys.exit('No C compiler found. Install gcc or clang, or set CC.')
    shared = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared']
    base = [cc, '-O3', '-fPIC', *shared, SRC, '-o', out, '-lm']
    if openmp == False:
        return [base]
    if sys.platform == 'darwin':
        # Apple clang needs libomp from Homebrew; fall back to one thread without it
        prefix = os.environ.get('LIBOMP_PREFIX', '/opt/homebrew/opt/libomp')
        omp = [cc, '-O3', '-fPIC', '-dynamiclib', '-Xpreprocessor', '-fopenmp',
               '-I' + prefix + '/include', SRC, '-o', out, '-L' + prefix + '/lib', '-lomp', '-lm']
        return [omp, base]
    return [base[:2] + ['-fopenmp'] + base[2:], base]

def build(openmp=True):
    out = PKG + os.sep + target()
    attempts = compile_commands(out, openmp)
    for i, cmd in enumerate(attempts):
        print('$ ' + ' '.join(cmd))
        if subprocess.run(cmd).returncode == 0:
            break
        if i + 1 < len(attempts):
            print('  failed, trying without OpenMP')
    else:
        sys.exit('Compilation failed.')
    lib = ctypes.CDLL(out)
    print('Built ' + out + ' (' + str(lib.flimkit_c_threads()) + ' thread(s))')
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
    ap.add_argument('--no-openmp', action='store_true')
    args = ap.parse_args()
    build(openmp=not args.no_openmp)
    if args.install == True:
        install()
