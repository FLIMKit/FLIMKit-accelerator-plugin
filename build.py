"""Compile the C fit backend and, optionally, install it as a FLIMKit add-on.

    python build.py              # build into flimkit_accelerator_plugin/
    python build.py --install    # build, then copy the folder into ~/.flimkit/plugins

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
SRC = PKG + os.sep + 'fixed_tau.c'

def target():
    if sys.platform == 'win32':
        return 'flimkit_accelerator.dll'
    if sys.platform == 'darwin':
        return 'libflimkit_accelerator.dylib'
    return 'libflimkit_accelerator.so'

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
    args = ap.parse_args()
    build()
    if args.install == True:
        install()
