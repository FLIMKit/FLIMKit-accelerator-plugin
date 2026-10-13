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


def cuda_arch_flags():
    env = os.environ.get('CUDA_ARCHS')
    if env is not None and env.strip() != '':
        caps = [part.strip() for part in env.split(',') if part.strip()]
    else:
        nvidia_smi = shutil.which('nvidia-smi')
        caps = []
        if nvidia_smi is not None:
            try:
                out = subprocess.run([
                    nvidia_smi,
                    '--query-gpu=compute_cap',
                    '--format=csv,noheader'],
                    capture_output=True, text=True, check=True).stdout
                caps = [cap.strip() for cap in out.splitlines() if cap.strip()]
            except Exception:
                caps = []
    if len(caps) == 0:
        return []
    unique = []
    seen = set()
    for cap in caps:
        cap = cap.strip()
        if cap in seen:
            continue
        seen.add(cap)
        unique.append(cap)
    unique.sort(key=lambda v: tuple(int(part) for part in v.split('.') if part.isdigit()))
    flags = []
    for cap in unique:
        major, minor = [int(part) for part in cap.split('.')[:2]]
        cc = str(major) + str(minor)
        sm = 'sm_' + cc
        flags.append('-gencode=arch=compute_' + cc + ',code=' + sm)
        flags.append('-gencode=arch=compute_' + cc + ',code=compute_' + cc)
    if len(flags) > 0:
        return flags
    return []


def compile_command(out):
    if sys.platform == 'win32':
        cl_path = shutil.which('cl')
        if cl_path is not None:
            print("Found MSVC 'cl' compiler. Compiling with cl...")
            return ['cl', '/nologo', '/O2', '/LD', SRC, '/Fe:' + out]
    cc = os.environ.get('CC') or shutil.which('zig') or shutil.which('cc') or shutil.which('gcc') or shutil.which('clang') 
    if cc is None:
        sys.exit('No C compiler found. Install Visual Studio Build Tools, Zig, gcc, or clang.')
    if 'zig' in cc.lower():
        print("MSVC 'cl' not found. Falling back to 'zig cc'...")
        cmd = [cc, 'cc', '-O3', '-shared', SRC, '-o', out, '-lm']
        if sys.platform != 'darwin':
            cmd.insert(2, '-pthread')
        return cmd
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
    arch_flags = cuda_arch_flags()
    base = [nvcc, '-O3', '-shared', '-o', out]
    if sys.platform != 'win32':
        base += ['-Xcompiler', '-fPIC']
    attempts = []
    if len(arch_flags) > 0:
        attempts.append(base + arch_flags + [CU_SRC])
    attempts.append(base + [CU_SRC])
    for i, cmd in enumerate(attempts):
        print('$ ' + ' '.join(cmd))
        if subprocess.run(cmd).returncode == 0:
            break
        if i + 1 < len(attempts):
            print('  GPU arch flags did not work for this CUDA install, '
                  'building for the toolkit default instead')
    else:
        print('CUDA build failed, leaving the add-on on the CPU kernel.')
        return None
    lib = ctypes.CDLL(out)
    print('Built ' + out + ' (device present: '
          + str(lib.flimkit_cu_available() == 1) + ')')
    return out

def install(plugin_dir=None):
    if plugin_dir == None:
        plugin_dir = os.path.expanduser('~') + os.sep + '.flimkit' + os.sep + 'plugins'
    dest = os.path.abspath(os.path.expanduser(plugin_dir)) + os.sep + 'flimkit_accelerator_plugin'
    if os.path.isdir(dest) == True:
        shutil.rmtree(dest)
    shutil.copytree(PKG, dest, ignore=shutil.ignore_patterns('__pycache__'))
    print('Installed to ' + dest)
    print('Tick "Load from ' + os.path.dirname(dest) + '" in File > Preferences > Plugins and restart.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--install', action='store_true')
    ap.add_argument('--plugin-dir', help='plugins folder to install into (implies --install)')
    ap.add_argument('--no-cuda', action='store_true')
    args = ap.parse_args()
    build()
    if args.no_cuda == False:
        build_cuda()
    if args.install == True or args.plugin_dir != None:
        install(args.plugin_dir)
