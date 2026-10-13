import importlib.util
from pathlib import Path
from types import SimpleNamespace

import shutil
import subprocess

module_path = Path(__file__).resolve().parents[1] / 'build.py'
spec = importlib.util.spec_from_file_location('project_build', module_path)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_cuda_arch_flags_support_multiple_visible_gpus(monkeypatch):
    monkeypatch.setattr(shutil, 'which', lambda name: '/usr/bin/nvidia-smi' if name == 'nvidia-smi' else None)

    def fake_run(cmd, capture_output, text, check):
        assert cmd == ['/usr/bin/nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader']
        return SimpleNamespace(stdout='8.6\n7.5\n7.5\n', returncode=0)

    monkeypatch.setattr(subprocess, 'run', fake_run)

    flags = build.cuda_arch_flags()

    assert '-gencode=arch=compute_75,code=sm_75' in flags
    assert '-gencode=arch=compute_86,code=sm_86' in flags
    assert '-gencode=arch=compute_86,code=compute_86' in flags
    assert '-arch=native' not in flags
