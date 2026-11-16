"""kgcp installs with PyTorch alone: no backend's KGE library is needed to import or use it."""

import subprocess
import sys
from importlib.metadata import requires
from pathlib import Path

import kgcp

BACKEND_LIBRARIES = {"kge", "unkge", "pykeen"}


def test_only_dependency_is_torch():
    """Extras may add backends; a plain install needs nothing but PyTorch."""
    required = [r for r in requires("kgcp") if "extra ==" not in r]
    assert required == ["torch>=2.0"]


def test_import_loads_no_backend_library():
    """In a new interpreter, as the libraries may be imported by other tests here."""
    code = f"import sys, kgcp; sys.exit(len({BACKEND_LIBRARIES!r} & set(sys.modules)))"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_typed():
    assert (Path(kgcp.__file__).parent / "py.typed").exists()
