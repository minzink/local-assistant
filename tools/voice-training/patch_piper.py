"""Apply Flay's four changes to a piper1-gpl checkout (idempotent).

1. monotonic_align: use the numba port instead of the Cython module (no compiler needed).
2. __main__: drop the val_mos checkpoint callback (training runs with --model.mos_metric none,
   and Lightning aborts when a monitored metric is never logged).
3. export_onnx: use the classic exporter (torch >= 2.9 defaults to dynamo, which fails).
4. __main__, export_onnx: on Windows, load PosixPath objects as WindowsPath (the Kristin
   checkpoint was pickled on Linux; unpickling it fails with "cannot instantiate 'PosixPath'").

Usage: python patch_piper.py <piper1-gpl dir>
"""

import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
here = Path(__file__).parent

align = root / "src/piper/train/vits/monotonic_align"
shutil.copy(here / "numba_core.py", align / "numba_core.py")
init = align / "__init__.py"
text = init.read_text(encoding="utf-8")
text = text.replace(
    "from .monotonic_align.core import maximum_path_c",
    "from .numba_core import maximum_path_c  # flay: numba instead of Cython",
)
init.write_text(text, encoding="utf-8")

main = root / "src/piper/train/__main__.py"
text = main.read_text(encoding="utf-8")
marker = '        ModelCheckpoint(\n            monitor="val_mos"'
if marker in text:
    start = text.index(marker)
    end = text.index("        ),\n", start) + len("        ),\n")
    note = "        # flay: val_mos callback removed (--model.mos_metric none)\n"
    text = text[:start] + note + text[end:]
    main.write_text(text, encoding="utf-8")

export = root / "src/piper/train/export_onnx.py"
text = export.read_text(encoding="utf-8")
if "dynamo=False" not in text:
    text = text.replace(
        "        verbose=False,\n", "        verbose=False,\n        dynamo=False,  # flay\n", 1
    )
    export.write_text(text, encoding="utf-8")

POSIX_FIX = (
    "import pathlib as _pathlib  # flay: checkpoints pickled on Linux hold PosixPath objects\n"
    "import sys as _sys\n"
    "\n"
    'if _sys.platform == "win32":\n'
    "    _pathlib.PosixPath = _pathlib.WindowsPath\n"
    "\n"
)
for entry in (main, export):
    text = entry.read_text(encoding="utf-8")
    if POSIX_FIX not in text:
        # After a shebang, before the first import.
        head, _, rest = text.partition("\nimport ") if text.startswith("#!") else ("", "", text)
        text = f"{head}\n{POSIX_FIX}import {rest}" if head else POSIX_FIX + text
        entry.write_text(text, encoding="utf-8")

checks = {
    "numba": "numba_core" in init.read_text(encoding="utf-8"),
    "val_mos": 'monitor="val_mos"' not in main.read_text(encoding="utf-8"),
    "dynamo": "dynamo=False" in export.read_text(encoding="utf-8"),
    "posix_path": all(POSIX_FIX in f.read_text(encoding="utf-8") for f in (main, export)),
}
print("Patches:", checks)
sys.exit(0 if all(checks.values()) else 1)
