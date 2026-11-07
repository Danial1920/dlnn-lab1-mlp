from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
(root / ".cache").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(root / ".cache" / "ipython"))
notebook_path = root / "notebook.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(notebook, timeout=900, km=manager, resources={"metadata": {"path": str(root)}})
client.execute(cleanup_kc=True)
nbformat.validate(notebook)
nbformat.write(notebook, notebook_path)
print("Notebook executed successfully; results/ and figures/ refreshed.")
