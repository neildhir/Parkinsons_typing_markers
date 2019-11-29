import os.path as osp
import socket
import sys
from pathlib import Path


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)
p = this_dir.split("haberrspd")[0]
# Add lib to PYTHONPATH
lib_path = osp.join(p, "haberrspd")
# Add to working space
add_path(lib_path)

# Depending on where I am, set the path
if socket.gethostname() == "pax":
    # Monster machine
    # data_root = "/home/neil/cloud/habitual_errors_NLP/data/"  # My local path
    # data_root = Path(data_root)
    pass
else:
    # Laptop
    # data_root = "/home/nd/data/liverpool/MJFF"  # My local path
    # data_root = Path(data_root)
    pass
