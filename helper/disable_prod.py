#!/usr/bin/env python3
"""
Disable production nodes: deactivate node_prod_* and activate dev nodes.

"Activating" means removing the leading underscore from a disabled file.
"Deactivating" means adding a leading underscore to an active file.

Nodes are detected automatically from the filesystem.
Utility nodes (node_perf.py, node_twin_vis.py) are never touched.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UTILITY_NODES = {"node_perf.py", "node_twin_vis.py"}


def classify_nodes(root):
    """Return (prod, dev) where each is a list of current filenames."""
    prod, dev = [], []
    for fname in sorted(os.listdir(root)):
        if not fname.endswith(".py"):
            continue
        base = fname.lstrip("_")
        if base in UTILITY_NODES:
            continue
        if base.startswith("node_prod_"):
            prod.append(fname)
        elif base.startswith("node_"):
            dev.append(fname)
    return prod, dev


def activate(root, fname):
    """Remove leading underscore — make the node active."""
    if fname.startswith("_"):
        new = fname.lstrip("_")
        os.rename(os.path.join(root, fname), os.path.join(root, new))
        print(f"[ENABLE]  {fname}  →  {new}")
    else:
        print(f"[SKIP]    {fname}  (already active)")


def deactivate(root, fname):
    """Add leading underscore — make the node inactive."""
    if not fname.startswith("_"):
        new = "_" + fname
        os.rename(os.path.join(root, fname), os.path.join(root, new))
        print(f"[DISABLE] {fname}  →  {new}")
    else:
        print(f"[SKIP]    {fname}  (already inactive)")


if __name__ == "__main__":
    prod_nodes, dev_nodes = classify_nodes(ROOT)

    print("── Deactivating production nodes ─────────────────────────────────────")
    for fname in prod_nodes:
        deactivate(ROOT, fname)

    print("── Activating dev nodes ──────────────────────────────────────────────")
    for fname in dev_nodes:
        activate(ROOT, fname)

    print("── Done ──────────────────────────────────────────────────────────────")
    print("Run helper/enable_prod.py to switch back to production nodes.")
