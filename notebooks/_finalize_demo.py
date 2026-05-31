"""Post-execution step: append the live ipywidgets explorer cell (left
UNEXECUTED so no widget-state metadata is written, which keeps GitHub's static
renderer happy) immediately after the static-snapshot cell, and defensively
strip any top-level ``widgets`` metadata.

Run after `jupyter nbconvert --execute --inplace notebooks/demo.ipynb`.
"""
import nbformat as nbf
from nbformat.v4 import new_code_cell

PATH = "notebooks/demo.ipynb"

EXPLORER_SRC = '''# Live controls — runs in Colab / Jupyter (Run All). On GitHub this stays
# static; the snapshot above is the default configuration.
import ipywidgets as widgets
from ipywidgets import interact

fluid_opts = ["Propane", "Isobutane", "n-Butane", "Isopentane",
              "n-Pentane", "Cyclopentane", "R245fa"]

interact(
    plot_design,
    fluid=widgets.Dropdown(options=fluid_opts, value="Isobutane",
                           description="fluid"),
    T_brine=widgets.FloatSlider(min=110, max=200, step=5, value=150,
                                description="brine °C", continuous_update=False),
    T_cond=widgets.FloatSlider(min=20, max=45, step=1, value=30,
                               description="cond °C", continuous_update=False),
    pinch=widgets.FloatSlider(min=2, max=12, step=0.5, value=5,
                              description="pinch K", continuous_update=False),
);'''

nb = nbf.read(PATH, as_version=4)

# Find the static-snapshot cell.
idx = None
for i, c in enumerate(nb.cells):
    if c.cell_type == "code" and "# Static default configuration" in c.source:
        idx = i
        break
if idx is None:
    raise SystemExit("could not locate the snapshot cell")

explorer = new_code_cell(EXPLORER_SRC)
explorer.outputs = []
explorer.execution_count = None
nb.cells.insert(idx + 1, explorer)

# Defensive: GitHub's renderer chokes on malformed widget state.
nb.metadata.pop("widgets", None)

nbf.write(nb, PATH)
print(f"inserted live explorer cell at index {idx + 1}; total cells now {len(nb.cells)}")
