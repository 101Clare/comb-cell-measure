"""UI panels, one module per panel. Each exposes a `render(...)` function called from app/streamlit_app.py.

  state.py         session-state helpers, records (load / save), samples + ground truth
  compute.py       cached wrappers around the hexengine pipeline (no maths of its own)
  sidebar.py       uploads, image picker, all settings
  clicking.py      the click-to-mark image widget (coordinate scaling, de-duplication, Undo / Clear / Auto-detect)
  single_view.py   "Single image" tab
  cluster_view.py  "Neighbour cluster" tab
  batch_view.py    "Batch review" tab
  export_panel.py  single-image exports (comparison sheet PNG, JSON record)
  help_panel.py    "Help" tab
"""
