"""Keyword differences between versions of streamlit-image-coordinates."""

from importlib.metadata import version

# 0.4 replaced use_column_width="always" with width="stretch"
_NEW = tuple(int(p) for p in version("streamlit-image-coordinates").split(".")[:2]) >= (0, 4)
FILL = {"width": "stretch"} if _NEW else {"use_column_width": "always"}
