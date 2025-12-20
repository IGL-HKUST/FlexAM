"""FlexAM package initialization - sets up Python paths for submodules"""
import os
import sys

# Get project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Add all submodule paths
submodule_paths = [
    "submodules/MoGe",
    "submodules/vggt",
    "submodules/DELTA",
    "submodules/DELTA/densetrack3d",
    "submodules/Pi3",
]

for path in submodule_paths:
    full_path = os.path.join(project_root, path)
    if full_path not in sys.path:
        sys.path.insert(0, full_path)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
