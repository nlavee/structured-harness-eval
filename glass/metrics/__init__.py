import os
import glob
import importlib

# Auto-import all .py files in this directory to populate the registry at glass/metrics/registry.py
# This enables straightforward `import glass.metrics` in the cli.

modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
for f in modules:
    # Skip __init__.py itself
    if os.path.isfile(f) and not f.endswith('__init__.py'):
        # Get module base name
        module_name = os.path.basename(f)[:-3]
        # Dynamically import using the package namespace
        importlib.import_module(f"glass.metrics.{module_name}")
