import hashlib
import json
import subprocess
import time
from typing import Any, Dict


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get_lib_versions() -> Dict[str, str]:
    try:
        # Simple pip freeze
        out = subprocess.check_output(["pip", "freeze"], text=True)
        return {line.split("==")[0]: line.split("==")[1] for line in out.splitlines() if "==" in line}
    except Exception:
        return {}


def create_manifest(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    config_json = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    config_hash = hashlib.sha256(config_json).hexdigest()

    return {
        "timestamp": time.time(),
        "git_hash": get_git_hash(),
        "config_hash": config_hash,
        "libraries": get_lib_versions(),
    }
