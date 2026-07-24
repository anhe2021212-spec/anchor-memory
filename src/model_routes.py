"""
模型路由 - 统一管理模型名映射
支持单模型和fallback列表。
改名字只改 model_routes.json，重启gateway即可。
"""
import json, os
from release_config import AnchorConfig

_CONFIG = AnchorConfig.load()
_ROUTES_PATH = os.environ.get("ANCHOR_MODEL_ROUTES_FILE", str(_CONFIG.data_dir / "model_routes.json"))
_routes = None

def _load():
    global _routes
    if not os.path.isfile(_ROUTES_PATH):
        _routes = {}
        return
    with open(_ROUTES_PATH, "r", encoding="utf-8") as f:
        _routes = json.load(f)

def get_model(logical_name: str) -> str:
    """逻辑名 → 第一优先模型名（向后兼容）。"""
    if _routes is None:
        _load()
    val = _routes.get(logical_name, logical_name)
    if isinstance(val, list):
        return val[0] if val else logical_name
    return val

def get_model_chain(logical_name: str) -> list:
    """逻辑名 → fallback列表。单模型也包成列表。"""
    if _routes is None:
        _load()
    val = _routes.get(logical_name, logical_name)
    if isinstance(val, list):
        return list(val)
    return [val]

def reload():
    """热重载路由表"""
    global _routes
    _routes = None
    _load()

def all_routes() -> dict:
    if _routes is None:
        _load()
    return dict(_routes)
