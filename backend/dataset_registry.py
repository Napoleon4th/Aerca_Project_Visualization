"""数据集注册表：把项目中的 6 个数据集统一封装。

为避免某个数据集模块（如 lotka_volterra 依赖 numba）的 import 错误拖垮整个后端，
这里采用**懒加载**：每个 dataset 只在第一次被使用时才执行真正的 import。
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

# 让 backend 模块能 import 项目根的 datasets / args / models
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logger = logging.getLogger("aerca.dataset_registry")


# 数据集元信息（不直接 import 任何 dataset 模块）
_DATASET_META: Dict[str, Dict[str, Any]] = {
    "linear": {
        "module": "datasets.linear", "class_name": "Linear",
        "args_module": "args.linear_args",
        "use_slice": True, "supports_adtype": True,
        "adtypes": ["spike", "step", "causal"],
    },
    "lotka_volterra": {
        "module": "datasets.lotka_volterra", "class_name": "LotkaVolterra",
        "args_module": "args.lotka_volterra_args",
        "use_slice": True, "supports_adtype": False,
        "adtypes": ["non_causal"],
    },
    "lorenz96": {
        "module": "datasets.lorenz96", "class_name": "Lorenz96",
        "args_module": "args.lorenz96_args",
        "use_slice": True, "supports_adtype": False,
        "adtypes": ["non_causal"],
    },
    "msds": {
        "module": "datasets.msds", "class_name": "MSDS",
        "args_module": "args.msds_args",
        "use_slice": False, "supports_adtype": False,
        "adtypes": [],
    },
    "swat": {
        "module": "datasets.swat", "class_name": "SWaT",
        "args_module": "args.swat_args",
        "use_slice": False, "supports_adtype": False,
        "adtypes": [],
    },
    "nonlinear": {
        "module": "datasets.nonlinear", "class_name": "Nonlinear",
        "args_module": "args.nonlinear_args",
        "use_slice": True, "supports_adtype": False,
        "adtypes": ["non_causal"],
    },
}


class _LazyDatasetEntry:
    """惰性持有 class 和 args parser。"""

    def __init__(self, name: str, meta: Dict[str, Any]) -> None:
        self.name = name
        self._meta = meta
        self._cls: Optional[type] = None
        self._args_fn: Optional[Callable] = None
        self._import_error: Optional[str] = None

    @property
    def use_slice(self) -> bool:
        return bool(self._meta["use_slice"])

    @property
    def supports_adtype(self) -> bool:
        return bool(self._meta["supports_adtype"])

    @property
    def adtypes(self) -> List[str]:
        return list(self._meta["adtypes"])

    def _ensure_loaded(self) -> None:
        if self._cls is not None and self._args_fn is not None:
            return
        try:
            mod = importlib.import_module(self._meta["module"])
            self._cls = getattr(mod, self._meta["class_name"])
            args_mod = importlib.import_module(self._meta["args_module"])
            self._args_fn = getattr(args_mod, "create_arg_parser")
            self._import_error = None
        except Exception as e:  # noqa: BLE001
            self._import_error = f"{type(e).__name__}: {e}"
            logger.warning("Lazy import failed for %s: %s", self.name, self._import_error)
            raise

    @property
    def cls(self):
        self._ensure_loaded()
        return self._cls

    @property
    def args_fn(self):
        self._ensure_loaded()
        return self._args_fn

    @property
    def import_status(self) -> Dict[str, Any]:
        return {
            "loaded": self._cls is not None,
            "import_error": self._import_error,
        }

    # 兼容旧字典访问
    def __getitem__(self, key: str):
        if key == "use_slice":
            return self.use_slice
        if key == "supports_adtype":
            return self.supports_adtype
        if key == "adtypes":
            return self.adtypes
        if key == "class":
            return self.cls
        if key == "args":
            return self.args_fn
        raise KeyError(key)

    def __contains__(self, key: str):
        return key in {"use_slice", "supports_adtype", "adtypes", "class", "args"}


DATASET_REGISTRY: Dict[str, _LazyDatasetEntry] = {
    name: _LazyDatasetEntry(name, meta) for name, meta in _DATASET_META.items()
}


def list_datasets():
    return [
        {
            "name": e.name,
            "use_slice": e.use_slice,
            "supports_adtype": e.supports_adtype,
            "adtypes": e.adtypes,
            "loaded": e._cls is not None,  # 仅供调试
        }
        for e in DATASET_REGISTRY.values()
    ]


def get_default_options(dataset_name: str) -> Dict[str, Any]:
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    parser = DATASET_REGISTRY[dataset_name].args_fn()
    args = parser.parse_args([])
    return vars(args)
