"""Test support helpers for dependency-light imports."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path


def _install_dotenv_stub() -> None:
    if "dotenv" in sys.modules:
        return

    module = types.ModuleType("dotenv")

    def load_dotenv() -> bool:
        return True

    module.load_dotenv = load_dotenv
    sys.modules["dotenv"] = module


def _install_fastmcp_stub() -> None:
    if "fastmcp" in sys.modules:
        return

    module = types.ModuleType("fastmcp")

    class FastMCP:
        def __init__(self, name: str):
            self.name = name

        def tool(self):
            def decorator(func):
                setattr(func, "_is_mcp_tool", True)
                return func

            return decorator

        def run(self, transport: str = "stdio"):
            return None

    module.FastMCP = FastMCP
    sys.modules["fastmcp"] = module


def _install_httpx_stub() -> None:
    if "httpx" in sys.modules:
        return

    module = types.ModuleType("httpx")

    class RequestError(Exception):
        pass

    class TimeoutException(Exception):
        pass

    class AsyncClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            raise RuntimeError("httpx stub get() called unexpectedly")

    module.RequestError = RequestError
    module.TimeoutException = TimeoutException
    module.AsyncClient = AsyncClient
    sys.modules["httpx"] = module


def _install_pydantic_stub() -> None:
    if "pydantic" in sys.modules:
        return

    module = types.ModuleType("pydantic")
    missing = object()

    class _FieldInfo:
        def __init__(self, *, default=missing, default_factory=None):
            self.default = default
            self.default_factory = default_factory

    def Field(default=missing, *, default_factory=None):
        return _FieldInfo(default=default, default_factory=default_factory)

    def ConfigDict(**kwargs):
        return dict(kwargs)

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, payload):
            if not isinstance(payload, dict):
                raise TypeError("model_validate expects a dict payload")

            values = {}
            annotations = getattr(cls, "__annotations__", {})
            for name in annotations:
                if name in payload:
                    values[name] = payload[name]
                    continue

                default_value = getattr(cls, name, missing)
                if isinstance(default_value, _FieldInfo):
                    if default_value.default_factory is not None:
                        values[name] = default_value.default_factory()
                        continue
                    if default_value.default is not missing:
                        values[name] = default_value.default
                        continue
                    raise ValueError(f"Missing required field: {name}")

                if default_value is not missing:
                    values[name] = default_value
                    continue

                raise ValueError(f"Missing required field: {name}")

            for key, value in payload.items():
                if key not in values:
                    values[key] = value
            return cls(**values)

        def model_dump(self, mode="python"):
            _ = mode
            return dict(self.__dict__)

    module.BaseModel = BaseModel
    module.ConfigDict = ConfigDict
    module.Field = Field
    sys.modules["pydantic"] = module


def import_penrs_server(force_reload: bool = False):
    _install_dotenv_stub()
    _install_fastmcp_stub()
    _install_httpx_stub()

    if force_reload:
        for module_name in ("penrs_mcp_server", "utils"):
            sys.modules.pop(module_name, None)

    if "penrs_mcp_server" in sys.modules:
        return sys.modules["penrs_mcp_server"]
    return importlib.import_module("penrs_mcp_server")


def import_worker_nodes(force_reload: bool = False):
    module_name = "worker_nodes_notebook"
    if force_reload:
        sys.modules.pop(module_name, None)

    if module_name in sys.modules:
        return sys.modules[module_name]

    notebook_path = Path(__file__).resolve().parent.parent / "worker_nodes.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    module = types.ModuleType(module_name)
    module.__file__ = str(notebook_path)

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        exec(compile(source, str(notebook_path), "exec"), module.__dict__)

    sys.modules[module_name] = module
    return module


def import_orchestrator(force_reload: bool = False):
    _install_pydantic_stub()

    module_name = "orchestrator_notebook"
    if force_reload:
        sys.modules.pop(module_name, None)

    if module_name in sys.modules:
        return sys.modules[module_name]

    notebook_path = Path(__file__).resolve().parent.parent / "orchestrator.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    module = types.ModuleType(module_name)
    module.__file__ = str(notebook_path)

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        exec(compile(source, str(notebook_path), "exec"), module.__dict__)

    sys.modules[module_name] = module
    return module
