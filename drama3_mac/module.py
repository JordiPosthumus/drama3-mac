"""A minimal parameter container for MLX models.

Parameters are plain ``mx.array`` attributes on ``Module`` instances (lists of
modules flatten with indices).  Runtime caches are kept in ``self._caches`` so
they never show up as parameters.
"""

from __future__ import annotations

import mlx.core as mx


class Module:
    def parameters(self) -> dict[str, mx.array]:
        params: dict[str, mx.array] = {}
        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(value, mx.array):
                params[name] = value
            elif isinstance(value, Module):
                for key, item in value.parameters().items():
                    params[f"{name}.{key}"] = item
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, Module):
                        for key, sub in item.parameters().items():
                            params[f"{name}.{index}.{key}"] = sub
        return params

    def update_parameters(self, flat: dict[str, mx.array]) -> None:
        """Set parameters by the dotted paths produced by :meth:`parameters`."""
        own: dict[str, mx.array] = {}
        nested: dict[str, dict] = {}
        for name, value in flat.items():
            head, _, rest = name.partition(".")
            if rest:
                nested.setdefault(head, {})[rest] = value
            else:
                own[head] = value
        for name, value in own.items():
            if not hasattr(self, name):
                raise KeyError(f"{type(self).__name__} has no parameter {name!r}")
            setattr(self, name, value)
        for name, values in nested.items():
            attr = getattr(self, name, None)
            if isinstance(attr, Module):
                attr.update_parameters(values)
            elif isinstance(attr, list):
                for index, item in enumerate(attr):
                    sub = values.get(str(index))
                    if sub is not None and isinstance(item, Module):
                        item.update_parameters(sub)
            else:
                raise KeyError(f"{type(self).__name__} has no module {name!r}")

    def cache(self, name: str, array: mx.array | None = None) -> mx.array | None:
        """Read/write a runtime cache stored under ``self._caches[name]``."""
        caches = self.__dict__.setdefault("_caches", {})
        if array is None:
            return caches.get(name)
        caches[name] = array
        return array
