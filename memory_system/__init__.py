"""Canonical SAGE memory subsystem.

Keep package import side-effect free.  The coordinator and durable job store
refer to each other through the worker, so eager re-exports here would create a
partially-initialised import cycle during application startup.
"""
