"""
auto_thievia - A comprehensive GIS toolkit for analyzing auto theft patterns and potential chop shop locations.

This package provides tools for mapping, analyzing, and visualizing auto theft data
with a focus on geographic patterns and spatial relationships.
"""

__version__ = "0.1.0"

from .gis_mapper import GISMapper
from .theft_data import TheftData, TheftDataPersistence
from .suspect_data import SuspectData, SuspectDataPersistence
from .recovery_data import RecoveryData, RecoveryDataPersistence
from .duckdb_persistence import DuckDbPersistence

__all__ = [
    "GISMapper",
    "TheftData",
    "TheftDataPersistence",
    "SuspectData",
    "SuspectDataPersistence",
    "RecoveryData",
    "RecoveryDataPersistence",
    "DuckDbPersistence",
]
