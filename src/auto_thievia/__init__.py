"""
auto_thievia package: Auto theft investigation and analysis tools.

This package provides tools for analyzing auto theft data, suspect information,
recovery data, GIS mapping, database persistence, and machine learning predictions.
"""

from .duckdb_persistence import DuckDbPersistence
from .gis_mapper import GISMapper
from .learning_models import LearningClass
from .recovery_data import RecoveryData, RecoveryDataPersistence
from .suspect_data import SuspectData, SuspectDataPersistence
from .theft_data import TheftData, TheftDataPersistence

__all__ = [
    "TheftData",
    "TheftDataPersistence",
    "SuspectData",
    "SuspectDataPersistence",
    "RecoveryData",
    "RecoveryDataPersistence",
    "GISMapper",
    "DuckDbPersistence",
    "LearningClass",
]

__version__ = "0.1.0"
