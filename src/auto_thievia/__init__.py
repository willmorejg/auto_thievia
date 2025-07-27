"""auto_thievia - A Python package using GIS to determine auto theft patterns."""

__version__ = "0.1.0"
__author__ = "James G Willmore"
__email__ = "willmorejg@gmail.com"

# Main package imports
from .gis_mapper import GISMapper

# from .theft_analyzer import TheftAnalyzer
# from .chop_shop_predictor import ChopShopPredictor

# Make key classes available at package level
__all__ = [
    "GISMapper",
]
