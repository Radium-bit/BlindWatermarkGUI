## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

"""
Watermark processing module for BlindWatermarkGUI
Provides embed and extract functionality for watermarks
"""

from .embed import WatermarkEmbedder
from .extract import WatermarkExtractor
from . import dataShielder

__all__ = ['WatermarkEmbedder', 'WatermarkExtractor', 'dataShielder']