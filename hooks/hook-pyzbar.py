## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = collect_dynamic_libs('pyzbar')