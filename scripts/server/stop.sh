#!/bin/bash
# ============================================================================
# Unified Stop Script
# Stops all vLLM and proxy processes regardless of configuration.
# ============================================================================
# Usage:
#   ./stop.sh              # Stop everything
#   ./stop.sh <config>     # Same as above (config argument ignored, kept for compatibility)
#
# Note: Individual stop_<config>.sh scripts are provided for convenience,
#       but they all call cleanup_all.sh internally.
# ============================================================================

SCRIPT_DIR="$(dirname "$0")"
bash "$SCRIPT_DIR/cleanup_all.sh"
