#!/usr/bin/env bash
# Run the OpenLane RTL-to-GDS flow on the 8x8 MAC array to get POST-LAYOUT
# area, timing (with wire delays), and power -- the physical-design half of the
# predicted-vs-measured loop that the Yosys+STA numbers only approximate.
#
# Meant for an x86-64 Linux box with Docker: a GitHub Codespace, a cloud VM, or
# a Linux workstation. (It will NOT work on the Apple-silicon Mac -- the tool
# images are x86 and stall under emulation, which is why this is a separate step.)
#
# Usage:   bash rtl/openlane/run.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== 0. staging RTL into the design dir (OpenLane only reads files here) =="
cp -f ../mac_int8.sv ../mac_array_8x8.sv ../mac_array_8x8_pnr.sv ./

echo "== 1. checking Docker =="
if ! docker version >/dev/null 2>&1; then
  echo "!! Docker isn't available. On Codespaces it usually is by default; if not,"
  echo "   add the 'docker-in-docker' feature to .devcontainer and rebuild the Codespace."
  exit 1
fi

echo "== 2. installing the OpenLane 2 orchestrator (needs Python 3.8-3.12) =="
PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
echo "   python3 = $PYV"
python3 -m venv /tmp/olvenv
/tmp/olvenv/bin/pip install --quiet --upgrade pip
if ! /tmp/olvenv/bin/pip install --quiet openlane; then
  echo "!! openlane install failed -- almost always a Python-version wheel issue"
  echo "   (3.13 has no libparse wheel). Install a supported Python and retry, e.g.:"
  echo "     sudo apt-get update && sudo apt-get install -y python3.11-venv"
  echo "     python3.11 -m venv /tmp/olvenv && /tmp/olvenv/bin/pip install openlane"
  exit 1
fi
echo "   $(/tmp/olvenv/bin/openlane --version | head -1)"

echo "== 3. running the flow (first run pulls the tool image + sky130 PDK; be patient) =="
# --skip the antenna-property CHECK: it's a buggy lint in OpenLane 2.3.10
# (StopIteration in get_top_level_cell) that crashes the flow before LVS. Skipping
# it lets the flow reach Magic streamout + DRC + Netgen LVS (real sign-off). The
# antenna-diode INSERTION earlier in the flow is unaffected.
/tmp/olvenv/bin/openlane --dockerized --docker-no-tty \
    --skip Odb.CheckDesignAntennaProperties ./config.json

echo "== 4. results =="
RUN=$(ls -dt runs/*/ 2>/dev/null | head -1)
echo "   run directory: ${RUN:-<none>}"
if [ -n "${RUN:-}" ]; then
  echo "   --- final metrics (area / timing / power) ---"
  find "$RUN" -name "metrics.csv" -o -name "*metrics*.json" 2>/dev/null | head
  echo "   Open the GDS/reports under $RUN; key numbers are in the metrics file and"
  echo "   the final STA (slack) and power reports."
fi
