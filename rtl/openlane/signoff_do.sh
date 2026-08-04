#!/usr/bin/env bash
# signoff_do.sh -- one-command OpenLane RTL-to-GDS sign-off on an x86-64 Linux box
# (e.g. a DigitalOcean droplet, cloud VM, or Codespace). Runs the original and/or
# the pipelined MAC-array top at a chosen clock, then prints LVS / DRC / area /
# power / multi-corner timing. This is the interactive-box counterpart of the
# GitHub Actions workflow (.github/workflows/asic-signoff.yml).
#
# Prereqs on a fresh Ubuntu droplet (run once, as root):
#   apt-get update && apt-get install -y docker.io git python3-venv python3-pip
#
# Usage (from the repo root):
#   bash rtl/openlane/signoff_do.sh                     # both tops @ 7.0 ns (143 MHz)
#   bash rtl/openlane/signoff_do.sh 11.0                # both tops @ 11 ns
#   bash rtl/openlane/signoff_do.sh 7.0 config_pipe.json   # pipelined top only
#   bash rtl/openlane/signoff_do.sh 7.0 config.json        # original top only
#
# The FIRST run pulls the OpenLane image + sky130 PDK (a few minutes); later runs
# on the same droplet reuse them and finish much faster. Full GDS + reports land
# under rtl/openlane/runs/<tag>/.  NOTE: this edits CLOCK_PERIOD in the config
# file(s) in place -- fine on a throwaway clone.
set -euo pipefail
cd "$(dirname "$0")"

CLK="${1:-7.0}"; [ $# -gt 0 ] && shift || true
CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(config.json config_pipe.json)
MHZ=$(python3 -c "print('%.1f' % (1000.0/$CLK))")

echo "== 0. prerequisites (docker + openlane orchestrator) =="
command -v docker >/dev/null || { echo "!! docker missing -> apt-get install -y docker.io"; exit 1; }
docker version >/dev/null 2>&1 || { echo "!! docker daemon not reachable (are you root?)"; exit 1; }
if [ ! -x /tmp/olvenv/bin/openlane ]; then
  echo "   installing OpenLane into /tmp/olvenv (first time only) ..."
  python3 -m venv /tmp/olvenv
  /tmp/olvenv/bin/pip -q install --upgrade pip
  /tmp/olvenv/bin/pip -q install openlane
fi
OL=/tmp/olvenv/bin/openlane
echo "   $($OL --version | head -1)"

echo "== 1. stage RTL (configs pick which top is hardened) =="
cp -f ../mac_int8.sv ../mac_array_8x8.sv ../mac_array_8x8_pnr.sv \
      ../mac_int8_pipe.sv ../mac_array_8x8_pipe.sv ../mac_array_8x8_pipe_pnr.sv ./

summarize() {  # $1 = run dir
  local RUN="$1"
  echo "   --- SIGN-OFF SUMMARY ---"
  echo -n "   [LVS]  "
  find "$RUN" -iname '*lvs*' -type f -exec grep -hiE \
    'circuits match|circuits do not match|netlists match|does not match' {} \; 2>/dev/null | sort -u | tr '\n' ' '
  echo
  local M; M=$(find "$RUN" -path '*final*metrics.json' 2>/dev/null | head -1)
  [ -z "$M" ] && M=$(ls -1 "$RUN"*stapostpnr*/state_out.json 2>/dev/null | tail -1)
  [ -z "$M" ] && { echo "   (no metrics file -- flow may not have finished)"; return; }
  python3 - "$M" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d = d.get('metrics', d)
g = lambda k: d.get(k)
print("   [DRC]  route %s  magic %s  klayout %s" %
      (g('route__drc_errors'), g('magic__drc_error__count'), g('klayout__drc_error__count')))
print("   [AREA] cell %.3f mm2  die %.3f mm2  util %.1f%%" % (
      (g('design__instance__area__stdcell') or 0)/1e6,
      (g('design__die__area') or 0)/1e6,
      (g('design__instance__utilization') or 0)*100))
print("   [PWR]  total %.3f W" % (g('power__total') or 0))
print("   [STA]  worst setup %.3f ns   worst hold %.3f ns" %
      (g('timing__setup__ws') or 0, g('timing__hold__ws') or 0))
ss = lambda t: {c.split('corner:')[1]: round(v, 3)
                for c, v in d.items() if t in c and '_ss_' in c}
print("          slow-corner setup:", ss('timing__setup__ws__corner'))
print("          slow-corner hold :", ss('timing__hold__ws__corner'))
PY
}

for cfg in "${CONFIGS[@]}"; do
  [ -f "$cfg" ] || { echo "!! no such config: $cfg"; continue; }
  echo ""
  echo "==================================================================="
  echo "==  $cfg   @   ${CLK} ns   (${MHZ} MHz)"
  echo "==================================================================="
  python3 -c "import json; c=json.load(open('$cfg')); c['CLOCK_PERIOD']=float('$CLK'); json.dump(c,open('$cfg','w'),indent=2); print('   top:', c['DESIGN_NAME'])"
  # script(1) hands OpenLane a pseudo-TTY so 'docker run -t' works even when this
  # script is piped/non-interactive; over an interactive SSH it's a harmless no-op.
  script -q -e -c "$OL --dockerized --skip Odb.CheckDesignAntennaProperties ./$cfg" /dev/null
  summarize "$(ls -dt runs/*/ | head -1)"
done

echo ""
echo "== done. GDS + full reports under rtl/openlane/runs/<tag>/ =="
