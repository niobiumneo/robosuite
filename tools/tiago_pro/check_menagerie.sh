#!/usr/bin/env bash
# Replicate MuJoCo Menagerie's CI checks for the generated pal_tiago_pro package, on a TEMPORARY copy.
# The menagerie clone is never modified (its `git status` is verified before and after).
#
#   tools/tiago_pro/check_menagerie.sh                # uses $MENAGERIE (default ~/google-deepmind/mujoco_menagerie)
#   PYTHON=/path/to/python MENAGERIE=/path/to/clone WORK=/tmp/mm tools/tiago_pro/check_menagerie.sh
#
# Needs: python with mujoco, absl-py, pytest, lxml.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PKG_SRC="$REPO/robosuite/models/assets/robots/tiago_pro/pal_tiago_pro"
MENAGERIE="${MENAGERIE:-/home/user/google-deepmind/mujoco_menagerie}"
PYTHON="${PYTHON:-python}"
WORK="${WORK:-$(mktemp -d -t menagerie_check_XXXXXX)}"
MM="$WORK/mm"       # minimal copy: tests + helper scripts + the package (+ pal_tiago_dual as reference)
LIC="$WORK/lic"     # license-only copy of every model dir (for regenerate_license.py)
fail=0

step() { printf '\n== %s\n' "$*"; }
ok()   { printf '   PASS: %s\n' "$*"; }
bad()  { printf '   FAIL: %s\n' "$*"; fail=1; }

before="$(git -C "$MENAGERIE" status --porcelain)"
rm -rf "$MM" "$LIC"; mkdir -p "$MM" "$LIC"

step "copy (clone untouched): helper scripts, test/, package -> $MM"
for f in format_xml.py catalog.py build_registry.py regenerate_license.py pyproject.toml LICENSE CONTRIBUTORS.md README.md; do
  cp "$MENAGERIE/$f" "$MM/"
done
cp -r "$MENAGERIE/test" "$MM/test"
cp -r "$MENAGERIE/pal_tiago_dual" "$MM/pal_tiago_dual"   # reference model (license md5, conventions)
cp -r "$PKG_SRC" "$MM/pal_tiago_pro"

step "register the model in the COPY of catalog.py (MODEL_MAP + PREVIEW_OVERRIDES)"
"$PYTHON" - "$MM/catalog.py" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = s.replace("  'pal_tiago_dual/tiago_dual': ModelType.MOBILE_MANIPULATOR,\n",
              "  'pal_tiago_dual/tiago_dual': ModelType.MOBILE_MANIPULATOR,\n"
              "  'pal_tiago_pro/tiago_pro': ModelType.MOBILE_MANIPULATOR,\n", 1)
s = s.replace("  'pal_tiago_dual/tiago_dual': 'pal_tiago_dual/scene_position.xml',\n",
              "  'pal_tiago_dual/tiago_dual': 'pal_tiago_dual/scene_position.xml',\n"
              "  'pal_tiago_pro/tiago_pro': 'pal_tiago_pro/scene_position.xml',\n", 1)
assert s.count("pal_tiago_pro/tiago_pro") == 2, "catalog.py layout changed"
open(p, "w").write(s)
PY
ok "catalog entries added to the copy"

step "package layout checks"
head -n1 "$MM/pal_tiago_pro/README.md" | grep -qx '## TIAGo Pro Description (MJCF)' && ok "README first line" || bad "README first line"
[ "$(md5sum < "$MM/pal_tiago_pro/LICENSE")" = "$(md5sum < "$MENAGERIE/pal_tiago_dual/LICENSE")" ] \
  && ok "LICENSE md5 equals pal_tiago_dual/LICENSE" || bad "LICENSE differs from pal_tiago_dual/LICENSE"
for f in README.md CHANGELOG.md LICENSE; do [ -f "$MM/pal_tiago_pro/$f" ] && ok "$f present" || bad "$f missing"; done
n=$( (grep -h '\.\./' "$MM"/pal_tiago_pro/*.xml || true) | wc -l)
[ "$n" = 0 ] && ok "no '../' in package XML" || bad "$n occurrences of '../'"
sz=$(du -sb "$MM/pal_tiago_pro" | cut -f1); mx=$(find "$MM/pal_tiago_pro" -type f -printf '%s\n' | sort -n | tail -1)
echo "   package size $sz bytes, largest file $mx bytes"
[ "$sz" -le 8000000 ] && [ "$mx" -le 1000000 ] && ok "size budget (<= 8 MB, no file > 1 MB)" || bad "size budget"
"$PYTHON" - "$MM/pal_tiago_pro" <<'PY' && ok "every referenced mesh exists, no unreferenced mesh" || bad "mesh reference check"
import os, sys
from lxml import etree
d = sys.argv[1]
refs = set()
for x in os.listdir(d):
    if x.endswith(".xml"):
        for m in etree.parse(os.path.join(d, x)).iter("mesh"):
            if m.get("file"):
                refs.add(m.get("file"))
have = {os.path.relpath(os.path.join(r, f), os.path.join(d, "assets")) for r, _, fs in os.walk(os.path.join(d, "assets")) for f in fs}
missing, extra = refs - have, have - refs
print("   referenced %d, on disk %d, missing %s, unreferenced %s" % (len(refs), len(have), sorted(missing), sorted(extra)))
sys.exit(1 if missing or extra else 0)
PY

step "format_xml.py --check (menagerie style)"
( cd "$MM" && "$PYTHON" format_xml.py --check pal_tiago_pro/*.xml ) && ok "all package XML formatted" || bad "format_xml.py --check"

step "pytest test/model_dir_test.py test/model_test.py -k 'pal_tiago_pro or Contributors'"
( cd "$MM" && "$PYTHON" -m pytest -q -p no:cacheprovider test/model_dir_test.py test/model_test.py \
    -k "pal_tiago_pro or Contributors" ) && ok "pytest" || bad "pytest"

step "regenerate_license.py --check (license-only copy of every model dir + the package)"
for d in "$MENAGERIE"/*/; do
  n="$(basename "$d")"; [ -f "$d/LICENSE" ] && { mkdir -p "$LIC/$n"; cp "$d/LICENSE" "$LIC/$n/LICENSE"; }
done
cp "$MENAGERIE/LICENSE" "$MENAGERIE/regenerate_license.py" "$LIC/"
( cd "$LIC" && "$PYTHON" regenerate_license.py --check ) && ok "baseline top-level LICENSE up to date" || bad "baseline LICENSE check (clone itself is stale)"
mkdir -p "$LIC/pal_tiago_pro"; cp "$PKG_SRC/LICENSE" "$LIC/pal_tiago_pro/LICENSE"
( cd "$LIC" && "$PYTHON" regenerate_license.py && "$PYTHON" regenerate_license.py --check ) \
  && ok "LICENSE regenerated with pal_tiago_pro on the copy and --check OK" || bad "regenerate_license.py"

step "menagerie clone untouched"
after="$(git -C "$MENAGERIE" status --porcelain)"
[ "$before" = "$after" ] && ok "git status --porcelain unchanged ('${after:-clean}')" || bad "clone changed!"

echo
[ "$fail" = 0 ] && echo "ALL MENAGERIE CHECKS PASSED (work dir: $WORK)" || { echo "SOME CHECKS FAILED (work dir: $WORK)"; exit 1; }
