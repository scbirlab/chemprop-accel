#!/usr/bin/env bash
# Replace the OWNER placeholder with your GitHub user/org name.   usage: scripts/set_owner.sh scbirlab
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 <github-owner>"; exit 2; }
cd "$(dirname "$0")/.."
git grep -lI "OWNER" -- . ':!scripts/set_owner.sh' 2>/dev/null | xargs sed -i.bak "s#OWNER#$1#g" || \
  grep -rlI "OWNER" . --exclude-dir=.git --exclude=set_owner.sh | xargs sed -i.bak "s#OWNER#$1#g"
find . -name '*.bak' -delete
echo "Replaced OWNER with $1"
