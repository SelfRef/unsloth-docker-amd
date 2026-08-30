#!/bin/bash
# Entrypoint for the unsloth-amd image.
#
# 1. Keep Studio's persistent state OUT of the install root.
#    UNSLOTH_STUDIO_HOME (/opt/studio) holds both the baked Python venv
#    (unsloth_studio/) and everything Studio persists (auth/, studio.db, cache/,
#    assets/, outputs/, ...). Mounting a volume over the whole root would freeze
#    the venv at the version first started, so instead every state entry is a
#    symlink into $UNSLOTH_STUDIO_DATA (/opt/studio-data) — the one directory
#    worth a volume. The `bin/` and `share/` entries stay in the image (the
#    `unsloth` shim + installer metadata; tools Studio downloads there are
#    regenerable).
#
# 2. Make UNSLOTH_STUDIO_PASSWORD safe to leave in the environment.
#    Studio only ever applies it to the INITIAL admin password and exits with
#    an error if one is already set. Drop the variable when the admin password
#    has already been changed, so a compose `environment:` entry does not break
#    every restart after the first.
set -euo pipefail

STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-/opt/studio}"
DATA="${UNSLOTH_STUDIO_DATA:-/opt/studio-data}"
# Mirrors studio/backend/utils/paths/storage_roots.py (+ llama.cpp from
# unsloth_cli/commands/studio.py). `studio.db` is a file: SQLite resolves the
# symlink itself, so its -wal/-shm sidecars land next to the real file.
STATE_ENTRIES=(auth cache assets outputs exports rag runs llama.cpp studio.db)

mkdir -p "$DATA"
for entry in "${STATE_ENTRIES[@]}"; do
    src="$STUDIO_HOME/$entry"
    dst="$DATA/$entry"
    if [ -L "$src" ]; then
        : # already relinked (image layer)
    elif [ -e "$src" ]; then
        # Real entry left by the build (or a pre-entrypoint run): adopt it into
        # the volume unless the volume already has one — then the volume wins.
        if [ -e "$dst" ]; then
            rm -rf "$src"
        else
            mv "$src" "$dst"
        fi
        ln -s "$dst" "$src"
    else
        ln -s "$dst" "$src"
    fi
    # Directories must exist: Studio's ensure_dir() (mkdir -p) fails on a
    # dangling symlink. studio.db is created by SQLite on first open.
    [ "$entry" = studio.db ] || mkdir -p "$dst"
done

if [ -n "${UNSLOTH_STUDIO_PASSWORD:-}" ] && [ -f "$DATA/auth/auth.db" ]; then
    if python3 - "$DATA/auth/auth.db" <<'PY'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
try:
    row = conn.execute(
        "SELECT must_change_password FROM auth_user WHERE username = 'unsloth'"
    ).fetchone()
except sqlite3.Error:
    row = None
# exit 0 (=> drop the env var) only when an admin exists whose password is set
sys.exit(0 if row is not None and not row[0] else 1)
PY
    then
        echo "unsloth-amd: admin password already set; ignoring UNSLOTH_STUDIO_PASSWORD" >&2
        unset UNSLOTH_STUDIO_PASSWORD
    fi
fi

exec "$@"
