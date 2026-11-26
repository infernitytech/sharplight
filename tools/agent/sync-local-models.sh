#!/usr/bin/env bash
# Mirrors $SL_MODELS into $SL_MODELS_LOCAL for fast test runs: tiny-* entries (and entries matching LOCAL_MODEL_DIRS, a case pattern)
# are copied; everything else is symlinked to the master copy. $SL_MODELS stays the master. Links $SL_TEST_HOME/models to the mirror.
# Safe to re-run.
set -u
. "$(cd "$(dirname "$0")" && pwd)/env.sh"
src=$SL_MODELS; dst=$SL_MODELS_LOCAL
mkdir -p "$dst" "$SL_TEST_HOME"
ln -sfn "$dst" "$SL_TEST_HOME/models"
for top in "$src"/*; do
    name=$(basename "$top")
    if [ ! -d "$top" ]; then
        ln -sfn "$top" "$dst/$name"
        continue
    fi
    mkdir -p "$dst/$name"
    for entry in "$top"/*; do
        e=$(basename "$entry")
        case "$e" in
            tiny-*|${LOCAL_MODEL_DIRS:-__none__})
                [ -L "$dst/$name/$e" ] && rm -f "$dst/$name/$e"
                rsync -a --delete "$entry" "$dst/$name/" ;;
            *)
                [ -e "$dst/$name/$e" ] && [ ! -L "$dst/$name/$e" ] && rm -rf "$dst/$name/$e"
                ln -sfn "$entry" "$dst/$name/$e" ;;
        esac
    done
    # drop mirror entries whose source is gone
    for m in "$dst/$name"/*; do
        [ -e "$top/$(basename "$m")" ] || rm -rf "$m"
    done
done
du -sh "$dst" 2>/dev/null | cut -f1
