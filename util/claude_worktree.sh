#!/bin/bash
# claude_worktree.sh NAME — a parallel checkout of the engine for one line of
# work (2026-10-08): ~/code/luanti-wt/NAME on branch NAME from one-tracer, with
# the untracked pieces a game needs linked in from the main checkout (the
# game, the test world, the bridge mod, translations) and its own ccache'd
# build. Anything that starts the game takes the GPU lock (util/claude_gpu_lock.py).
set -euo pipefail
NAME="${1:?usage: claude_worktree.sh NAME}"
MAIN="$HOME/code/luanti"
WT="$HOME/code/luanti-wt/$NAME"
mkdir -p "$HOME/code/luanti-wt"
git -C "$MAIN" worktree add "$WT" -b "$NAME" one-tracer
# links for the untracked pieces a game needs. A fresh worktree has these
# paths only as git placed them: games/ and worlds/ hold no tracked files
# there, locale does not exist; so nothing is removed, links are only made.
[ -n "$WT" ] && [ -d "$WT" ] || { echo "no worktree at '$WT'"; exit 1; }
for p in games/mineclonia worlds/gallery locale; do
  if [ -e "$WT/$p" ] || [ -L "$WT/$p" ]; then echo "refusing: $WT/$p already exists"; exit 1; fi
  ln -s "$MAIN/$p" "$WT/$p"
done
# the bridge mod has a TRACKED file (init.base.lua): keep the worktree's own
# folder and COPY the untracked pieces, so nothing writes through to main
cp -r "$MAIN/mods/claude_bridge/init.lua" "$MAIN/mods/claude_bridge/mod.conf" \
      "$MAIN/mods/claude_bridge/textures" "$WT/mods/claude_bridge/"
cp "$MAIN/minetest.conf" "$WT/minetest.conf"
BT=$(grep "^CMAKE_BUILD_TYPE:" "$MAIN/build/CMakeCache.txt" | cut -d= -f2)
GEN=$(grep "^CMAKE_GENERATOR:" "$MAIN/build/CMakeCache.txt" | cut -d= -f2)
cmake -S "$WT" -B "$WT/build" -G "$GEN" -DCMAKE_BUILD_TYPE="$BT" -DRUN_IN_PLACE=TRUE \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache > "$WT/build-configure.log"
echo "worktree $WT on branch $NAME (build type $BT); build: cmake --build $WT/build -j16"
