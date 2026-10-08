#!/bin/bash
# claude_worktree.sh NAME — a parallel checkout of the engine for one line of
# work (2026-10-08): ~/code/luanti-wt/NAME on branch NAME from one-tracer, with
# the untracked pieces a game needs linked in from the main checkout (the
# game, the test world, the bridge mod, translations) and its own ccache'd
# build. Anything that starts the game takes the GPU lock (util/claude_gpu_lock.py).
set -euo pipefail
NAME="$1"
MAIN="$HOME/code/luanti"
WT="$HOME/code/luanti-wt/$NAME"
mkdir -p "$HOME/code/luanti-wt"
git -C "$MAIN" worktree add "$WT" -b "$NAME" one-tracer
for p in games/mineclonia worlds/gallery mods/claude_bridge locale; do
  rm -rf "$WT/$p"; ln -s "$MAIN/$p" "$WT/$p"
done
cp "$MAIN/minetest.conf" "$WT/minetest.conf"
BT=$(grep "^CMAKE_BUILD_TYPE:" "$MAIN/build/CMakeCache.txt" | cut -d= -f2)
GEN=$(grep "^CMAKE_GENERATOR:" "$MAIN/build/CMakeCache.txt" | cut -d= -f2)
cmake -S "$WT" -B "$WT/build" -G "$GEN" -DCMAKE_BUILD_TYPE="$BT" -DRUN_IN_PLACE=TRUE \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache > "$WT/build-configure.log"
echo "worktree $WT on branch $NAME (build type $BT); build: cmake --build $WT/build -j16"
