#!/bin/sh
set -e
cd "$(dirname "$0")"
glslc -O fill.comp -o fill.spv
cc -O2 -Wall -o claude_vk_spike spike.c -lvulkan -lEGL -lOpenGL
echo built
