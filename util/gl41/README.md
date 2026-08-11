# GL 4.1 core reproducer harness

Minimal standalone test for the remaining macOS core-profile bug: with
`enable_post_processing = true`, terrain does not render on a core context,
while the sky and HUD do. `= false` renders the world correctly.

Build and run (seconds, versus a ~3 minute engine rebuild per guess):

```
clang -DGL_SILENCE_DEPRECATION gl41test.c -o gl41test \
    -I/opt/homebrew/include -L/opt/homebrew/lib -lSDL2 -framework OpenGL
./gl41test
```

## What it has already established

On this machine (`GL_VERSION = 4.1 Metal - 90.5`, GLSL 4.10) **all of the
following work correctly on a core profile** and are therefore ruled out as
platform quirks:

- FBO with a colour texture (`GL_RGB10_A2`) plus a depth texture
  (`GL_DEPTH_COMPONENT24`) — reports `FRAMEBUFFER_COMPLETE`
- Rendering geometry into that FBO with depth test on (`GL_LEQUAL`), depth
  writes on — the triangle lands in the texture (`FBO centre pixel = 255,64,0`)
- Sampling that colour texture onto a fullscreen quad on the default
  framebuffer (`SCREEN centre pixel = 255,64,0`)
- `glBindAttribLocation` + `glBindFragDataLocation` before link, since GLSL
  1.50 has no `layout(location=)` for either attributes or fragment outputs
- Indexed drawing through an element buffer
- Binding and unbinding the FBO repeatedly between draws

So the platform handles exactly what Luanti needs. **The bug is in Luanti's or
Irrlicht's usage, not in macOS core.**

## How to use it

Add one Luanti behaviour at a time until the centre pixel stops matching.
Candidates not yet tried: Irrlicht's cached-GL-state handler diverging from
real state; multiple programs and texture units churning between draws;
`OnResize`/viewport changes mid-pipeline; MSAA resolve.
