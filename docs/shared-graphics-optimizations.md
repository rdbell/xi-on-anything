# Shared graphics work from the Android port

CPU skinning kernels that keep the game's own SSE arithmetic, and guarded adapters that let the
recompiled client use them in place of the translated x87 code. They are on by default on the
verified `2025-11-12` build; `FFXI_NATIVE_GEOMETRY=0` keeps the translation. The work came from the
[Android port](android.md); the last sections cover what of that port carries over to the desktop
renderers, which are unchanged.

## Geometry kernels

`runtime/portable/geometry_simd.c` holds the game's rigid and weighted position/normal
transforms, apart from any guest-memory adapter, as ARM64 NEON and x86 SSE2 code behind one plain
C interface. It depends on no graphics API, so the Metal, D3D12 and Vulkan hosts can all use it.

The kernels keep the original SSE sequence of binary32 operations, with FMA and reassociation
off. When a result is a NaN they redo the vertex one operation at a time, to keep SSE's NaN
selection, quieting and negative indefinite. This is not the arithmetic of the translated scalar
x87 path, which keeps more intermediate precision, but it is the retail client's: the game picks
its skinning path from CPUID, so on any PC with SSE it runs the SSE branch. The recompiled client
reports no SSE (`rt_cpuid`, `IsProcessorFeaturePresent`), which is why the translation takes the
x87 branch. Vertices differ from the x87 path's by a few units in the last place.

Inputs and outputs are unaligned byte spans, not guest pointers. Each kernel reads every input
before writing either 12-byte XYZ result, so outputs may alias matrices, source vertices or each
other. The caller must own those spans for the whole call. `geometry_simd_supported()` checks the
host floating-point controls once per batch without changing them; if they are unsupported, the
caller runs the original path. The arithmetic may set floating-point status flags.

The kernels are linked into the portable game hosts. They install the adapters, unless
`FFXI_NATIVE_GEOMETRY=0`, only for a build whose `meta/builds.json` entry has a verified `geometry`
layout; other builds keep the translation. No global CPUID or feature-byte change is made.
Generated game C, captures and game files are not in the repository.

### Tests

These need no game installation or graphics device:

```sh
python3 tests/geometry_simd_test.py
CFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' python3 tests/geometry_simd_test.py
CC='clang -arch x86_64' python3 tests/geometry_simd_test.py  # macOS with Rosetta
```

From a Windows x64 Native Tools prompt:

```bat
set CC=cl
python tests/geometry_simd_test.py
```

The test builds a SIMD and a forced-scalar variant in a temporary folder. Each is checked against
a separate per-lane reference: on x86 with GCC or Clang the reference runs scalar SSE instructions
with explicit operand order; elsewhere it rounds double intermediates to binary32 and handles
special values explicitly. Both variants must produce the same digest of the complete buffer.

Cases cover rigid and weighted transforms, all 16 input alignments, partial and exact aliases,
output canaries, signed zeros, subnormals, overflow, infinities, quiet and signalling NaNs,
FMA/reassociation counterexamples, and host FP control admission. Controls the CPU or emulator
ignores are counted separately; Rosetta, for example, may ignore attempts to unmask SSE
exceptions. These synthetic checks say nothing about whole-game fidelity or performance.

CI (`.github/workflows/geometry-simd.yml`) runs them with macOS Clang, Linux GCC and Clang, and
Windows MSVC. A cross-compiled Windows executable alone does not count as Windows runtime
validation.

## Guarded game adapters

`runtime/portable/geometry_hooks.c` uses the recompiler's per-build function wrappers. The
translated parent keeps its own control flow, and the original leaf bodies stay in place. The
parent admits a batch only while the current thread holds the guest lock, and keeps it across the
checks and at most 4,096 weighted work units. Before changing any guest state it checks mapped
guest pages, address overflow, stack bounds, aliases, counts, remap indices, the callback and
feature modes, the guest FCW and the host FP controls. A rejected batch runs the original
translation. The log names each reason the first time a batch is rejected for it, and again at
every power of two, with the count admitted so far, e.g. `geometry: 4 batches left to the
translation (work cap), 91233 admitted`. The reasons are game or host state, no work, layout,
work cap, unmapped page and alias.

Only inside an admitted parent, on the same thread and Guest, do the feature getters report SSE
and the two leaves call the kernels. Every other call runs the original body, and an extension's
existing wrappers are never replaced. Both feature bytes stay unchanged in guest memory. The
POSIX, MSVC, MinGW and Linux kit source lists include the adapter; the hooks that depend on build
metadata compile per game build.

The guard and production-hook tests also need no game files:

```sh
python3 tests/geometry_guest_test.py
CFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' python3 tests/geometry_guest_test.py
```

`tests/geometry_replay_test.py` compares the adapter with the game's own code. It needs your own
prepared build with geometry addresses in `meta/builds.json`, `generated/all`, `pefile` and
`unicorn`. It first checks that the code names each global (`info` in both feature getters, the rest
in the parent): the adapter reads them only for its guards, so a wrong one could still give matching
output. It copies the translated functions, unchanged, into a temporary folder, and runs the
original x86/SSE parent separately in Unicorn. Nothing from the game is published.

```sh
python3 tests/geometry_replay_test.py
```

On macOS ARM64, 480 animated parent calls (120 frames each for single and dual streams, with and
without bone remapping) match the original SSE code. The comparison covers every page the fixture
maps (the mesh's, and those from the first global to the last), the general registers and the
modeled flags, and the guest's x87 registers and control state are unchanged. Every data access by
the original code must fall in those pages. The oracle supplies the boundary return address and the
SSE feature flag for the run and restores both before comparing memory.

The game-free adapter tests (58 admission cases, and the hook tests with and without a verified
layout) pass on ARM64 macOS, Rosetta x86 and the Pixel Fold, and pass under the sanitizers on
macOS. They check the geometry contract only; play is covered under Desktop
measurements.

## Desktop measurements

On Linux x86-64 (Bazzite 44, Ryzen AI Max+ 395 with Radeon 8060S, the desktop Vulkan
renderer), with a `2025-11-12` HorizonXI install and the committed replay scenes:

- `tests/geometry_replay_test.py` passes on x86-64: 480 animated parent calls, 76,677,120 bytes
  of mapped pages, no mismatches with the original SSE code in Unicorn.
- Frame captures 2, 6 and 10 s into seven scenes (city, crowds, spells, dawn and noon), with the
  adapter off, on and off again, show no defect: the same characters, poses and lighting. Pixels
  that differ are character outlines and small pose offsets, the kind two runs without the adapter
  also show; where the adapter's differ more (one noon frame, the player crowd), it is outlines and
  poses, as from SSE arithmetic or a slightly different moment, not misplaced or missing geometry.
- Frame rate, adapter off and on in alternating runs on a shared machine whose load varied: with
  the standard Modern FX settings, no difference shows, since replaying sun-shadow casters takes
  most of each frame. With `fx = 0`, where the game's own code is most of the frame, the crowd and
  spell scenes ranged from no change to 50% faster with the adapter, in step with the machine's
  load more than with the adapter; in the round with matched load, player crowd was unchanged and
  mob spells 16% faster. Scenes at the 60 FPS cap (city, dawn, wind) were unchanged.

So on this desktop the adapter matches the game's SSE code and draws the same scenes, with no
regression. These frame-rate runs could not show a gain: with Modern FX the frame was bound by the
shadow replay, the cities were at the 60 FPS cap, and the machine's load varied between runs. On a
ROG Ally (Ryzen Z1), where the long player-crowd scene with `fx = 0` is bound by the game's own
code, the adapter took it from 77 to 87 FPS in each of two sessions, and skinning from 3.65 ms to
2.29 ms a frame. macOS and Windows frame rates are not measured.

## Android build

See [Android build and controls](android.md). It includes the Android SDL host and APK tooling and
a native C++ Vulkan renderer with a BCn fallback, keyed asynchronous visibility, launch-time
policy controls and a command/descriptor cache. The dependency pass planner, preserved-mask
visibility storage and render worker are included but off, with their tests and the measurements
that argued against them. None of it replaces `gfx_vulkan.c`, Metal or D3D12 on desktop.

Offscreen tests on the Fold pass for formats, render state, keyed histories and lifetimes,
bounded areas, pass planning and encoding. The preserved-mask fixture passes 44,093,573 checks
with no mismatches in each of three runs with it enabled; the keyed readback and worker-mailbox
tests also pass three runs. These fixtures exercise changing GPU contents and ownership; they do
not show that delayed visibility matches in the game.

Desktop frame rates have not been measured for the Android code; for the geometry adapter, see
"Desktop measurements" above.

## Porting to the desktop renderers

| Work | Porting decision |
| --- | --- |
| Keyed asynchronous visibility | Android only. Desktop answers the occlusion probe without a GPU read (below), so there is nothing for it to save. |
| Command/descriptor encoding | Not the Android cache: each desktop renderer has its own binding policy (below). Measure reuse per renderer, with the sun-shadow work. |
| Pass/load-store reduction | Keep experimental. The Android preserved-mask rewrite showed no clear gain on its own, and needs correct invalidation, attachment contents and Modern FX interaction. |
| Launch-time controls | Read new launch-only controls once, outside hot loops. On Android the cache only won back overhead added by local experimental options; it is not a desktop gain. |
| Render worker | Shelve the current FIFO design. It was slower in the Android comparison; a separate thread alone does not give useful overlap. |

Visibility on desktop is decided in the shared Direct3D 8 front end (`d3d8.c`, `lock_rect`), so
Vulkan, D3D12 and Metal all get the same policy. The game's 16x16 occlusion probe reads fully
visible with no GPU read (`FFXI_PROBE=gpu` reads it), and the sun's sky probe reads the newest
completed copy, a frame or more late, through `gfx_tex_read_async`, which all three renderers
implement. Since nothing reads a character's probe then, its draws are dropped too (`d3d8_probe_test`,
wrapping the game's probe on the 2025-11-12 build): two screen-space quads of up to 3,000x3,000 pixels each into
a 4096x4096 target, up to 12 a frame. On a Radeon 8060S that took mob spells with effects off from 117 to 151
FPS, and crowds with default effects 10-16% higher. So there is no synchronous wait on desktop for keyed visibility to remove: the Android
speedup came from replacing the Android baseline's exact reads (`FFXI_PROBE=gpu`) and does not
carry over. Keyed visibility stays Android-only until a desktop renderer reads probes exactly.

Each desktop renderer already has its own binding policy: Vulkan pushes descriptors per draw
(`vkCmdPushDescriptorSetKHR`), D3D12 binds root descriptor tables, and Metal skips unchanged
vertex buffers (`g_bound`). With Modern FX at the standard settings, `FFXI_PROFILE` on Linux puts
most of the Vulkan renderer's encode time in replaying sun-shadow casters (24 to 83 ms a frame in
Markets, 7 to 10 ms with `sun_casters=1`), so binding reuse there belongs with that work, measured
per renderer, not with the Android cache.

## Pixel Fold measurements

An earlier 29-scene comparison on the Pixel Fold measured Markets at 24.82 to 60.16 FPS and Mines
at 22.96 to 57.72 FPS; heavy crowds stayed below 30 FPS and seven lighting or weather scenes got
slower. That build changed the Vulkan backend, geometry and visibility policy together, so it
shows nothing about these kernels alone and gives no desktop estimate.

The first geometry-only Android run, off/on/off, measured 15.033 / 16.831 / 15.677 FPS, with one
candidate and unequal thermal and clock conditions. Thirty selected captured inputs matched the
original SSE output, but neither result validates the portable implementation in a desktop game.
The parent oracle adds synthetic motion and selected real inputs. For Linux, see "Desktop
measurements" above; game FPS on macOS and Windows, and the Android renderer's fidelity in live
motion, are still open.
