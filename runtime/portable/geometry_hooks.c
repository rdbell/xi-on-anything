/* geometry_guest.c's wrappers on this build's skinning functions (meta/builds.json "wraps" and
 * "geometry"). On by default; FFXI_NATIVE_GEOMETRY=0 keeps the translation, as do builds without a
 * verified layout. Each wrapper falls back to the original body it replaces. */
#include "build.h"
#include "geometry_guest.h"
#include "runtime.h"
#include <stdlib.h>
#include <string.h>

#if FFXI_GEOMETRY_LAYOUT == 1
#define ORIGINAL(name)                                                                                                 \
    extern GuestFn rt_wrap_##name;                                                                                     \
    extern const GuestFn rt_orig_##name
ORIGINAL(geometry_parent);
ORIGINAL(geometry_feature_sse);
ORIGINAL(geometry_feature_sse2);
ORIGINAL(geometry_rigid);
ORIGINAL(geometry_weighted);
#undef ORIGINAL
static const GeometryLayout layout = {FFXI_GEOMETRY_INFO, FFXI_GEOMETRY_CALLBACK, FFXI_GEOMETRY_PALETTE,
                                      FFXI_GEOMETRY_COUNTS};
static void parent(Guest* g)
{
    if (!geometry_guest_run(g, &layout, rt_orig_geometry_parent))
        rt_orig_geometry_parent(g);
}
static void feature_sse(Guest* g) { geometry_guest_feature(g, rt_orig_geometry_feature_sse); }
static void feature_sse2(Guest* g) { geometry_guest_feature(g, rt_orig_geometry_feature_sse2); }
static void rigid(Guest* g) { geometry_guest_rigid(g, rt_orig_geometry_rigid); }
static void weighted(Guest* g) { geometry_guest_weighted(g, rt_orig_geometry_weighted); }
#endif

void geometry_hooks_init(void)
{
    const char* enabled = getenv("FFXI_NATIVE_GEOMETRY");
    if (enabled && !strcmp(enabled, "0"))
    {
        rt_log("[recomp] geometry: FFXI_NATIVE_GEOMETRY=0: original translation retained\n");
        return;
    }
#if FFXI_GEOMETRY_LAYOUT == 1
    /* all or nothing, and never over another extension's wrapper */
    if (rt_wrap_geometry_parent || rt_wrap_geometry_feature_sse || rt_wrap_geometry_feature_sse2 ||
        rt_wrap_geometry_rigid || rt_wrap_geometry_weighted)
        return;
    rt_wrap_geometry_feature_sse = feature_sse;
    rt_wrap_geometry_feature_sse2 = feature_sse2;
    rt_wrap_geometry_rigid = rigid;
    rt_wrap_geometry_weighted = weighted;
    rt_wrap_geometry_parent = parent;
    rt_log("[recomp] geometry: SSE-order adapter installed for %s (FFXI_NATIVE_GEOMETRY=0 turns it off); guarded "
           "fallback active\n",
           FFXI_BUILD);
#else
    rt_log("[recomp] geometry: no verified adapter for %s; original translation retained\n", FFXI_BUILD);
#endif
}
