/* Independent headless admission of Android-only keyed, bounded GPU readback.
 * Exercises actual gfx API and GPU pixels; no game/phone/timing claim. Delayed
 * output remains a temporal rendering tradeoff even when these checks pass.
 * The original ordinal-only negative fixture is archived in Android evidence.
 */
#include "gfx.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(FFXI_ANDROID_VULKAN)
int main(void)
{
    puts("{\"event\":\"async-policy\",\"result\":\"UNSUPPORTED\",\"reason\":\"Android keyed API required; no ordinal fallback\",\"game_or_FPS_claim\":false}");
    return 77;
}
#else
static unsigned tests, checks, pixels, errors, shown, frame, delayed_reads, exact_reads;

static uint64_t actor_key(unsigned actor)
{
    return UINT64_C(0x1006c72f00000000) | (UINT64_C(0x01800000) + actor * 16u);
}
static uint32_t code(unsigned family, unsigned variant)
{
    return UINT32_C(0xa5000000) | ((family & 255u) << 16) |
        ((variant & 255u) << 8) | (frame & 255u);
}
static GfxTex* texture(int type, unsigned levels)
{
    GfxTex* t = gfx_tex_create(type, 21, 16, 16, levels, GFX_USE_RT);
    if (!t) { fputs("async target allocation failed\n", stderr); exit(2); }
    return t;
}
static void fill(GfxTex* t, unsigned face, unsigned level, uint32_t color)
{
    unsigned n = 16u >> level;
    uint32_t vp[] = {0, 0, n, n, 0, UINT32_C(0x3f800000)};
    gfx_set_targets(t, face, level, NULL);
    gfx_clear(0, NULL, 1, color, 1, 0, vp);
}
static void advance(void)
{
    gfx_present(NULL);
    gfx_finish();
    ++frame;
}
/* Every read checks row padding and both allocation guards, including cold and
 * sync fallbacks. family policy permits only this key's historical color family
 * at a nonnegative issuing-frame age <= maxAge. exact requires current bytes.
 */
static void read_check(const char* name, GfxTex* t, unsigned face, unsigned level,
    uint64_t key, uint32_t max_age, uint32_t expected, int exact)
{
    unsigned n = 16u >> level, pitch = n * 4u + 12u;
    unsigned body = pitch * n, before = errors, max_seen = 0;
    uint8_t bytes[16 + (16 * 4 + 12) * 16 + 16];
    memset(bytes, 0xcd, sizeof bytes);
    gfx_tex_read_async_keyed(t, face, level, bytes + 16, pitch, key, max_age);
    for (unsigned y = 0; y < n; ++y) {
        for (unsigned x = 0; x < n; ++x) {
            uint32_t got;
            memcpy(&got, bytes + 16 + y * pitch + x * 4, 4);
            unsigned age = (frame - (got & 255u)) & 255u;
            if (age > max_seen) max_seen = age;
            int good = exact ? got == expected :
                ((got & UINT32_C(0xffff0000)) == (expected & UINT32_C(0xffff0000)) && age <= max_age);
            ++checks;
            if (!good) {
                ++errors;
                if (shown++ < 16) fprintf(stderr,
                    "%s frame%u pixel%u,%u got%08x expected%s%08x age%u max%u\n",
                    name, frame, x, y, got, exact ? " " : " family ", expected, age, max_age);
            }
        }
        for (unsigned i = n * 4; i < pitch; ++i) {
            ++checks;
            if (bytes[16 + y * pitch + i] != 0xcd) ++errors;
        }
    }
    for (unsigned i = 0; i < 16; ++i) {
        checks += 2;
        if (bytes[i] != 0xcd) ++errors;
        if (bytes[16 + body + i] != 0xcd) ++errors;
    }
    pixels += n * n;
    if (max_seen && errors == before) ++delayed_reads;
    if (exact) ++exact_reads;
    printf("{\"event\":\"async-case\",\"name\":\"%s\",\"frame\":%u,\"key\":\"%016llx\",\"max_age\":%u,\"exact_required\":%s,\"checked_pixels\":%u,\"checks\":%u,\"mismatches\":%u,\"largest_age_seen\":%u}\n",
        name, frame, (unsigned long long)key, max_age, exact ? "true" : "false",
        n * n, n * n + n * 12 + 32, errors - before, max_seen);
    ++tests;
}
static void cold_and_conservative(void)
{
    GfxTex* t = texture(GFX_TEX_2D, 1);
    for (unsigned p = 0; p < 12; ++p) {
        uint32_t c = code(p + 1, 0);
        char name[72];
        fill(t, 0, 0, c);
        snprintf(name, sizeof name, "fresh-key%u-exact-including-pressure", p);
        read_check(name, t, 0, 0, actor_key(p), 16, c, 1);
    }
    advance();
    uint32_t c = code(99, 1);
    fill(t, 0, 0, c);
    read_check("key-zero-is-current-sync", t, 0, 0, 0, 16, c, 1);
    c = code(99, 2); fill(t, 0, 0, c);
    read_check("known-key-maxage-zero-is-current-sync", t, 0, 0, actor_key(0), 0, c, 1);
    c = code(99, 3); fill(t, 0, 0, c);
    read_check("key-and-maxage-zero-is-current-sync", t, 0, 0, 0, 0, c, 1);
    c = code(98, 0); fill(t, 0, 0, c);
    read_check("same-object-different-caller-cold-exact", t, 0, 0,
        actor_key(0) ^ UINT64_C(0x0000100000000000), 16, c, 1);
    gfx_tex_destroy(t); advance();
}
static void textures_and_ring(void)
{
    GfxTex* a = texture(GFX_TEX_2D, 1), *b = texture(GFX_TEX_2D, 1);
    for (unsigned f = 0; f < 20; ++f) {
        for (unsigned i = 0; i < 2; ++i) {
            GfxTex* t = i ? b : a;
            uint32_t c = code(i + 20, 0);
            char name[72];
            fill(t, 0, 0, c);
            snprintf(name, sizeof name, "same-key-texture%u-ring-wrap-frame%u", i, f);
            read_check(name, t, 0, 0, actor_key(20), 16, c, f == 0);
        }
        advance();
    }
    gfx_tex_destroy(a); gfx_tex_destroy(b); advance();
}
static void face_and_mip(void)
{
    GfxTex* t = texture(GFX_TEX_CUBE, 3);
    for (unsigned f = 0; f < 5; ++f) {
        for (unsigned i = 0; i < 18; ++i) {
            unsigned face = i / 3, level = i % 3;
            uint32_t c = code(i + 30, 0);
            char name[72];
            fill(t, face, level, c);
            snprintf(name, sizeof name, "same-key-cube-face%u-mip%u-frame%u", face, level, f);
            read_check(name, t, face, level, actor_key(30), 16, c, f == 0);
        }
        advance();
    }
    gfx_tex_destroy(t); advance();
}
static void frames_inflight(void)
{
    GfxTex* a = texture(GFX_TEX_2D, 1), *b = texture(GFX_TEX_2D, 1);
    for (unsigned f = 0; f < 24; ++f) {
        for (unsigned i = 0; i < 2; ++i) {
            GfxTex* t = i ? b : a;
            uint32_t c = code(180 + i, 0); fill(t, 0, 0, c);
            char name[72]; snprintf(name, sizeof name, "inflight-texture%u-frame%u-family-agebound", i, f);
            read_check(name, t, 0, 0, actor_key(180), 16, c, f == 0);
        }
        /* Keep actual submissions in flight. The backend frame fences, not
         * this test's gfx_finish, must protect copy destination reuse. */
        gfx_present(NULL); ++frame;
    }
    gfx_tex_destroy(a); gfx_tex_destroy(b); advance();
}
static void duplicate_after_exact_fallback(void)
{
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        GfxTex* t = texture(GFX_TEX_2D, 1);
        uint64_t key = actor_key(174);
        uint32_t c = code(174 + repeat, 0); fill(t, 0, 0, c);
        read_check("fallback-duplicate-history-initial-exact", t, 0, 0, key, 16, c, 1);
        advance();
        c = code(174 + repeat, 1); fill(t, 0, 0, c);
        read_check("fallback-maxage-zero-current-exact", t, 0, 0, key, 0, c, 1);
        c = code(174 + repeat, 2); fill(t, 0, 0, c);
        read_check("sameframe-after-agezero-changed-pixels-current-exact", t, 0, 0, key, 16, c, 1);
        advance();
        c = code(174 + repeat, 3);
        uint32_t upload[256]; for (unsigned i = 0; i < 256; ++i) upload[i] = c;
        gfx_tex_upload(t, 0, 0, upload, 16 * 4);
        read_check("fallback-cpu-valid-shadow-current-exact", t, 0, 0, key, 16, c, 1);
        c = code(174 + repeat, 4); fill(t, 0, 0, c);
        read_check("sameframe-after-cpu-cache-changed-pixels-current-exact", t, 0, 0, key, 16, c, 1);
        gfx_tex_destroy(t); advance();
    }
}
static void age_boundary(void)
{
    for (unsigned gap = 1; gap <= 17; gap += (gap == 1 ? 15 : 1)) {
        GfxTex* t = texture(GFX_TEX_2D, 1);
        uint32_t c = code(70 + gap, 0);
        fill(t, 0, 0, c);
        read_check("age-bound-first-read-exact", t, 0, 0, actor_key(70), 16, c, 1);
        for (unsigned f = 0; f < gap; ++f) advance();
        c = code(70 + gap, 0); fill(t, 0, 0, c);
        char name[72];
        snprintf(name, sizeof name, "resume-after%uframes-age16-policy", gap);
        read_check(name, t, 0, 0, actor_key(70), 16, c, gap > 16);
        gfx_tex_destroy(t); advance();
    }
    GfxTex* t = texture(GFX_TEX_2D, 1);
    uint32_t c = code(89, 0); fill(t, 0, 0, c);
    read_check("age-one-policy-initial-exact", t, 0, 0, actor_key(89), 1, c, 1);
    advance(); advance();
    c = code(89, 0); fill(t, 0, 0, c);
    read_check("age-one-policy-twoframe-gap-current-exact", t, 0, 0, actor_key(89), 1, c, 1);
    gfx_tex_destroy(t); advance();
}
static void actor_reorder(void)
{
    GfxTex* t = texture(GFX_TEX_2D, 1);
    for (unsigned f = 0; f < 8; ++f) {
        for (unsigned i = 0; i < 12; ++i) {
            unsigned actor = (f & 1) ? 11 - i : (i + f) % 12;
            uint32_t c = code(actor + 100, 0);
            char name[72];
            fill(t, 0, 0, c);
            snprintf(name, sizeof name, "shared-target-actor%u-reorder-frame%u", actor, f);
            read_check(name, t, 0, 0, actor_key(actor + 100), 16, c, f == 0);
        }
        advance();
    }
    gfx_tex_destroy(t); advance();
}
static void logical_generation_reuse(void)
{
    GfxTex* t = texture(GFX_TEX_2D, 1);
    uint64_t key = actor_key(77);
    uint32_t c = code(60, 0); fill(t, 0, 0, c);
    read_check("logical-lifetime-original-current-exact", t, 0, 0, key, 16, c, 1);
    advance();
    for (unsigned generation = 0; generation < 8; ++generation) {
        c = code(60 + generation, 0); fill(t, 0, 0, c);
        read_check("logical-lifetime-old-key-queues-copy", t, 0, 0, key, 16, c, 0);
        /* Same caller, guest object and physical target: only the logical
         * generation changes. A new lifetime cannot borrow the preceding mask. */
        key = actor_key(77) ^ ((uint64_t)(generation + 1) << 48);
        c = code(61 + generation, 0); fill(t, 0, 0, c);
        read_check("logical-generation-same-object-current-exact", t, 0, 0, key, 16, c, 1);
        advance();
    }
    gfx_tex_destroy(t); advance();
}
static void same_frame_duplicate(void)
{
    GfxTex* t = texture(GFX_TEX_2D, 1);
    for (unsigned f = 0; f < 3; ++f) {
        uint32_t c = code(115, 0); fill(t, 0, 0, c);
        read_check("sameframe-first-read-family-policy", t, 0, 0, actor_key(115), 16, c, f == 0);
        /* An intervening fresh-key exact read completes the earlier copy. Even
         * though it is completed and has age0, it is not these current bytes. */
        c = code(116, f); fill(t, 0, 0, c);
        read_check("sameframe-intervening-otherkey-current-exact", t, 0, 0,
            actor_key(116 + f), 16, c, 1);
        c = code(115, 1); fill(t, 0, 0, c);
        read_check("sameframe-duplicate-changed-pixels-current-exact", t, 0, 0,
            actor_key(115), 16, c, 1);
        c = code(115, 2); fill(t, 0, 0, c);
        read_check("sameframe-third-read-current-exact", t, 0, 0, actor_key(115), 16, c, 1);
        advance();
    }
    gfx_tex_destroy(t); advance();
}
static void key_eviction_pressure(void)
{
    GfxTex* t = texture(GFX_TEX_2D, 1);
    uint32_t c = code(120, 0); fill(t, 0, 0, c);
    read_check("eviction-original-first-exact", t, 0, 0, actor_key(120), 16, c, 1);
    advance();
    c = code(120, 0); fill(t, 0, 0, c);
    read_check("eviction-original-queues-pending", t, 0, 0, actor_key(120), 16, c, 0);
    for (unsigned i = 0; i < 160; ++i) {
        c = code(1 + i + (i >= 119), 0); fill(t, 0, 0, c);
        char name[72]; snprintf(name, sizeof name, "pressure-fresh-key%u-exact", i);
        read_check(name, t, 0, 0, actor_key(200 + i), 16, c, 1);
    }
    advance();
    c = code(120, 0); fill(t, 0, 0, c);
    read_check("evicted-or-retained-original-has-own-bounded-family", t, 0, 0,
        actor_key(120), 16, c, 0);
    /* Once the128-entry table is full and old entries have become eligible,
     * fresh keys must evict safely or take exact fallback, never borrow masks. */
    for (unsigned f = 0; f < 17; ++f) advance();
    for (unsigned i = 0; i < 160; ++i) {
        c = code(1 + i + (i >= 119), 1); fill(t, 0, 0, c);
        char name[72]; snprintf(name, sizeof name, "idle-eviction-fresh-key%u-exact", i);
        read_check(name, t, 0, 0, actor_key(1000 + i), 16, c, 1);
    }
    c = code(120, 1); fill(t, 0, 0, c);
    read_check("after-idle-eviction-original-restarts-current-exact", t, 0, 0,
        actor_key(120), 16, c, 1);
    gfx_tex_destroy(t); advance();
}
static void destroy_pending(void)
{
    for (unsigned repeat = 0; repeat < 8; ++repeat) {
        GfxTex* t = texture(GFX_TEX_2D, 1);
        uint32_t c = code(240 + repeat, 0); fill(t, 0, 0, c);
        read_check("recreated-samekey-no-foreign-texture-history", t, 0, 0,
            actor_key(240), 16, c, 1);
        advance();
        c = code(240 + repeat, 0); fill(t, 0, 0, c);
        read_check("destroy-with-current-copy-pending", t, 0, 0, actor_key(240), 16, c, 0);
        gfx_tex_destroy(t);
        /* Force another allocation while old queued-copy resources are pending. */
        t = texture(GFX_TEX_2D, 1);
        c = code(250, repeat); fill(t, 0, 0, c);
        read_check("immediate-replacement-samekey-current-exact", t, 0, 0,
            actor_key(240), 16, c, 1);
        gfx_tex_destroy(t); advance();
    }
}
int main(void)
{
    gfx_set_sync_pipelines(1);
    if (!gfx_init(NULL, 0)) return 2;
    puts("{\"event\":\"async-policy\",\"identity_api\":\"Android-keyed-texture-subresource\",\"max_age_required\":16,\"same_frame_pixels_claim\":false,\"zero_key_or_age_sync_required\":true}");
    cold_and_conservative(); textures_and_ring(); face_and_mip(); frames_inflight();
    age_boundary(); actor_reorder(); logical_generation_reuse(); same_frame_duplicate(); duplicate_after_exact_fallback(); key_eviction_pressure(); destroy_pending();
    gfx_finish();
    printf("{\"event\":\"async-final\",\"tests\":%u,\"checked_pixels\":%u,\"checks\":%u,\"mismatches\":%u,\"pipeline_failures\":%u,\"exact_required_cases\":%u,\"passing_delayed_cases\":%u,\"game_or_FPS_claim\":false}\n",
        tests, pixels, checks, errors, gfx_failures(), exact_reads, delayed_reads);
    return errors || gfx_failures();
}
#endif
