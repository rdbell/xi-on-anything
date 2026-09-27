/* The graphics back end on Metal (macOS, R3.2): what gfx.h asks for, on one MTLDevice.
 *
 *   - Frames: one command buffer at a time, committed at Present (or when the CPU needs a result);
 *     up to three frames in flight, each with its own upload ring - vertex, index and uniform bytes
 *     are copied into the ring per draw, so the game may rewrite a buffer the moment a draw returns,
 *     as D3D lets it.
 *   - Render passes open on the first draw or clear after the targets change, and close when they
 *     change again, at Present, or for a blit. A clear of the whole target becomes the next pass's
 *     load action; a partial one draws a quad.
 *   - Pipelines come from the generated MSL (gfx_msl.c), cached by key; so are depth-stencil states
 *     and samplers.
 *   - Textures: sampled ones are shared storage, filled with replaceRegion when the GPU is done with
 *     them and through a blit otherwise; render targets and depth are private. Formats Metal has no
 *     match for (the 16-bit color ones) are widened to BGRA8 on upload.
 *
 * Built without ARC: every object here is retained and released by hand, and each entry point runs
 * in its own autorelease pool (the callers are guest threads with none). */
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <SDL3/SDL.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gfx.h"
#include "gfx_msl.h"

#if __has_feature(objc_arc)
#error gfx_metal.m manages its references by hand: build it with -fno-objc-arc
#endif

#define FRAMES 3
#define GFX_PROBES 8     /* reads of one surface per frame that keep their own history */
#define GFX_READBACKS ((FRAMES + 1) * GFX_PROBES)
#define RING_CHUNK (8u << 20)

/* D3DFORMAT values the back end maps */
enum
{
    F_A8R8G8B8 = 21,
    F_X8R8G8B8 = 22,
    F_R5G6B5 = 23,
    F_X1R5G5B5 = 24,
    F_A1R5G5B5 = 25,
    F_A4R4G4B4 = 26,
    F_A8 = 28,
    F_L8 = 50,
    F_A8L8 = 51,
    F_V8U8 = 60,
    F_D24S8 = 75,
    F_D24X8 = 77,
    F_D16 = 80,
};
#define FOURCC(a, b, c, d) ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))

enum
{
    CONV_NONE,
    CONV_565,
    CONV_X555,
    CONV_1555,
    CONV_4444,
};

struct GfxTex
{
    id<MTLTexture> tex;  /* what is rendered to and uploaded into */
    id<MTLTexture> view; /* what is sampled: tex, or a swizzled view of it (only level 0 of a scene filter's chain) */
    id<MTLTexture> mipview; /* the whole chain of a large render target, once the scene filter filled it */
    int type, use, conv;
    uint32_t fmt, w, h, levels;
    uint32_t block;  /* bytes per 4x4 block for the compressed formats, else 0 */
    uint32_t texel;  /* bytes per texel in Metal's layout */
    uint64_t used;   /* the last frame serial that referenced it */
    int has_stencil, x8;
    id<MTLTexture> depth_seen; /* the depth its first level was last drawn with */
    id<MTLTexture> depth_world; /* the depth the scene's casters (its world) were last drawn with: what the scene
                                 * effects read - a later pass on it with another depth does not change it */
    uint32_t mips;   /* levels of tex: t->levels, or a whole chain for a large render target (the scene filter) */
    uint32_t filled; /* the levels uploaded so far, a bit each */
    uint64_t scene;  /* its mips hold its first level as it is now (the scene filter): 0 once it is drawn to again */
    /* asynchronous readbacks (gfx_tex_read_async): staging buffers and the frame each was recorded in */
    id<MTLBuffer> rb[GFX_READBACKS];
    uint64_t rb_serial[GFX_READBACKS];
    uint32_t rb_face[GFX_READBACKS], rb_level[GFX_READBACKS], rb_index[GFX_READBACKS];
    uint64_t rb_frame; /* the frame the reads below were counted in */
    uint32_t rb_count; /* reads of this surface so far in that frame */
};

typedef struct Chunk
{
    id<MTLBuffer> buf;
    uint32_t used, size;
} Chunk;

typedef struct Frame
{
    Chunk* chunks;
    uint32_t nchunks, cur;
} Frame;

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static CAMetalLayer* g_layer;
static SDL_MetalView g_view;
static SDL_Window* g_window;
static id<MTLCommandBuffer> g_cmd;
static id<MTLRenderCommandEncoder> g_enc;
static uint64_t g_pass_serial; /* render passes begun so far: what a pass has been given (rt_draw_bind) */
static dispatch_semaphore_t g_frames_sem;
static Frame g_frames[FRAMES];
static uint32_t g_frame;              /* index into g_frames */
static uint64_t g_serial = 1;         /* the frame being recorded */
static _Atomic uint64_t g_completed;  /* the last frame the GPU finished */
static int g_frame_open;              /* the semaphore was taken for g_serial */
static uint32_t g_cmd_draws;          /* draws in the command buffer being recorded */
static _Atomic uint64_t g_gpu_ns;      /* GPU time of the committed command buffers (profile) */

static GfxTex* g_rt;
static uint32_t g_rt_face, g_rt_level;
static GfxTex* g_ds;
static uint32_t g_pending_clear; /* D3DCLEAR flags for the next pass's load actions */
static float g_clear_color[4], g_clear_z;
static uint32_t g_clear_stencil;
static id<MTLBuffer> g_dummy;
static id<MTLRenderPipelineState> g_present_pipe, g_present_cas_pipe, g_overlay_pipe;
/* the frame-rate overlay (g_fxs.fps): presents counted over half-second windows */
static double g_fps_since;
static uint32_t g_fps_frames;
static char g_fps_text[32] = "-- FPS";
static id<MTLSamplerState> g_present_samp;
static id<MTLTexture> g_scratch_depth;

/* The scene effects' settings: FFXI_FX_* in the environment, then FFXI_FX_FILE while the game runs
 * (gfx_scene_done, fx_reload). fx = 0 is the game as it was. */
static struct
{
    float fx, ao, radius, grade, sat, contrast, sharpen, filter, aniso, fog, fog_falloff, fog_height, fog_max, fog_sun,
        fog_g, bloom, threshold, rays, rays_decay, rays_length, light, shadow, shadow_length, sun, sun_distance, sun_soft, sun_face, sun_min, sun_direct, sun_casters, sun_near, temporal, debug, rt, rt_proj, rt_sun_elev, rt_sun_azim, rt_shadow, rt_distance, rt_near, rt_debug, hide_ui, rt_sm_keep, draw,
        draw_entities, fps;
} g_fxs;

/* --- small hash maps (key bytes -> object) ------------------------------------------------------------- */
typedef struct MapEnt
{
    uint64_t hash;
    void* key;
    size_t klen;
    id obj;
} MapEnt;

typedef struct Map
{
    MapEnt* e;
    uint32_t cap, n;
} Map;

static uint64_t fnv(const void* p, size_t n)
{
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i)
        h = (h ^ b[i]) * 1099511628211ull;
    return h ? h : 1;
}

static MapEnt* map_find(Map* m, const void* key, size_t klen, uint64_t h)
{
    if (!m->cap)
        return NULL;
    for (uint32_t i = (uint32_t)h & (m->cap - 1);; i = (i + 1) & (m->cap - 1))
    {
        MapEnt* e = &m->e[i];
        if (!e->hash)
            return NULL;
        if (e->hash == h && e->klen == klen && !memcmp(e->key, key, klen))
            return e;
    }
}

static id map_get(Map* m, const void* key, size_t klen)
{
    MapEnt* e = map_find(m, key, klen, fnv(key, klen));
    return e ? e->obj : nil;
}

static void map_put(Map* m, const void* key, size_t klen, id obj) /* takes the reference */
{
    if ((m->n + 1) * 2 > m->cap)
    {
        Map old = *m;
        m->cap = m->cap ? m->cap * 2 : 256;
        m->e = (MapEnt*)calloc(m->cap, sizeof *m->e);
        m->n = 0;
        for (uint32_t i = 0; i < old.cap; ++i)
            if (old.e[i].hash)
            {
                uint32_t j = (uint32_t)old.e[i].hash & (m->cap - 1);
                while (m->e[j].hash)
                    j = (j + 1) & (m->cap - 1);
                m->e[j] = old.e[i];
                m->n++;
            }
        free(old.e);
    }
    uint64_t h = fnv(key, klen);
    uint32_t j = (uint32_t)h & (m->cap - 1);
    while (m->e[j].hash)
        j = (j + 1) & (m->cap - 1);
    m->e[j].hash = h;
    m->e[j].key = malloc(klen);
    memcpy(m->e[j].key, key, klen);
    m->e[j].klen = klen;
    m->e[j].obj = obj;
    m->n++;
}

static Map g_libs, g_pipes, g_depths, g_samplers, g_clear_pipes;

/* --- the frame profile ------------------------------------------------------------------------------------ */
int gfx_profiling;

typedef struct Prof
{
    uint64_t frames, draws, bytes, pipelines, front_ns, draw_ns, sem_ns, drawable_ns, present_ns, last_ns, since_ns;
    uint64_t skips[GFX_NSKIPS];
    uint64_t shim_ns, probe_ns;
} Prof;

static Prof g_prof;

uint64_t gfx_now_ns(void)
{
    static mach_timebase_info_data_t tb;
    if (!tb.denom)
        mach_timebase_info(&tb);
    return mach_absolute_time() * tb.numer / tb.denom;
}

void gfx_prof_front(uint64_t ns) { g_prof.front_ns += ns; }
void gfx_prof_skip(int reason) { if (reason >= 0 && reason < GFX_NSKIPS) g_prof.skips[reason]++; }

static pthread_t g_present_thread;

void gfx_prof_shim(uint64_t ns)
{
    if (pthread_equal(pthread_self(), g_present_thread))
        g_prof.shim_ns += ns;
}

/* at each Present: every two seconds, the average frame and where it went */
static void prof_frame(uint64_t present_start)
{
    uint64_t now = gfx_now_ns();
    g_prof.present_ns += now - present_start;
    g_prof.frames++;
    if (!g_prof.since_ns)
        g_prof.since_ns = now;
    if (now - g_prof.since_ns < 2000000000ull)
        return;
    double f = (double)g_prof.frames, ms = 1e-6 / f;
    double frame = (double)(now - g_prof.since_ns) * ms;
    double front = (double)g_prof.front_ns * ms, enc = (double)g_prof.draw_ns * ms;
    double sem = (double)g_prof.sem_ns * ms, drw = (double)g_prof.drawable_ns * ms, pres = (double)g_prof.present_ns * ms;
    double shims = (double)g_prof.shim_ns * ms, probe = (double)g_prof.probe_ns * ms;
    if (shims > 0)
        fprintf(stderr,
            "[gfx] %.1f fps: frame %.2f ms = game code %.2f + API calls %.2f (draws %.2f [encode %.2f], probe wait %.2f, "
            "present %.2f [drawable %.2f], other %.2f) | GPU %.2f ms | %.0f draws, %.0f KB up, %llu new pipelines\n",
            f / ((double)(now - g_prof.since_ns) * 1e-9), frame, frame - shims, shims, front, enc, probe, pres, drw,
            shims - front - probe - pres, (double)atomic_exchange(&g_gpu_ns, 0) * ms, (double)g_prof.draws / f,
            (double)g_prof.bytes / f / 1024.0, (unsigned long long)g_prof.pipelines);
    else
    fprintf(stderr,
        "[gfx] %.1f fps: frame %.2f ms = game %.2f + d3d %.2f (encode %.2f) + present %.2f (gpu wait %.2f, drawable %.2f) | "
        "GPU %.2f ms | %.0f draws, %.0f KB up, %llu new pipelines\n",
        f / ((double)(now - g_prof.since_ns) * 1e-9), frame, frame - front - pres, front, enc, pres, sem, drw,
        (double)atomic_exchange(&g_gpu_ns, 0) * ms,
        (double)g_prof.draws / f, (double)g_prof.bytes / f / 1024.0, (unsigned long long)g_prof.pipelines);
    static const char* const WHY[GFX_NSKIPS] = { "no shader", "stream>=4", "no position", "no buffer data", "range past buffer",
        "no indices", "pipeline not ready", "no target" };
    int any = 0;
    for (int i = 0; i < GFX_NSKIPS; ++i)
        if (g_prof.skips[i])
            fprintf(stderr, "%s%s %llu", any++ ? ", " : "[gfx]   skipped draws (2 s): ", WHY[i], (unsigned long long)g_prof.skips[i]);
    if (any)
        fprintf(stderr, "\n");
    memset(&g_prof, 0, sizeof g_prof);
    g_prof.since_ns = now;
}
static uint32_t g_failures;

uint32_t gfx_failures(void) { return g_failures; }

/* --- frames and the upload ring ----------------------------------------------------------------------- */
static void frame_begin(void)
{
    if (g_frame_open)
        return;
    uint64_t t0 = gfx_profiling ? gfx_now_ns() : 0;
    dispatch_semaphore_wait(g_frames_sem, DISPATCH_TIME_FOREVER);
    if (gfx_profiling)
        g_prof.sem_ns += gfx_now_ns() - t0;
    g_frame_open = 1;
    Frame* f = &g_frames[g_frame];
    for (uint32_t i = 0; i < f->nchunks; ++i)
        f->chunks[i].used = 0;
    f->cur = 0;
}

static id<MTLCommandBuffer> cmd(void)
{
    frame_begin();
    if (!g_cmd)
        g_cmd = [[g_queue commandBuffer] retain];
    return g_cmd;
}

/* n bytes of this frame's ring; the buffer and offset for binding */
static void* ring(size_t n, size_t align, id<MTLBuffer>* buf, NSUInteger* off)
{
    frame_begin();
    Frame* f = &g_frames[g_frame];
    for (;; f->cur++)
    {
        if (f->cur == f->nchunks)
        {
            uint32_t size = n + align > RING_CHUNK ? (uint32_t)(n + align) : RING_CHUNK;
            f->chunks = (Chunk*)realloc(f->chunks, (f->nchunks + 1) * sizeof(Chunk));
            f->chunks[f->nchunks].buf = [g_dev newBufferWithLength:size options:MTLResourceStorageModeShared];
            f->chunks[f->nchunks].size = size;
            f->chunks[f->nchunks].used = 0;
            f->nchunks++;
        }
        Chunk* c = &f->chunks[f->cur];
        uint32_t at = (uint32_t)((c->used + align - 1) & ~(align - 1));
        if (at + n <= c->size)
        {
            c->used = at + (uint32_t)n;
            g_prof.bytes += n;
            *buf = c->buf;
            *off = at;
            return (uint8_t*)[c->buf contents] + at;
        }
    }
}

static void end_pass(void);
static int begin_pass(void);

/* closes the render pass, first opening one if a clear is still pending, so the clear happens
 * before whatever comes next (a blit, a readback, other targets) */
static void flush_pass(void)
{
    if (g_pending_clear)
        begin_pass();
    end_pass();
}

static void commit_cmd(int wait)
{
    if (gfx_profiling)
        [g_cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
            atomic_fetch_add(&g_gpu_ns, (uint64_t)((done.GPUEndTime - done.GPUStartTime) * 1e9));
        }];
    [g_cmd commit];
    if (wait)
        [g_cmd waitUntilCompleted];
    [g_cmd release];
    g_cmd = nil;
    g_cmd_draws = 0;
}

/* commits what is recorded; the frame stays open (its ring is still in use) */
static void submit(int wait)
{
    flush_pass();
    if (!g_cmd)
        return;
    commit_cmd(wait);
}

/* a chunk of the frame to the GPU (the render pass is already closed) */
static void commit_chunk(void)
{
    if (g_cmd)
        commit_cmd(0);
}

/* --- formats ---------------------------------------------------------------------------------------------- */
static MTLPixelFormat pixel_format(uint32_t fmt, int use, int* conv, uint32_t* block, uint32_t* texel, MTLTextureSwizzleChannels* sw,
    int* swizzled)
{
    *conv = CONV_NONE, *block = 0, *texel = 4, *swizzled = 0;
    *sw = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
    if (use == GFX_USE_DEPTH)
        return fmt == F_D16 ? MTLPixelFormatDepth16Unorm : MTLPixelFormatDepth32Float_Stencil8;
    switch (fmt)
    {
    case F_A8R8G8B8: return MTLPixelFormatBGRA8Unorm;
    case F_X8R8G8B8:
        sw->alpha = MTLTextureSwizzleOne, *swizzled = 1;
        return MTLPixelFormatBGRA8Unorm;
    case F_R5G6B5: *conv = CONV_565; return MTLPixelFormatBGRA8Unorm;
    case F_X1R5G5B5: *conv = CONV_X555; return MTLPixelFormatBGRA8Unorm;
    case F_A1R5G5B5: *conv = CONV_1555; return MTLPixelFormatBGRA8Unorm;
    case F_A4R4G4B4: *conv = CONV_4444; return MTLPixelFormatBGRA8Unorm;
    case F_A8: *texel = 1; return MTLPixelFormatA8Unorm;
    case F_L8:
        *texel = 1, *swizzled = 1;
        *sw = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleOne);
        return MTLPixelFormatR8Unorm;
    case F_A8L8:
        *texel = 2, *swizzled = 1;
        *sw = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleGreen);
        return MTLPixelFormatRG8Unorm;
    case F_V8U8: *texel = 2; return MTLPixelFormatRG8Snorm;
    }
    if (fmt == FOURCC('D', 'X', 'T', '1'))
        return *block = 8, MTLPixelFormatBC1_RGBA;
    if (fmt == FOURCC('D', 'X', 'T', '2') || fmt == FOURCC('D', 'X', 'T', '3'))
        return *block = 16, MTLPixelFormatBC2_RGBA;
    if (fmt == FOURCC('D', 'X', 'T', '4') || fmt == FOURCC('D', 'X', 'T', '5'))
        return *block = 16, MTLPixelFormatBC3_RGBA;
    return MTLPixelFormatBGRA8Unorm;
}

static uint32_t d3d_bpp(uint32_t fmt)
{
    switch (fmt)
    {
    case F_A8:
    case F_L8: return 1;
    case F_R5G6B5:
    case F_X1R5G5B5:
    case F_A1R5G5B5:
    case F_A4R4G4B4:
    case F_A8L8:
    case F_V8U8: return 2;
    default: return 4;
    }
}

static uint32_t expand(uint32_t v, int bits)
{
    return bits == 1 ? (v ? 255u : 0u) : (v << (8 - bits)) | (v >> (2 * bits - 8 > 0 ? 2 * bits - 8 : 0));
}

/* one row of a 16-bit D3D format into BGRA8 */
static void convert_row(int conv, const uint8_t* src, uint8_t* dst, uint32_t w)
{
    for (uint32_t x = 0; x < w; ++x)
    {
        uint32_t p = (uint32_t)src[2 * x] | ((uint32_t)src[2 * x + 1] << 8), r, g, b, a;
        switch (conv)
        {
        case CONV_565: r = expand(p >> 11, 5), g = expand((p >> 5) & 63, 6), b = expand(p & 31, 5), a = 255; break;
        case CONV_X555: r = expand((p >> 10) & 31, 5), g = expand((p >> 5) & 31, 5), b = expand(p & 31, 5), a = 255; break;
        case CONV_1555:
            r = expand((p >> 10) & 31, 5), g = expand((p >> 5) & 31, 5), b = expand(p & 31, 5), a = (p & 0x8000) ? 255 : 0;
            break;
        default: r = (p >> 8 & 15) * 17, g = (p >> 4 & 15) * 17, b = (p & 15) * 17, a = (p >> 12) * 17; break;
        }
        dst[4 * x] = (uint8_t)b, dst[4 * x + 1] = (uint8_t)g, dst[4 * x + 2] = (uint8_t)r, dst[4 * x + 3] = (uint8_t)a;
    }
}

/* --- static buffers --------------------------------------------------------------------------------------------- */
struct GfxBuf
{
    id<MTLBuffer> b;
    uint32_t size;
    uint64_t used; /* the last frame serial that drew from it */
};

GfxBuf* gfx_buf_create(uint32_t size)
{
    if (!g_dev || !size)
        return NULL;
    GfxBuf* b = (GfxBuf*)calloc(1, sizeof *b);
    b->size = size;
    b->b = [g_dev newBufferWithLength:size options:MTLResourceStorageModeShared];
    return b;
}

static void sun_cache_forget(id<MTLBuffer> buf);

void gfx_buf_destroy(GfxBuf* b)
{
    if (!b)
        return;
    sun_cache_forget(b->b);
    [b->b release]; /* command buffers that draw from it hold their own references */
    free(b);
}

void gfx_buf_upload(GfxBuf* b, const void* data, uint32_t size)
{
    if (!b)
        return;
    if (size > b->size)
        size = b->size;
    sun_cache_forget(b->b); /* new contents: the cache's copy of what it drew is no longer it */
    if (b->used > atomic_load(&g_completed))
    {
        /* recorded or running work still reads the old contents: rename */
        [b->b release];
        b->b = [g_dev newBufferWithLength:b->size options:MTLResourceStorageModeShared];
    }
    memcpy([b->b contents], data, size);
}

/* --- textures ------------------------------------------------------------------------------------------------ */
GfxTex* gfx_tex_create(int type, uint32_t fmt, uint32_t w, uint32_t h, uint32_t levels, int use)
{
    if (!g_dev)
        return NULL;
    @autoreleasepool
    {
        GfxTex* t = (GfxTex*)calloc(1, sizeof *t);
        MTLTextureSwizzleChannels sw;
        int swizzled;
        MTLPixelFormat pf = pixel_format(fmt, use, &t->conv, &t->block, &t->texel, &sw, &swizzled);
        t->type = type, t->use = use, t->fmt = fmt, t->w = w ? w : 1, t->h = h ? h : 1, t->levels = levels ? levels : 1;
        t->has_stencil = pf == MTLPixelFormatDepth32Float_Stencil8;
        t->x8 = fmt == F_X8R8G8B8;
        MTLTextureDescriptor* d = [[MTLTextureDescriptor alloc] init];
        d.textureType = type == GFX_TEX_CUBE ? MTLTextureTypeCube : MTLTextureType2D;
        d.pixelFormat = pf;
        d.width = t->w;
        d.height = type == GFX_TEX_CUBE ? t->w : t->h;
        t->mips = t->levels;
        /* a large color target (FFXI's background) gets a whole mip chain: the finished scene is made
         * smaller through it rather than one bilinear sample per screen pixel (gfx_scene_done) */
        if (use == GFX_USE_RT && type == GFX_TEX_2D && t->levels == 1 && t->w >= 1024 && t->h >= 1024)
            while ((t->w >> t->mips) || (t->h >> t->mips))
                t->mips++;
        d.mipmapLevelCount = t->mips;
        if (use == GFX_USE_SAMPLE)
        {
            d.storageMode = MTLStorageModeShared;
            d.usage = MTLTextureUsageShaderRead | (swizzled ? MTLTextureUsagePixelFormatView : 0);
        }
        else
        {
            d.storageMode = MTLStorageModePrivate;
            d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | (swizzled ? MTLTextureUsagePixelFormatView : 0);
        }
        t->tex = [g_dev newTextureWithDescriptor:d];
        [d release];
        if (!t->tex)
        {
            fprintf(stderr, "[recomp] gfx: texture %ux%u format %u failed\n", w, h, fmt);
            free(t);
            return NULL;
        }
        if (swizzled && use != GFX_USE_DEPTH)
            t->view = [t->tex newTextureViewWithPixelFormat:pf textureType:t->tex.textureType levels:NSMakeRange(0, t->levels)
                                                     slices:NSMakeRange(0, type == GFX_TEX_CUBE ? 6 : 1) swizzle:sw];
        else if (t->mips > t->levels)
        {
            /* the game's draws see the one level it made: the rest hold nothing until the scene
             * filter builds them, and a sampler with a mip filter would read them */
            t->view = [t->tex newTextureViewWithPixelFormat:pf textureType:MTLTextureType2D levels:NSMakeRange(0, 1)
                                                     slices:NSMakeRange(0, 1)];
            t->mipview = [t->tex retain];
        }
        else
            t->view = [t->tex retain];
        return t;
    }
}

void gfx_tex_destroy(GfxTex* t)
{
    if (!t)
        return;
    if (g_rt == t)
        end_pass(), g_rt = NULL;
    if (g_ds == t)
        end_pass(), g_ds = NULL;
    [t->view release];
    [t->mipview release];
    [t->depth_seen release];
    [t->depth_world release];
    [t->tex release]; /* the command buffers that use it hold their own references */
    for (int i = 0; i < GFX_READBACKS; ++i)
        [t->rb[i] release];
    free(t);
}

static void level_size(const GfxTex* t, uint32_t level, uint32_t* w, uint32_t* h)
{
    *w = t->w >> level ? t->w >> level : 1;
    *h = (t->type == GFX_TEX_CUBE ? t->w : t->h) >> level;
    if (!*h)
        *h = 1;
}

void gfx_tex_upload_rect(GfxTex* t, uint32_t face, uint32_t level, uint32_t x, uint32_t y, uint32_t w, uint32_t h,
    const void* src, uint32_t pitch)
{
    if (!t || level >= t->levels || !w || !h)
        return;
    if (level < 32)
        t->filled |= 1u << level;
    @autoreleasepool
    {
        uint32_t lw, lh;
        level_size(t, level, &lw, &lh);
        if (x >= lw || y >= lh)
            return;
        if (x + w > lw)
            w = lw - x;
        if (y + h > lh)
            h = lh - y;
        uint32_t rows = t->block ? (h + 3) / 4 : h;
        uint32_t row_bytes = t->block ? ((w + 3) / 4) * t->block : w * t->texel;
        /* the bytes in Metal's layout: as given, or widened */
        const uint8_t* data = (const uint8_t*)src;
        uint8_t* conv = NULL;
        uint32_t data_pitch = pitch;
        if (t->conv)
        {
            conv = (uint8_t*)malloc((size_t)row_bytes * rows);
            for (uint32_t r = 0; r < rows; ++r)
                convert_row(t->conv, data + (size_t)r * pitch, conv + (size_t)r * row_bytes, w);
            data = conv, data_pitch = row_bytes;
        }
        MTLRegion region = MTLRegionMake2D(x, y, w, h);
        int busy = t->used > atomic_load(&g_completed);
        if (t->use == GFX_USE_SAMPLE && !busy)
            [t->tex replaceRegion:region mipmapLevel:level slice:face withBytes:data bytesPerRow:data_pitch bytesPerImage:0];
        else
        {
            /* in use by recorded or running work, or private: through the ring, in order */
            id<MTLBuffer> buf;
            NSUInteger off;
            uint8_t* dst = (uint8_t*)ring((size_t)row_bytes * rows, 256, &buf, &off);
            for (uint32_t r = 0; r < rows; ++r)
                memcpy(dst + (size_t)r * row_bytes, data + (size_t)r * data_pitch, row_bytes);
            flush_pass();
            id<MTLBlitCommandEncoder> blit = [cmd() blitCommandEncoder];
            [blit copyFromBuffer:buf sourceOffset:off sourceBytesPerRow:row_bytes sourceBytesPerImage:(NSUInteger)row_bytes * rows
                      sourceSize:MTLSizeMake(w, h, 1) toTexture:t->tex destinationSlice:face destinationLevel:level
               destinationOrigin:MTLOriginMake(x, y, 0)];
            [blit endEncoding];
            t->used = g_serial;
        }
        free(conv);
    }
}

void gfx_tex_upload(GfxTex* t, uint32_t face, uint32_t level, const void* src, uint32_t pitch)
{
    if (!t || level >= t->levels)
        return;
    uint32_t w, h;
    level_size(t, level, &w, &h);
    gfx_tex_upload_rect(t, face, level, 0, 0, w, h, src, pitch);
}

/* rows of a level read back in Metal's layout -> D3D's */
static void copy_out(const GfxTex* t, const uint8_t* s, uint32_t w, uint32_t h, uint32_t row, void* dst, uint32_t pitch)
{
    uint32_t bpp = d3d_bpp(t->fmt);
    for (uint32_t y = 0; y < h; ++y)
    {
        uint8_t* d = (uint8_t*)dst + (size_t)y * pitch;
        if (!t->conv)
            memcpy(d, s + (size_t)y * row, (size_t)w * bpp);
        else /* 16-bit color targets: back from BGRA8 */
            for (uint32_t x = 0; x < w; ++x)
            {
                const uint8_t* p = s + (size_t)y * row + 4 * x;
                uint32_t b = p[0], g = p[1], r = p[2], a = p[3], v;
                switch (t->conv)
                {
                case CONV_565: v = (r >> 3) << 11 | (g >> 2) << 5 | b >> 3; break;
                case CONV_4444: v = (a >> 4) << 12 | (r >> 4) << 8 | (g >> 4) << 4 | b >> 4; break;
                default: v = (a >= 128 ? 0x8000u : 0) | (r >> 3) << 10 | (g >> 3) << 5 | b >> 3; break;
                }
                d[2 * x] = (uint8_t)v, d[2 * x + 1] = (uint8_t)(v >> 8);
            }
    }
}

/* a blit of the level into buf, recorded with the frame's other work */
static void queue_readback(GfxTex* t, uint32_t face, uint32_t level, uint32_t w, uint32_t h, uint32_t row, id<MTLBuffer> buf)
{
    flush_pass();
    id<MTLBlitCommandEncoder> blit = [cmd() blitCommandEncoder];
    [blit copyFromTexture:t->tex sourceSlice:face sourceLevel:level sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
                 toBuffer:buf destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:(NSUInteger)row * h];
    [blit endEncoding];
    t->used = g_serial;
}

void gfx_tex_read(GfxTex* t, uint32_t face, uint32_t level, void* dst, uint32_t pitch)
{
    if (!t || level >= t->levels || t->use == GFX_USE_DEPTH || t->block)
        return;
    @autoreleasepool
    {
        uint32_t w, h;
        level_size(t, level, &w, &h);
        uint32_t row = w * t->texel;
        id<MTLBuffer> buf = [g_dev newBufferWithLength:(NSUInteger)row * h options:MTLResourceStorageModeShared];
        queue_readback(t, face, level, w, h, row, buf);
        uint64_t t0 = gfx_profiling ? gfx_now_ns() : 0;
        submit(1);
        if (gfx_profiling)
            g_prof.probe_ns += gfx_now_ns() - t0;
        copy_out(t, (const uint8_t*)[buf contents], w, h, row, dst, pitch);
        [buf release];
    }
}

/* A surface read several times a frame (the game reuses one 16x16 target for more than one
 * probe: copy a region, lock, read; copy another, lock, read) keeps each read's history apart:
 * the k-th read this frame gets the k-th read of the newest frame the GPU has finished, never
 * another probe's pixels. */
void gfx_tex_read_async(GfxTex* t, uint32_t face, uint32_t level, void* dst, uint32_t pitch)
{
    if (!t || level >= t->levels || t->use == GFX_USE_DEPTH || t->block)
        return;
    @autoreleasepool
    {
        uint32_t w, h;
        level_size(t, level, &w, &h);
        uint32_t row = w * t->texel;
        if (t->rb_frame != g_serial)
            t->rb_frame = g_serial, t->rb_count = 0;
        uint32_t index = t->rb_count++;
        if (index >= GFX_PROBES)
        {
            gfx_tex_read(t, face, level, dst, pitch); /* more reads a frame than we keep apart */
            return;
        }
        uint64_t done = atomic_load(&g_completed);
        int best = -1;
        for (int i = 0; i < GFX_READBACKS; ++i)
            if (t->rb[i] && t->rb_serial[i] && t->rb_serial[i] <= done && t->rb_index[i] == index && t->rb_face[i] == face &&
                t->rb_level[i] == level && (best < 0 || t->rb_serial[i] > t->rb_serial[best]))
                best = i;
        if (best < 0)
            gfx_tex_read(t, face, level, dst, pitch); /* nothing finished for this read yet: wait, once */
        else
            copy_out(t, (const uint8_t*)[t->rb[best] contents], w, h, row, dst, pitch);
        /* this read's copy for a later frame: a slot the GPU is done with, not the one just read */
        int slot = -1;
        for (int i = 0; i < GFX_READBACKS && slot < 0; ++i)
            if (i != best && (!t->rb_serial[i] || t->rb_serial[i] <= done))
                slot = i;
        if (slot < 0)
            return;
        if (!t->rb[slot] || [t->rb[slot] length] < (NSUInteger)row * h)
        {
            [t->rb[slot] release];
            t->rb[slot] = [g_dev newBufferWithLength:(NSUInteger)row * h options:MTLResourceStorageModeShared];
        }
        queue_readback(t, face, level, w, h, row, t->rb[slot]);
        t->rb_serial[slot] = g_serial, t->rb_face[slot] = face, t->rb_level[slot] = level, t->rb_index[slot] = index;
    }
}

void gfx_copy(GfxTex* src, uint32_t sface, uint32_t slevel, uint32_t sx, uint32_t sy, uint32_t w, uint32_t h, GfxTex* dst,
    uint32_t dface, uint32_t dlevel, uint32_t dx, uint32_t dy)
{
    if (!src || !dst || src->tex.pixelFormat != dst->tex.pixelFormat || !w || !h)
        return;
    @autoreleasepool
    {
        flush_pass();
        id<MTLBlitCommandEncoder> blit = [cmd() blitCommandEncoder];
        [blit copyFromTexture:src->tex sourceSlice:sface sourceLevel:slevel sourceOrigin:MTLOriginMake(sx, sy, 0)
                   sourceSize:MTLSizeMake(w, h, 1) toTexture:dst->tex destinationSlice:dface destinationLevel:dlevel
            destinationOrigin:MTLOriginMake(dx, dy, 0)];
        [blit endEncoding];
        src->used = dst->used = g_serial;
        if (!dlevel)
            dst->scene = 0;
    }
}

/* --- render passes --------------------------------------------------------------------------------------------- */
/* Draws recorded since the last commit. A frame goes to the GPU in chunks - the command buffer is
 * committed when a render pass ends with this many behind it - so the GPU works while the game
 * builds the rest, and a mid-frame readback (the game's per-frame probe) waits only for the tail. */
#define CHUNK_DRAWS 96
#define SPLIT_DRAWS 320
static void commit_chunk(void);

static void end_pass(void)
{
    if (g_enc)
    {
        [g_enc endEncoding];
        [g_enc release];
        g_enc = nil;
        if (g_cmd_draws >= CHUNK_DRAWS)
            commit_chunk();
    }
}

static void color_size(uint32_t* w, uint32_t* h)
{
    level_size(g_rt, g_rt_level, w, h);
}

/* the depth attachment for the current color target: the bound one, or a scratch one of the color
 * target's size when they differ (D3D lets depth be larger; Metal wants them equal) */
static id<MTLTexture> depth_attachment(void)
{
    if (!g_ds || !g_rt)
        return nil;
    uint32_t w, h;
    color_size(&w, &h);
    if (g_ds->w == w && g_ds->h == h)
        return g_ds->tex;
    if (!g_scratch_depth || g_scratch_depth.width != w || g_scratch_depth.height != h ||
        g_scratch_depth.pixelFormat != g_ds->tex.pixelFormat)
    {
        [g_scratch_depth release];
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:g_ds->tex.pixelFormat width:w height:h
                                                                                mipmapped:NO];
        d.storageMode = MTLStorageModePrivate;
        d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        g_scratch_depth = [g_dev newTextureWithDescriptor:d];
    }
    return g_scratch_depth;
}

static int begin_pass(void)
{
    if (g_enc)
        return 1;
    if (!g_rt)
        return 0;
    MTLRenderPassDescriptor* p = [MTLRenderPassDescriptor renderPassDescriptor];
    p.colorAttachments[0].texture = g_rt->tex;
    p.colorAttachments[0].slice = g_rt_face;
    p.colorAttachments[0].level = g_rt_level;
    p.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (g_pending_clear & 1)
    {
        p.colorAttachments[0].loadAction = MTLLoadActionClear;
        p.colorAttachments[0].clearColor = MTLClearColorMake(g_clear_color[0], g_clear_color[1], g_clear_color[2], g_clear_color[3]);
    }
    else
        p.colorAttachments[0].loadAction = MTLLoadActionLoad;
    id<MTLTexture> depth = depth_attachment();
    if (depth)
    {
        p.depthAttachment.texture = depth;
        p.depthAttachment.storeAction = MTLStoreActionStore;
        p.depthAttachment.loadAction = (g_pending_clear & 2) ? MTLLoadActionClear : MTLLoadActionLoad;
        p.depthAttachment.clearDepth = g_clear_z;
        if (g_ds->has_stencil)
        {
            p.stencilAttachment.texture = depth;
            p.stencilAttachment.storeAction = MTLStoreActionStore;
            p.stencilAttachment.loadAction = (g_pending_clear & 4) ? MTLLoadActionClear : MTLLoadActionLoad;
            p.stencilAttachment.clearStencil = g_clear_stencil;
        }
        if (!g_rt_face && !g_rt_level && g_rt->depth_seen != depth)
        {
            [g_rt->depth_seen release];
            g_rt->depth_seen = [depth retain];
        }
    }
    g_pending_clear = 0;
    g_enc = [[cmd() renderCommandEncoderWithDescriptor:p] retain];
    g_pass_serial++;
    g_rt->used = g_serial;
    if (!g_rt_face && !g_rt_level)
        g_rt->scene = 0; /* drawn to: its mips are behind */
    if (g_ds)
        g_ds->used = g_serial;
    return 1;
}

void gfx_set_targets(GfxTex* color, uint32_t face, uint32_t level, GfxTex* depth)
{
    if (color == g_rt && face == g_rt_face && level == g_rt_level && depth == g_ds)
        return;
    @autoreleasepool
    {
        flush_pass(); /* a clear nothing drew after still happens */
    }
    g_rt = color, g_rt_face = face, g_rt_level = level, g_ds = depth;
}

/* --- pipelines ------------------------------------------------------------------------------------------------- */
typedef struct LibKey
{
    GfxVsKey vs;
    GfxFsKey fs;
} LibKey;

static int alpha_tested(const GfxFsKey* k) { return k->alpha_func && k->alpha_func != 8; }

typedef struct PipeKey
{
    LibKey lib;
    GfxPipeKey pipe;
    uint32_t color, depth, stencil; /* MTLPixelFormat */
    uint8_t x8, pad[3];
} PipeKey;

static MTLBlendFactor blend_factor(uint32_t f, int x8)
{
    switch (f)
    {
    case 1: return MTLBlendFactorZero;
    case 2: return MTLBlendFactorOne;
    case 3: return MTLBlendFactorSourceColor;
    case 4: return MTLBlendFactorOneMinusSourceColor;
    case 5: return MTLBlendFactorSourceAlpha;
    case 6: return MTLBlendFactorOneMinusSourceAlpha;
    case 7: return x8 ? MTLBlendFactorOne : MTLBlendFactorDestinationAlpha;
    case 8: return x8 ? MTLBlendFactorZero : MTLBlendFactorOneMinusDestinationAlpha;
    case 9: return MTLBlendFactorDestinationColor;
    case 10: return MTLBlendFactorOneMinusDestinationColor;
    case 11: return MTLBlendFactorSourceAlphaSaturated;
    default: return MTLBlendFactorOne;
    }
}

static MTLBlendOperation blend_op(uint32_t op)
{
    switch (op)
    {
    case 2: return MTLBlendOperationSubtract;
    case 3: return MTLBlendOperationReverseSubtract;
    case 4: return MTLBlendOperationMin;
    case 5: return MTLBlendOperationMax;
    default: return MTLBlendOperationAdd;
    }
}

static id<MTLLibrary> compile(const char* src)
{
    NSError* err = nil;
    NSString* s = [[NSString alloc] initWithUTF8String:src];
    MTLCompileOptions* o = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:s options:o error:&err];
    [o release];
    [s release];
    if (!lib)
        __atomic_fetch_add(&g_failures, 1, __ATOMIC_RELAXED), fprintf(stderr, "[recomp] gfx: MSL compile failed: %s\n%s\n", err ? [[err localizedDescription] UTF8String] : "?", src);
    return lib;
}

/* --- pipelines, built off the game's thread ---------------------------------------------------------------
 * A pipeline for a key seen for the first time is built on a background queue; the draws that need
 * it are skipped until it is ready (a new effect may miss its first frames; the game never waits for
 * the Metal compiler). Every key built is recorded in the pipeline cache file, and at start-up the
 * recorded keys are built again in the background, so a second session has them before it needs
 * them. gfx_set_sync_pipelines(1) (the tests) builds in place and records nothing. */
#define PIPE_FAILED ((void*)1)
#define PIPE_MAGIC 0x314B5053u /* "SPK1" */

typedef struct PipeEntry
{
    void* _Atomic state; /* NULL while building, PIPE_FAILED, or the pipeline (retained) */
} PipeEntry;

typedef struct PipeJob
{
    PipeKey k;
    uint32_t *vs, *ps; /* copies of the shader tokens behind k.lib.vs.prog / fs.prog */
    uint32_t nvs, nps;
    PipeEntry* e;
    int record;
} PipeJob;

static int g_sync_pipelines;
static char g_pipe_cache[1024];
static dispatch_queue_t g_pipe_queue, g_pipe_file_queue;
static _Atomic uint32_t g_pipes_building;

void gfx_set_sync_pipelines(int on) { g_sync_pipelines = on; }

static uint32_t* copy_tok(const uint32_t* t, uint32_t* n)
{
    *n = 0;
    if (!t)
        return NULL;
    uint32_t k = 0;
    while (k < 65536 && t[k] != 0x0000FFFFu)
        k++;
    k++;
    uint32_t* c = (uint32_t*)malloc(4u * k);
    memcpy(c, t, 4u * k);
    *n = k;
    return c;
}

/* the pipeline for a key (retained), or nil */
static id<MTLRenderPipelineState> build_pipeline(const PipeKey* k, const uint32_t* vs, const uint32_t* ps)
{
    char* src = gfx_msl_generate(&k->lib.vs, &k->lib.fs, vs, ps);
    if (!src)
    {
        __atomic_fetch_add(&g_failures, 1, __ATOMIC_RELAXED);
        return nil;
    }
    id<MTLLibrary> lib = compile(src);
    free(src);
    if (!lib)
        return nil;
    id p = nil;
    @autoreleasepool
    {
        MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
        id<MTLFunction> vf = [lib newFunctionWithName:@"vs_main"], ff = [lib newFunctionWithName:@"fs_main"];
        if (k->lib.vs.shadow && !alpha_tested(&k->lib.fs)) /* depth alone: nothing for the fragments to do */
            [ff release], ff = nil;
        if (k->lib.vs.capture) /* the triangles out to a buffer (rt_capture): nothing drawn */
            [ff release], ff = nil, pd.rasterizationEnabled = NO;
        pd.vertexFunction = vf;
        pd.fragmentFunction = ff;
        pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassUnspecified;
        MTLRenderPipelineColorAttachmentDescriptor* c = pd.colorAttachments[0];
        c.pixelFormat = (MTLPixelFormat)k->color;
        uint32_t wm = k->pipe.write_mask;
        c.writeMask = ((wm & 1) ? MTLColorWriteMaskRed : 0) | ((wm & 2) ? MTLColorWriteMaskGreen : 0) |
            ((wm & 4) ? MTLColorWriteMaskBlue : 0) | ((wm & 8) ? MTLColorWriteMaskAlpha : 0);
        if (k->pipe.blend)
        {
            uint32_t sf = k->pipe.src, df = k->pipe.dst;
            if (sf == 12) /* BOTHSRCALPHA */
                sf = 5, df = 6;
            else if (sf == 13) /* BOTHINVSRCALPHA */
                sf = 6, df = 5;
            c.blendingEnabled = YES;
            c.sourceRGBBlendFactor = c.sourceAlphaBlendFactor = blend_factor(sf, k->x8);
            c.destinationRGBBlendFactor = c.destinationAlphaBlendFactor = blend_factor(df, k->x8);
            c.rgbBlendOperation = c.alphaBlendOperation = blend_op(k->pipe.op);
        }
        pd.depthAttachmentPixelFormat = (MTLPixelFormat)k->depth;
        pd.stencilAttachmentPixelFormat = (MTLPixelFormat)k->stencil;
        NSError* err = nil;
        p = [g_dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!p)
        {
            __atomic_fetch_add(&g_failures, 1, __ATOMIC_RELAXED);
            fprintf(stderr, "[recomp] gfx: pipeline failed: %s\n", err ? [[err localizedDescription] UTF8String] : "?");
        }
        [vf release];
        [ff release];
        [pd release];
    }
    [lib release];
    return p;
}

/* one record of the cache file: magic, the key, then each shader's tokens (count first) */
static void record_job(const PipeJob* j)
{
    if (!g_pipe_cache[0])
        return;
    size_t n = 4 + 4 + sizeof(PipeKey) + 4 + 4u * j->nvs + 4 + 4u * j->nps;
    uint8_t* rec = (uint8_t*)malloc(n);
    uint8_t* w = rec;
    uint32_t v = PIPE_MAGIC;
    memcpy(w, &v, 4), w += 4;
    v = (uint32_t)sizeof(PipeKey);
    memcpy(w, &v, 4), w += 4;
    memcpy(w, &j->k, sizeof(PipeKey)), w += sizeof(PipeKey);
    memcpy(w, &j->nvs, 4), w += 4;
    if (j->nvs)
        memcpy(w, j->vs, 4u * j->nvs), w += 4u * j->nvs;
    memcpy(w, &j->nps, 4), w += 4;
    if (j->nps)
        memcpy(w, j->ps, 4u * j->nps);
    dispatch_async(g_pipe_file_queue, ^{
        FILE* f = fopen(g_pipe_cache, "ab");
        if (f)
        {
            fwrite(rec, 1, n, f);
            fclose(f);
        }
        free(rec);
    });
}

static void run_job(PipeJob* j)
{
    id p = build_pipeline(&j->k, j->vs, j->ps);
    atomic_store(&j->e->state, p ? (void*)p : PIPE_FAILED);
    if (p && j->record)
        record_job(j);
    free(j->vs);
    free(j->ps);
    free(j);
    atomic_fetch_sub(&g_pipes_building, 1);
}

static void queue_job(const PipeKey* k, const uint32_t* vs, uint32_t nvs, const uint32_t* ps, uint32_t nps, int record)
{
    if (map_get(&g_pipes, k, sizeof *k))
        return;
    PipeEntry* e = (PipeEntry*)calloc(1, sizeof *e);
    map_put(&g_pipes, k, sizeof *k, (id)e);
    PipeJob* j = (PipeJob*)calloc(1, sizeof *j);
    j->k = *k, j->e = e, j->record = record;
    if (nvs)
        j->vs = (uint32_t*)malloc(4u * nvs), memcpy(j->vs, vs, 4u * nvs), j->nvs = nvs;
    if (nps)
        j->ps = (uint32_t*)malloc(4u * nps), memcpy(j->ps, ps, 4u * nps), j->nps = nps;
    g_prof.pipelines++;
    atomic_fetch_add(&g_pipes_building, 1);
    if (g_sync_pipelines)
        run_job(j);
    else
        dispatch_async(g_pipe_queue, ^{ run_job(j); });
}

/* at start-up: the keys earlier sessions built, built again in the background */
static void prewarm_pipelines(void)
{
    const char* dir = getenv("FFXI_CACHE_DIR");
    char path[900];
    if (dir && *dir)
        snprintf(path, sizeof path, "%s", dir);
    else if (getenv("HOME"))
        snprintf(path, sizeof path, "%s/Library/Caches/FFXI", getenv("HOME"));
    else
        return;
    mkdir(path, 0755);
    snprintf(g_pipe_cache, sizeof g_pipe_cache, "%s/pipelines.v1", path);
    FILE* f = fopen(g_pipe_cache, "rb");
    if (!f)
        return;
    uint32_t n = 0, hdr[2];
    while (fread(hdr, 4, 2, f) == 2 && hdr[0] == PIPE_MAGIC && hdr[1] == sizeof(PipeKey))
    {
        PipeKey k;
        uint32_t nvs = 0, nps = 0, *vs = NULL, *ps = NULL;
        int ok = fread(&k, sizeof k, 1, f) == 1 && fread(&nvs, 4, 1, f) == 1 && nvs <= 65536;
        if (ok && nvs)
            vs = (uint32_t*)malloc(4u * nvs), ok = fread(vs, 4, nvs, f) == nvs;
        ok = ok && fread(&nps, 4, 1, f) == 1 && nps <= 65536;
        if (ok && nps)
            ps = (uint32_t*)malloc(4u * nps), ok = fread(ps, 4, nps, f) == nps;
        if (ok)
            queue_job(&k, vs, nvs, ps, nps, 0), n++;
        free(vs);
        free(ps);
        if (!ok)
            break;
    }
    fclose(f);
    fprintf(stderr, "[recomp] gfx: building %u pipelines from %s in the background\n", n, g_pipe_cache);
}

/* the pipeline for a key, queued for building the first time (nil until built) */
static id<MTLRenderPipelineState> pipeline_for(const PipeKey* k, const uint32_t* vs_tokens, const uint32_t* ps_tokens)
{
    PipeEntry* e = (PipeEntry*)map_get(&g_pipes, k, sizeof *k);
    if (!e)
    {
        uint32_t nvs = 0, nps = 0;
        uint32_t* vs = k->lib.vs.prog ? copy_tok(vs_tokens, &nvs) : NULL;
        uint32_t* ps = k->lib.fs.prog ? copy_tok(ps_tokens, &nps) : NULL;
        queue_job(k, vs, nvs, ps, nps, !g_sync_pipelines);
        free(vs);
        free(ps);
        e = (PipeEntry*)map_get(&g_pipes, k, sizeof *k);
    }
    void* st = atomic_load(&e->state);
    return st && st != PIPE_FAILED ? (id)st : nil;
}

static int rt_draw_wanted(const GfxDraw* d);
static void rt_draw_bind(const GfxDraw* d, id<MTLRenderCommandEncoder> e);
static int rt_inpass(void);
static int g_draw_rt; /* the draw being encoded has a traced pipeline (pipeline) */

static id<MTLRenderPipelineState> pipeline(const GfxDraw* d)
{
    PipeKey k;
    memset(&k, 0, sizeof k);
    k.lib.vs = d->vs, k.lib.fs = d->fs, k.pipe = d->pipe;
    /* the world's lit draws lit per pixel (gfx_msl.c pixel_lit) */
    if (g_fxs.fx != 0.0f && g_fxs.light != 0.0f && d->vs.lighting && !d->vs.rhw && !d->vs.prog && !d->vs.flat)
        k.lib.vs.pixel = g_fxs.light >= 2.0f ? 2 : 1; /* 1: the sun per pixel, the game's torches per vertex */
    k.color = (uint32_t)g_rt->tex.pixelFormat;
    id<MTLTexture> depth = depth_attachment();
    k.depth = depth ? (uint32_t)depth.pixelFormat : 0;
    k.stencil = depth && g_ds->has_stencil ? k.depth : 0;
    k.x8 = (uint8_t)g_rt->x8;
    /* the world's opaque draws traced: shaded by the scene's triangles as they are drawn (rt_draw_bind) */
    if (rt_draw_wanted(d))
        k.lib.vs.rt = k.lib.fs.rt = 1;
    id<MTLRenderPipelineState> p = pipeline_for(&k, d->vs_tokens, d->ps_tokens);
    if (!p && k.lib.vs.rt)
    {
        /* traced, still building: drawn as it was meanwhile */
        k.lib.vs.rt = k.lib.fs.rt = 0;
        p = pipeline_for(&k, d->vs_tokens, d->ps_tokens);
    }
    if (!p && k.lib.vs.pixel)
    {
        /* lit per pixel, still building: lit per vertex meanwhile (a new mix of the game's lights
         * would blink the object out for a few frames) */
        k.lib.vs.pixel = 0;
        p = pipeline_for(&k, d->vs_tokens, d->ps_tokens);
    }
    g_draw_rt = p && k.lib.vs.rt;
    return p;
}

/* --- the sun's shadow map: the scene's casters, drawn again from the sun (gfx_scene_done) --------------
 * Each opaque draw of the scene (GfxDraw.caster) is recorded as it is encoded - its pipeline key and
 * the buffers, uniforms and textures it was bound with, all alive until the frame ends - and when the
 * scene is done they are drawn again into a depth map from the sun: the same functions, with the
 * clip-space position they make taken through the camera's inverse into the sun's view (the shadow
 * key, gfx_msl_vs_return). The scene effects then look each pixel up in it. */
typedef struct Caster
{
    LibKey lib;
    const uint32_t *vs, *ps;
    id<MTLBuffer> vb[GFX_NSTREAMS], ub, ib; /* retained */
    NSUInteger voff[GFX_NSTREAMS], uoff, ioff;
    id<MTLTexture> tex[8]; /* retained; only for an alpha test */
    GfxSampler samp[8];
    MTLPrimitiveType prim;
    uint32_t n, vstart;
    uint8_t itype; /* 0 no indices, 2 or 4 bytes each */
    uint8_t fixed; /* every vertex and index from buffers the game keeps (the zone's): cached (sun_cache) */
    uint8_t has_cw;
    int32_t zbias;
    float cw[16]; /* a fixed-function draw's clip space back to the world, its own view and projection's */
} Caster;

static Caster* g_casters;
static uint32_t g_ncasters, g_casters_cap;
/* the world's opaque draws after its scene's effects ran (GfxDraw.caster 2): traced with the next scene,
 * through the camera of theirs (rt_frame) */
static Caster* g_late;
static uint32_t g_nlate, g_late_cap;
static uint64_t g_late_serial; /* the frame they were drawn in */

static void caster_release(Caster* c)
{
    for (int s = 0; s < GFX_NSTREAMS; ++s)
        [c->vb[s] release];
    [c->ub release];
    [c->ib release];
    for (int t = 0; t < 8; ++t)
        [c->tex[t] release];
}

static void late_clear(void)
{
    for (uint32_t i = 0; i < g_nlate; ++i)
        caster_release(&g_late[i]);
    g_nlate = 0;
}

static Caster* late_new(const GfxDraw* d)
{
    if (g_late_serial != g_serial) /* the frame's first: those of frames before go */
        late_clear(), g_late_serial = g_serial;
    if (g_nlate == g_late_cap)
    {
        g_late_cap = g_late_cap ? g_late_cap * 2 : 256;
        g_late = (Caster*)realloc(g_late, g_late_cap * sizeof(Caster));
    }
    Caster* c = &g_late[g_nlate++];
    memset(c, 0, sizeof *c);
    c->lib.vs = d->vs, c->lib.fs = d->fs;
    c->vs = d->vs_tokens, c->ps = d->ps_tokens;
    c->fixed = d->prim != GFX_TRIANGLEFAN && (!d->indices || d->ibuf);
    return c;
}

static void casters_clear(void)
{
    for (uint32_t i = 0; i < g_ncasters; ++i)
    {
        Caster* c = &g_casters[i];
        for (int s = 0; s < GFX_NSTREAMS; ++s)
            [c->vb[s] release];
        [c->ub release];
        [c->ib release];
        for (int t = 0; t < 8; ++t)
            [c->tex[t] release];
    }
    g_ncasters = 0;
}

static Caster* caster_new(const GfxDraw* d)
{
    if (g_ncasters == g_casters_cap)
    {
        g_casters_cap = g_casters_cap ? g_casters_cap * 2 : 1024;
        g_casters = (Caster*)realloc(g_casters, g_casters_cap * sizeof(Caster));
    }
    Caster* c = &g_casters[g_ncasters++];
    memset(c, 0, sizeof *c);
    c->lib.vs = d->vs, c->lib.fs = d->fs;
    c->vs = d->vs_tokens, c->ps = d->ps_tokens;
    c->zbias = d->zbias;
    c->fixed = d->prim != GFX_TRIANGLEFAN && (!d->indices || d->ibuf);
    return c;
}



static MTLCompareFunction compare(uint32_t f)
{
    switch (f)
    {
    case 1: return MTLCompareFunctionNever;
    case 2: return MTLCompareFunctionLess;
    case 3: return MTLCompareFunctionEqual;
    case 4: return MTLCompareFunctionLessEqual;
    case 5: return MTLCompareFunctionGreater;
    case 6: return MTLCompareFunctionNotEqual;
    case 7: return MTLCompareFunctionGreaterEqual;
    default: return MTLCompareFunctionAlways;
    }
}

static MTLStencilOperation stencil_op(uint32_t op)
{
    switch (op)
    {
    case 2: return MTLStencilOperationZero;
    case 3: return MTLStencilOperationReplace;
    case 4: return MTLStencilOperationIncrementClamp;
    case 5: return MTLStencilOperationDecrementClamp;
    case 6: return MTLStencilOperationInvert;
    case 7: return MTLStencilOperationIncrementWrap;
    case 8: return MTLStencilOperationDecrementWrap;
    default: return MTLStencilOperationKeep;
    }
}

static id<MTLDepthStencilState> depth_state(const GfxDepthKey* k)
{
    id s = map_get(&g_depths, k, sizeof *k);
    if (s)
        return s;
    MTLDepthStencilDescriptor* d = [[MTLDepthStencilDescriptor alloc] init];
    d.depthCompareFunction = k->zenable ? compare(k->zfunc) : MTLCompareFunctionAlways;
    d.depthWriteEnabled = k->zenable && k->zwrite;
    if (k->stencil)
    {
        MTLStencilDescriptor* st = [[MTLStencilDescriptor alloc] init];
        st.stencilCompareFunction = compare(k->sfunc);
        st.stencilFailureOperation = stencil_op(k->sfail);
        st.depthFailureOperation = stencil_op(k->szfail);
        st.depthStencilPassOperation = stencil_op(k->spass);
        st.readMask = k->sread;
        st.writeMask = k->swrite;
        d.frontFaceStencil = st;
        d.backFaceStencil = st;
        [st release];
    }
    s = [g_dev newDepthStencilStateWithDescriptor:d];
    [d release];
    map_put(&g_depths, k, sizeof *k, s);
    return s;
}

static MTLSamplerAddressMode address(uint32_t a)
{
    switch (a)
    {
    case 2: return MTLSamplerAddressModeMirrorRepeat;
    case 3: return MTLSamplerAddressModeClampToEdge;
    case 4: return MTLSamplerAddressModeClampToBorderColor;
    case 5: return MTLSamplerAddressModeMirrorClampToEdge;
    default: return MTLSamplerAddressModeRepeat;
    }
}

static id<MTLSamplerState> sampler(const GfxSampler* k)
{
    id s = map_get(&g_samplers, k, sizeof *k);
    if (s)
        return s;
    MTLSamplerDescriptor* d = [[MTLSamplerDescriptor alloc] init];
    d.sAddressMode = address(k->addr_u);
    d.tAddressMode = address(k->addr_v);
    d.rAddressMode = address(k->addr_w);
    d.magFilter = k->mag >= 2 ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.minFilter = k->min >= 2 ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.mipFilter = k->mip == 0 ? MTLSamplerMipFilterNotMipmapped : k->mip == 1 ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear;
    if ((k->min == 3 || k->mag == 3) && k->max_aniso > 1)
        d.maxAnisotropy = k->max_aniso > 16 ? 16 : k->max_aniso;
    d.lodMinClamp = k->max_level;
    if (k->lod_cap)
        d.lodMaxClamp = (float)(k->lod_cap - 1);
    uint32_t a = k->border >> 24, rgb = k->border & 0xFFFFFF;
    d.borderColor = a < 128 ? MTLSamplerBorderColorTransparentBlack
        : rgb >= 0x808080 ? MTLSamplerBorderColorOpaqueWhite : MTLSamplerBorderColorOpaqueBlack;
    s = [g_dev newSamplerStateWithDescriptor:d];
    [d release];
    map_put(&g_samplers, k, sizeof *k, s);
    return s;
}

/* --- drawing --------------------------------------------------------------------------------------------------- */
static void set_viewport(const uint32_t vp[6])
{
    float zmin, zmax;
    memcpy(&zmin, &vp[4], 4);
    memcpy(&zmax, &vp[5], 4);
    uint32_t w, h;
    color_size(&w, &h);
    double x = vp[0], y = vp[1], vw = vp[2], vh = vp[3];
    if (x > w)
        x = w;
    if (y > h)
        y = h;
    if (x + vw > w)
        vw = w - x;
    if (y + vh > h)
        vh = h - y;
    [g_enc setViewport:(MTLViewport){ x, y, vw, vh, zmin, zmax }];
}

/* index count (Metal) for a D3D primitive count */
static uint32_t vertex_count(uint32_t prim, uint32_t n)
{
    switch (prim)
    {
    case GFX_POINTLIST: return n;
    case GFX_LINELIST: return n * 2;
    case GFX_LINESTRIP: return n + 1;
    case GFX_TRIANGLELIST: return n * 3;
    case GFX_TRIANGLESTRIP: return n + 2;
    case GFX_TRIANGLEFAN: return n * 3; /* as a list */
    default: return 0;
    }
}

static MTLPrimitiveType metal_prim(uint32_t prim)
{
    switch (prim)
    {
    case GFX_POINTLIST: return MTLPrimitiveTypePoint;
    case GFX_LINELIST: return MTLPrimitiveTypeLine;
    case GFX_LINESTRIP: return MTLPrimitiveTypeLineStrip;
    case GFX_TRIANGLESTRIP: return MTLPrimitiveTypeTriangleStrip;
    default: return MTLPrimitiveTypeTriangle;
    }
}

static void draw_encode(const GfxDraw* d);
static int matd_inverse(double* out, const double* m);

/* The scene filter: a large render target drawn smaller onto a large target (FFXI's world onto the
 * back buffer - or the copy of it the game shows some frames, while the camera moves) is sampled
 * through its mips, made here from it as it is now. Every frame alike: when only some were, the fine
 * detail of the sky flickered between the two. */
static void scene_mips(const GfxDraw* d)
{
    if (g_fxs.fx == 0.0f || g_fxs.filter == 0.0f || !d->vs.rhw || !g_rt)
        return;
    NSUInteger tw = g_rt->tex.width, th = g_rt->tex.height;
    if (tw * th < 1024)
        return; /* not the sun flare's 16x16 occlusion probe */
    for (int i = 0; i < 8; ++i)
    {
        GfxTex* t = d->tex[i];
        int wanted = d->fs.prog || i < d->fs.nstages ? d->fs.st[i].tex : 0;
        if (!wanted || !t || t == g_rt || !t->mipview || t->mips < 2 || t->scene)
            continue;
        if (t->tex.width <= tw && t->tex.height <= th)
            continue; /* not made smaller */
        flush_pass();
        id<MTLBlitCommandEncoder> b = [cmd() blitCommandEncoder];
        [b generateMipmapsForTexture:t->tex];
        [b endEncoding];
        t->scene = g_serial;
    }
}

void gfx_draw(const GfxDraw* d)
{
    if (!g_dev || !d->count)
        return;
    /* testing: the 3D alone - no screen-space draws but those of a rendered image (the scene's own) */
    if (g_fxs.hide_ui != 0.0f && d->vs.rhw && !(d->tex[0] && d->tex[0]->use == GFX_USE_RT))
        return;
    uint64_t t0 = gfx_profiling ? gfx_now_ns() : 0;
    scene_mips(d);
    draw_encode(d);
    g_cmd_draws++;
    /* a long pass (the 3D scene) goes to the GPU in pieces too: end it here, commit, and the next
     * draw resumes it with its contents loaded */
    if (g_cmd_draws >= SPLIT_DRAWS && g_enc)
        end_pass();
    if (gfx_profiling)
        g_prof.draw_ns += gfx_now_ns() - t0, g_prof.draws++;
}

static void draw_encode(const GfxDraw* d)
{
    @autoreleasepool
    {
        if (!begin_pass())
        {
            gfx_prof_skip(GFX_SKIP_NO_TARGET);
            return;
        }
        id<MTLRenderPipelineState> p = pipeline(d);
        if (!p)
        {
            gfx_prof_skip(GFX_SKIP_PIPELINE); /* still building (or failed) */
            return;
        }
        [g_enc setRenderPipelineState:p];
        GfxDepthKey dk = d->depth;
        if (!depth_attachment())
            memset(&dk, 0, sizeof dk);
        [g_enc setDepthStencilState:depth_state(&dk)];
        if (dk.stencil)
            [g_enc setStencilReferenceValue:d->stencil_ref];
        /* D3D's front faces are clockwise on screen; CULL_CCW (the default) culls the back ones */
        [g_enc setFrontFacingWinding:MTLWindingClockwise];
        [g_enc setCullMode:d->cull == 3 ? MTLCullModeBack : d->cull == 2 ? MTLCullModeFront : MTLCullModeNone];
        [g_enc setTriangleFillMode:d->fill == 2 ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
        [g_enc setDepthBias:-(float)d->zbias slopeScale:-(float)d->zbias * 0.5f clamp:0];
        set_viewport(d->vp);

        id<MTLBuffer> buf;
        NSUInteger off;
        /* the uniforms the draw's functions read: the lights only when lit, the vertex shader's
         * constants only for a vertex shader, the pixel shader's only for a pixel shader (the ring
         * keeps room for the whole struct: that is what the functions are compiled against) */
        size_t need = offsetof(GfxU, light) + (size_t)d->vs.nlights * sizeof(GfxLight);
        if (d->vs.prog)
            need = offsetof(GfxU, psc);
        if (d->fs.prog)
            need = sizeof(GfxU);
        void* u = ring(sizeof(GfxU), 256, &buf, &off);
        memcpy(u, &d->u, need);
        [g_enc setVertexBuffer:buf offset:off atIndex:4];
        [g_enc setFragmentBuffer:buf offset:off atIndex:4];
        if (g_draw_rt)
            rt_draw_bind(d, g_enc);
        if (d->caster == 1 && !g_rt_face && !g_rt_level)
        {
            id<MTLTexture> dw = depth_attachment();
            if (dw && g_rt->depth_world != dw)
                [g_rt->depth_world release], g_rt->depth_world = [dw retain];
        }
        Caster* rec = d->caster == 1 && g_fxs.fx != 0.0f && (g_fxs.sun > 0.0f || g_fxs.rt > 0.0f) ? caster_new(d)
            : d->caster == 2 && g_fxs.fx != 0.0f && g_fxs.rt > 0.0f && !g_rt_face && !g_rt_level ? late_new(d) : NULL;
        if (rec)
        {
            rec->ub = [buf retain], rec->uoff = off;
            if (!d->vs.prog) /* the fixed function's own camera: its clip space back to the world */
            {
                double vp[16], inv[16];
                for (int j = 0; j < 16; ++j)
                    vp[j] = d->view_proj[j];
                if (matd_inverse(inv, vp))
                    for (int j = 0; j < 16; ++j)
                        rec->cw[j] = (float)inv[j], rec->has_cw = 1;
            }
        }
        for (int s = 0; s < GFX_NSTREAMS; ++s)
        {
            if (d->buf[s])
            {
                [g_enc setVertexBuffer:d->buf[s]->b offset:d->buf_off[s] atIndex:(NSUInteger)s];
                d->buf[s]->used = g_serial;
                if (rec)
                    rec->vb[s] = [d->buf[s]->b retain], rec->voff[s] = d->buf_off[s];
            }
            else if (d->data[s] && d->size[s])
            {
                void* v = ring(d->size[s], 16, &buf, &off);
                memcpy(v, d->data[s], d->size[s]);
                [g_enc setVertexBuffer:buf offset:off atIndex:(NSUInteger)s];
                if (rec)
                    rec->vb[s] = [buf retain], rec->voff[s] = off, rec->fixed = 0;
            }
            else
            {
                [g_enc setVertexBuffer:g_dummy offset:0 atIndex:(NSUInteger)s];
                if (rec)
                    rec->vb[s] = [g_dummy retain];
            }
        }
        for (int i = 0; i < 8; ++i)
        {
            GfxTex* t = d->tex[i];
            int wanted = d->fs.prog || i < d->fs.nstages ? d->fs.st[i].tex : 0;
            if (!wanted || !t)
                continue;
            id<MTLTexture> view = t->view;
            GfxSampler sk = d->samp[i];
            if (g_fxs.fx != 0.0f)
            {
                /* the world's solid textures (not the interface's, not the sky's or effects' - which write
                 * no depth: the sky's clouds, mapped at steep angles near the horizon, flicker through
                 * an anisotropic filter as the camera turns), where every mip is there: trilinear and
                 * anisotropic, so ground and walls at a slant stay sharp and do not swim */
                if (!d->vs.rhw && d->depth.zwrite && !d->fs.st[i].projected && g_fxs.aniso > 1.0f && t->levels > 1 &&
                    t->levels < 32 &&
                    t->filled == (1u << t->levels) - 1 && sk.min >= 2)
                    sk.min = 3, sk.mip = 2, sk.max_aniso = (uint8_t)(g_fxs.aniso > 16.0f ? 16.0f : g_fxs.aniso);
                /* the finished scene made smaller (FFXI's background onto the back buffer): through
                 * its mips, anisotropic for a squeeze that differs across and down */
                else if (t->scene && t->mipview)
                    sk.min = 3, sk.mag = 2, sk.mip = 2, sk.max_aniso = 16, sk.max_level = 0, view = t->mipview;
            }
            [g_enc setFragmentTexture:view atIndex:(NSUInteger)i];
            [g_enc setFragmentSamplerState:sampler(&sk) atIndex:(NSUInteger)i];
            t->used = g_serial;
            if (rec && alpha_tested(&d->fs))
                rec->tex[i] = [view retain], rec->samp[i] = sk;
        }

        uint32_t n = vertex_count(d->prim, d->count);
        MTLPrimitiveType mp = metal_prim(d->prim);
        if (rec)
            rec->prim = mp, rec->n = n, rec->vstart = d->vertex_start;
        if (d->prim == GFX_TRIANGLEFAN)
        {
            /* no fans in Metal: a list with the same vertices */
            uint32_t* idx = (uint32_t*)ring((size_t)n * 4, 16, &buf, &off);
            for (uint32_t i = 0; i < d->count; ++i)
            {
                uint32_t k[3] = { 0, i + 1, i + 2 };
                for (int j = 0; j < 3; ++j)
                {
                    if (!d->indices)
                        idx[3 * i + j] = d->vertex_start + k[j];
                    else if (d->index_size == 2)
                        idx[3 * i + j] = ((const uint16_t*)d->indices)[k[j]];
                    else
                        idx[3 * i + j] = ((const uint32_t*)d->indices)[k[j]];
                }
            }
            [g_enc drawIndexedPrimitives:mp indexCount:n indexType:MTLIndexTypeUInt32 indexBuffer:buf indexBufferOffset:off];
            if (rec)
                rec->ib = [buf retain], rec->ioff = off, rec->itype = 4;
        }
        else if (d->ibuf)
        {
            d->ibuf->used = g_serial;
            if (rec)
                rec->ib = [d->ibuf->b retain], rec->ioff = d->ibuf_off, rec->itype = (uint8_t)(d->index_size == 2 ? 2 : 4);
            [g_enc drawIndexedPrimitives:mp indexCount:n indexType:d->index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                             indexBuffer:d->ibuf->b indexBufferOffset:d->ibuf_off];
        }
        else if (d->indices)
        {
            void* idx = ring((size_t)n * d->index_size, 16, &buf, &off);
            memcpy(idx, d->indices, (size_t)n * d->index_size);
            [g_enc drawIndexedPrimitives:mp indexCount:n indexType:d->index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                             indexBuffer:buf indexBufferOffset:off];
            if (rec)
                rec->ib = [buf retain], rec->ioff = off, rec->itype = (uint8_t)(d->index_size == 2 ? 2 : 4);
        }
        else
            [g_enc drawPrimitives:mp vertexStart:d->vertex_start vertexCount:n];
    }
}

/* --- clears ---------------------------------------------------------------------------------------------------- */
static const char CLEAR_MSL[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct CU { float4 rect; float4 color; float z; };\n"
    "struct CO { float4 pos [[position]]; };\n"
    "vertex CO clear_vs(uint vid [[vertex_id]], constant CU& u [[buffer(0)]]) {\n"
    "  float2 c = float2((vid & 1) ? u.rect.z : u.rect.x, (vid & 2) ? u.rect.w : u.rect.y);\n"
    "  CO o; o.pos = float4(c, u.z, 1.0); return o;\n"
    "}\n"
    "fragment float4 clear_fs(constant CU& u [[buffer(0)]]) { return u.color; }\n"
    "struct PO { float4 pos [[position]]; float2 uv; };\n"
    "vertex PO present_vs(uint vid [[vertex_id]]) {\n"
    "  float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "  PO o; o.pos = float4(p * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = p; return o;\n"
    "}\n"
    /* the frame-rate overlay: a 5x7 bitmap font drawn per pixel, no texture */
    "struct OU { float4 rect; float scale; uint n; uint pad0, pad1; uint4 text[8]; };\n"
    "constant uchar FONT[17 * 7] = {\n"
    "  0x0E,0x11,0x13,0x15,0x19,0x11,0x0E, 0x04,0x0C,0x04,0x04,0x04,0x04,0x0E, 0x0E,0x11,0x01,0x02,0x04,0x08,0x1F,\n"
    "  0x1F,0x02,0x04,0x02,0x01,0x11,0x0E, 0x02,0x06,0x0A,0x12,0x1F,0x02,0x02, 0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E,\n"
    "  0x06,0x08,0x10,0x1E,0x11,0x11,0x0E, 0x1F,0x01,0x02,0x04,0x08,0x08,0x08, 0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E,\n"
    "  0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C, 0x1F,0x10,0x10,0x1E,0x10,0x10,0x10, 0x1E,0x11,0x11,0x1E,0x10,0x10,0x10,\n"
    "  0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E, 0x00,0x00,0x1A,0x15,0x15,0x11,0x11, 0x00,0x00,0x0E,0x10,0x0E,0x01,0x1E,\n"
    "  0x00,0x00,0x00,0x00,0x00,0x0C,0x0C, 0x00,0x00,0x00,0x00,0x00,0x00,0x00 };\n"
    "struct OO { float4 pos [[position]]; };\n"
    "vertex OO overlay_vs(uint vid [[vertex_id]], constant OU& u [[buffer(0)]], constant float2& size [[buffer(1)]]) {\n"
    "  float2 c = u.rect.xy + float2((vid & 1) ? u.rect.z : 0.0, (vid & 2) ? u.rect.w : 0.0);\n"
    "  OO o; o.pos = float4(c.x / size.x * 2.0 - 1.0, 1.0 - c.y / size.y * 2.0, 0, 1); return o;\n"
    "}\n"
    "fragment float4 overlay_fs(OO in [[stage_in]], constant OU& u [[buffer(0)]]) {\n"
    "  float2 p = (in.pos.xy - u.rect.xy) / u.scale - 2.0;\n"
    "  int cell = int(floor(p.x / 6.0)), gx = int(floor(p.x)) - cell * 6, gy = int(floor(p.y));\n"
    "  if (p.x >= 0.0 && cell < int(u.n) && gx < 5 && gy >= 0 && gy < 7) {\n"
    "    uint ch = u.text[cell >> 2][cell & 3];\n"
    "    if ((FONT[ch * 7 + uint(gy)] >> (4 - gx)) & 1) return float4(1.0, 0.85, 0.2, 1.0);\n"
    "  }\n"
    "  return float4(0, 0, 0, 0.55);\n"
    "}\n"
    "fragment float4 present_fs(PO in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
    "  return float4(t.sample(s, in.uv).rgb, 1.0);\n"
    "}\n"
    /* the same, sharpened by k (0..1): contrast-adaptive, a negative lobe over the four neighbors
     * that shrinks where the neighborhood is already near black or white (no halos on hard edges) */
    "fragment float4 present_cas_fs(PO in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],\n"
    "                               constant float& k [[buffer(0)]]) {\n"
    "  float2 tx = 1.0 / float2(t.get_width(), t.get_height());\n"
    "  float3 c = t.sample(s, in.uv).rgb;\n"
    "  float3 n = t.sample(s, in.uv - float2(0, tx.y)).rgb, so = t.sample(s, in.uv + float2(0, tx.y)).rgb;\n"
    "  float3 w = t.sample(s, in.uv - float2(tx.x, 0)).rgb, e = t.sample(s, in.uv + float2(tx.x, 0)).rgb;\n"
    "  float3 mn = min(c, min(min(n, so), min(w, e))), mx = max(c, max(max(n, so), max(w, e)));\n"
    "  float3 amp = sqrt(saturate(min(mn, 2.0 - mx) / max(mx, 1e-4)));\n"
    "  float3 lobe = -amp * mix(0.125, 0.2, saturate(k));\n"
    "  return float4(saturate((c + (n + so + w + e) * lobe) / (1.0 + 4.0 * lobe)), 1.0);\n"
    "}\n";

static id<MTLLibrary> g_util;

static id<MTLRenderPipelineState> clear_pipeline(uint32_t flags)
{
    uint32_t k[4] = { (uint32_t)g_rt->tex.pixelFormat, 0, 0, flags & 1 };
    id<MTLTexture> depth = depth_attachment();
    k[1] = depth ? (uint32_t)depth.pixelFormat : 0;
    k[2] = depth && g_ds->has_stencil ? k[1] : 0;
    id p = map_get(&g_clear_pipes, k, sizeof k);
    if (p)
        return p;
    MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
    id<MTLFunction> vf = [g_util newFunctionWithName:@"clear_vs"], ff = [g_util newFunctionWithName:@"clear_fs"];
    pd.vertexFunction = vf;
    pd.fragmentFunction = ff;
    pd.colorAttachments[0].pixelFormat = (MTLPixelFormat)k[0];
    pd.colorAttachments[0].writeMask = (flags & 1) ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
    pd.depthAttachmentPixelFormat = (MTLPixelFormat)k[1];
    pd.stencilAttachmentPixelFormat = (MTLPixelFormat)k[2];
    p = [g_dev newRenderPipelineStateWithDescriptor:pd error:NULL];
    [vf release];
    [ff release];
    [pd release];
    map_put(&g_clear_pipes, k, sizeof k, p);
    return p;
}

void gfx_clear(uint32_t nrects, const int32_t* rects, uint32_t flags, uint32_t color, float z, uint32_t stencil, const uint32_t vp[6])
{
    if (!g_dev || !g_rt)
        return;
    if (!g_ds)
        flags &= 1;
    if (!flags)
        return;
    @autoreleasepool
    {
        float c[4] = { ((color >> 16) & 255) / 255.0f, ((color >> 8) & 255) / 255.0f, (color & 255) / 255.0f, (color >> 24) / 255.0f };
        uint32_t w, h;
        color_size(&w, &h);
        int whole = !nrects && vp[0] == 0 && vp[1] == 0 && vp[2] >= w && vp[3] >= h;
        if (whole)
        {
            /* the next pass starts cleared: close this one (what it drew to cleared attachments is
             * overwritten anyway) */
            end_pass();
            g_pending_clear |= flags;
            if (flags & 1)
                memcpy(g_clear_color, c, sizeof c);
            if (flags & 2)
                g_clear_z = z;
            if (flags & 4)
                g_clear_stencil = stencil;
            return;
        }
        if (!begin_pass())
            return;
        /* the viewport, intersected with each rectangle, as a quad at depth z */
        int32_t vx0 = (int32_t)vp[0], vy0 = (int32_t)vp[1], vx1 = vx0 + (int32_t)vp[2], vy1 = vy0 + (int32_t)vp[3];
        int32_t whole_rect[4] = { vx0, vy0, vx1, vy1 };
        if (!nrects)
            rects = whole_rect, nrects = 1;
        [g_enc setRenderPipelineState:clear_pipeline(flags)];
        GfxDepthKey dk;
        memset(&dk, 0, sizeof dk);
        dk.zenable = (flags & 2) != 0, dk.zwrite = 1, dk.zfunc = 8;
        if (flags & 4)
            dk.stencil = 1, dk.sfunc = 8, dk.sfail = dk.szfail = dk.spass = 3, dk.sread = dk.swrite = 0xFF;
        [g_enc setDepthStencilState:depth_state(&dk)];
        [g_enc setStencilReferenceValue:stencil];
        [g_enc setCullMode:MTLCullModeNone];
        [g_enc setTriangleFillMode:MTLTriangleFillModeFill];
        [g_enc setDepthBias:0 slopeScale:0 clamp:0];
        [g_enc setViewport:(MTLViewport){ 0, 0, w, h, 0, 1 }];
        for (uint32_t i = 0; i < nrects; ++i)
        {
            int32_t x0 = rects[4 * i] > vx0 ? rects[4 * i] : vx0, y0 = rects[4 * i + 1] > vy0 ? rects[4 * i + 1] : vy0;
            int32_t x1 = rects[4 * i + 2] < vx1 ? rects[4 * i + 2] : vx1, y1 = rects[4 * i + 3] < vy1 ? rects[4 * i + 3] : vy1;
            if (x1 <= x0 || y1 <= y0)
                continue;
            struct
            {
                float rect[4], color[4], z, pad[3];
            } cu = { { x0 * 2.0f / w - 1, 1 - y0 * 2.0f / h, x1 * 2.0f / w - 1, 1 - y1 * 2.0f / h }, { c[0], c[1], c[2], c[3] }, z, { 0 } };
            [g_enc setVertexBytes:&cu length:sizeof cu atIndex:0];
            [g_enc setFragmentBytes:&cu length:sizeof cu atIndex:0];
            [g_enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        }
    }
}

/* --- scene effects (gfx_scene_done) -----------------------------------------------------------------------------
 * On the finished 3D scene, in place, when fx is on:
 *   - ambient occlusion: view-space positions rebuilt from the scene's depth and projection, a spiral
 *     of samples around each pixel (Scalable Ambient Obscurance, simplified) at about 1000 pixels
 *     across, then a depth-aware blur across and down;
 *   - sun shadows, in the same pass and blur: each pixel looked up in the sun's shadow map (the
 *     scene's casters drawn again from the sun, sun_map), and a short ray toward the sun through the
 *     depth buffer for contact shadows where things meet the ground; by day;
 *   - height fog: the density falls off with world height (up from the view matrix), integrated along
 *     each view ray, so low ground mists over while hills stand clear; lit by the sun when looking
 *     toward it (Henyey-Greenstein scattering);
 *   - bloom: the bright parts, at a quarter and an eighth of the scene, blurred and added back;
 *   - god rays: the sky's bright pixels around the sun, blurred along lines toward the sun's place
 *     on screen (the sun is the scene's directional light);
 *   - a color grade (saturation, a contrast curve).
 * Then the scene gets mips, and the draw that makes it smaller for the back buffer samples it
 * through them (draw_encode): FFXI's background is larger than the screen, and one bilinear sample
 * per screen pixel skips rows of it - the edges crawl as the camera moves.
 *
 * Settings: FFXI_FX=1 and FFXI_FX_<KEY> (the table in fx_config), then while the game runs
 * FFXI_FX_FILE (default ~/Library/Caches/FFXI/fx.txt), lines of key=value. debug shows one part
 * alone: 1 occlusion, 2 fog, 3 bloom, 4 god rays, 5 sun shadows (map and contact), 6 the ray-traced scene
 * (rt_frame). light = 1 lights the world's
 * lit draws per pixel rather than per vertex (gfx_msl.c). */
static const char FX_MSL[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct FxU {\n"
    "  float4 proj;   // P00, P11, P20, P21\n"
    "  float4 zp;     // P22, P32, viewport MinZ, MaxZ\n"
    "  float4 vp;     // the scene viewport in target pixels: x, y, width, height\n"
    "  float4 size;   // target width, height; occlusion width, height\n"
    "  float4 ao;     // radius, strength, bias, largest radius in pixels\n"
    "  float4 grade;  // strength, saturation, contrast, debug\n"
    "  float4 hand;   // P23: 1 for a left-handed projection, -1 for a right-handed one\n"
    "  float4 up;     // world up in view space: height above the camera = dot(P, up.xyz)\n"
    "  float4 sun;    // view space, toward the light; w = 1 when the scene has one\n"
    "  float4 suncol; // its color\n"
    "  float4 sunuv;  // its place in the viewport (0..1), z = how much of it shows (0 behind the camera)\n"
    "  float4 fogc;   // fog color, a = density at the camera's height\n"
    "  float4 fogp;   // falloff with height, most fog, sun glow, its forward scattering (g)\n"
    "  float4 bloom;  // threshold, strength, knee\n"
    "  float4 rays;   // strength, decay, length\n"
    "  float4 shadow; // contact shadows: strength (0: none, or night), ray length, thickness, how far out\n"
    "  float4x4 lmat; // view space to the sun's map: x, y -1..1 (y up), z 0..1\n"
    "  float4 smap;   // the map: strength (0: none), a texel in world units, depth bias, penumbra (map width per depth unit)\n"
    "  float4 smap2;  // depth units per map width, sun_face, sun_min, world units per depth unit\n"
    "  float4x4 reproj; // view space to the frame before's clip space\n"
    "  float4 hist;   // 1 when the frame before is there to blend with, the pattern's turn this frame, its weight\n"
    "  float4x4 lmatn; // the near cascade: view space to its map\n"
    "  float4 smapn;  // its texel in world units, depth bias, penumbra, slope\n"
    "  float4 smapn2; // its depth units, 1 when it is there\n"
    "  float4x4 vw;   // view space to the world (the ray tracing's)\n"
    "  float4x4 ivp;  // clip space to the world less the camera's place (the rays')\n"
    "  float4 cam;    // the camera's place in the world (the rays')\n"
    "  float4 sunw;   // toward the sun (or moon) in the world; w = 1 when the scene has one\n"
    "};\n"
    "struct FO { float4 pos [[position]]; float2 uv; };\n"
    "vertex FO fx_vs(uint vid [[vertex_id]]) {\n"
    "  float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "  FO o; o.pos = float4(p * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = p; return o;\n"
    "}\n"
    "constant float3 LUMA = float3(0.2126, 0.7152, 0.0722);\n"
    /* view-space z of a depth value (clip w = z * P23, so z is negative in front of a right-handed
     * camera); 0 for the far plane (the sky, cleared depth) */
    "static float view_z(constant FxU& u, float d) {\n"
    "  d = (d - u.zp.z) / max(u.zp.w - u.zp.z, 1e-6);\n"
    "  float z = u.zp.y / (d * u.hand.x - u.zp.x);\n"
    "  return d >= 0.999999 || !(z * u.hand.x > 0.0) ? 0.0 : z;\n"
    "}\n"
    "static float3 view_pos(constant FxU& u, float2 px, float z) {\n"
    "  float2 ndc = float2((px.x - u.vp.x) / u.vp.z * 2.0 - 1.0, 1.0 - (px.y - u.vp.y) / u.vp.w * 2.0);\n"
    "  float w = z * u.hand.x;\n"
    "  return float3((ndc.x * w - u.proj.z * z) / u.proj.x, (ndc.y * w - u.proj.w * z) / u.proj.y, z);\n"
    "}\n"
    "static float3 pos_at(constant FxU& u, depth2d<float> dt, float2 px) {\n"
    "  px = clamp(px, u.vp.xy, u.vp.xy + u.vp.zw - 1.0);\n"
    "  px = floor(px) + 0.5;\n"
    "  return view_pos(u, px, view_z(u, dt.read(uint2(px))));\n"
    "}\n"
    /* in the sun (1) or not (0): a ray from P toward the sun, stepped through the depth buffer; a
     * surface in front of it (by less than the thickness: what is far nearer the camera hides the
     * ray, it does not block it) shades P. Only on surfaces that face the sun (the rest are dark from
     * their own lighting), and fading out with distance, where the steps grow coarse. */
    "static float sun_shadow(constant FxU& u, depth2d<float> dt, float3 P, float3 N, float dist, float k) {\n"
    "  float nl = dot(N, u.sun.xyz);\n"
    "  float fade = smoothstep(0.0, 0.15, nl) * (1.0 - smoothstep(0.6 * u.shadow.w, u.shadow.w, dist));\n"
    "  if (fade <= 0.0) return 1.0;\n"
    "  const int NS = 16;\n"
    "  float3 O = P + N * (0.01 * dist);\n"
    "  for (int i = 0; i < NS; ++i) {\n"
    "    float a = (float(i) + k) / float(NS);\n"
    "    float3 R = O + u.sun.xyz * (u.shadow.y * a * a);\n"
    "    float rd = R.z * u.hand.x;\n"
    "    if (rd <= 0.05) break;\n"
    "    float2 ndc = float2(R.x * u.proj.x + R.z * u.proj.z, R.y * u.proj.y + R.z * u.proj.w) / rd;\n"
    "    float2 q = u.vp.xy + float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * u.vp.zw;\n"
    "    if (any(q < u.vp.xy) || any(q >= u.vp.xy + u.vp.zw)) break;\n"
    "    float sd = view_z(u, dt.read(uint2(q))) * u.hand.x;\n"
    "    float in_front = rd - sd;\n"
    /* weaker the farther along the hit: no hard edge where the ray ends */
    "    if (sd > 0.0 && in_front > 0.005 * rd + 0.02 && in_front < u.shadow.z + 0.01 * rd) return 1.0 - fade * (1.0 - a * a);\n"
    "  }\n"
    "  return 1.0;\n"
    "}\n"
    /* in the sun (1) or not (0) by the shadow map, softer the farther the shadow falls from what casts
     * it (percentage-closer soft shadows): the casters' average depth around the point sets how wide
     * the filter is; sixteen samples on a disk turned by the pixel's step of the 4x4 pattern, which
     * the blur then averages. P is moved off its surface by a texel and more with distance (no acne).
     * Surfaces turned away from the sun shade gently by their angle alone - the map's own edges on
     * them are the facets of low-polygon rock. */
    "constant float2 DISK[16] = { float2(-0.94, -0.40), float2(0.95, -0.77), float2(-0.09, -0.93), float2(0.34, 0.29),\n"
    "  float2(-0.92, 0.46), float2(-0.82, -0.88), float2(-0.38, 0.28), float2(0.97, 0.76), float2(0.44, -0.98),\n"
    "  float2(0.54, -0.47), float2(-0.26, -0.42), float2(-0.42, 0.87), float2(0.31, 0.92), float2(0.79, 0.19),\n"
    "  float2(-0.03, 0.04), float2(0.15, -0.33) };\n"
    /* one cascade: p = its texel (world units), depth bias, penumbra (map width per depth unit),
     * slope (depth units per map width); du its depth units; edge how far inside it P is (0 out) */
    "static float sun_look(depth2d<float> sm, sampler cmp, float4x4 lm, float4 p, float du, float minw, float3 P,\n"
    "                      float3 N, float dist, float k, thread float& edge) {\n"
    "  float3 Q = P + N * (1.5 * p.x + 0.002 * dist);\n"
    "  float4 lc = lm * float4(Q, 1.0);\n"
    "  float2 uv = float2(lc.x * 0.5 + 0.5, 0.5 - lc.y * 0.5);\n"
    "  float2 e = abs(lc.xy);\n"
    "  edge = lc.z >= 1.0 ? 0.0 : 1.0 - smoothstep(0.8, 0.95, max(e.x, e.y));\n"
    "  if (edge <= 0.0) return 1.0;\n"
    "  float z = lc.z - p.y, sz = float(sm.get_width()), tx = 1.0 / sz;\n"
    "  float a = k * 6.2831853, ca = cos(a), sa = sin(a);\n"
    "  float2x2 rot = float2x2(float2(ca, sa), float2(-sa, ca));\n"
    "  float bs = 0.0, bn = 0.0;\n"
    "  for (int i = 0; i < 16; ++i) {\n"
    "    float2 q = clamp((uv + DISK[i] * (32.0 * tx)) * sz, 0.0, sz - 1.0);\n"
    "    float d = sm.read(uint2(q));\n"
    "    if (d < z) bs += d, bn += 1.0;\n"
    "  }\n"
    "  if (bn == 0.0) return 1.0;\n"
    /* the penumbra: wider the farther the caster, never narrower than 2 cm (no hard pixel edge) */
    "  float pen = clamp((z - bs / bn) * p.z, max(1.5 * tx, 0.02 / (p.x * sz)), 32.0 * tx);\n"
    /* casters nearer the surface than sun_min are its own neighbouring faces (a ledge of the same
     * rock): no shadow from them */
    "  float near = smoothstep(0.5 * minw, 1.5 * minw, (z - bs / bn) * du);\n"
    /* a wider filter reaches farther across the surface, to points of it nearer the sun: the bias
     * grows with it (a slope of one) */
    "  float zc = z - pen * p.w, s = 0.0;\n"
    "  for (int i = 0; i < 16; ++i) s += sm.sample_compare(cmp, uv + rot * DISK[i] * pen, zc);\n"
    "  return mix(1.0, s / 16.0, near);\n"
    "}\n"
    /* in the sun (1) or not (0) by the maps: the near cascade where P is in it, the far one beyond,
     * blended across the near one's edge */
    "static float sun_map(constant FxU& u, depth2d<float> sm, depth2d<float> smn, sampler cmp, float3 P, float3 N,\n"
    "                     float dist, float k) {\n"
    "  float nl = dot(N, u.sun.xyz);\n"
    /* turned from the sun: shaded by the angle only as much as sun_face asks (FFXI's own lighting
     * shades its rock already; per face, low-polygon rock turns into a patchwork) */
    "  float face = mix(1.0 - 0.6 * u.smap2.y, 1.0, smoothstep(-0.3, 0.25, nl)), use = smoothstep(-0.05, 0.15, nl);\n"
    "  if (use <= 0.0) return face;\n"
    "  float en = 0.0, ef = 0.0, s = 1.0;\n"
    "  if (u.smapn2.y > 0.0)\n"
    "    s = sun_look(smn, cmp, u.lmatn, u.smapn, u.smapn2.x, u.smap2.z, P, N, dist, k, en);\n"
    "  if (en < 1.0) {\n"
    "    float sf = sun_look(sm, cmp, u.lmat, float4(u.smap.yzw, u.smap2.x), u.smap2.w, u.smap2.z, P, N, dist, k, ef);\n"
    "    s = mix(sf, s, en);\n"
    "  }\n"
    "  return mix(1.0, min(mix(1.0, s, use), face), max(en, ef));\n"
    "}\n"
    /* occlusion in x (1 open, 0 closed), distance in y (0: sky), the sun by the map in z and by
     * contact in w (1 lit, 0 shaded) */
    "fragment float4 fx_ao(FO in [[stage_in]], constant FxU& u [[buffer(0)]], depth2d<float> dt [[texture(0)]],\n"
    "                      depth2d<float> sm [[texture(1)]], depth2d<float> smn [[texture(2)]], sampler cmp [[sampler(1)]]) {\n"
    "  float2 px = floor(u.vp.xy + in.uv * u.vp.zw) + 0.5;\n"
    "  float3 P = pos_at(u, dt, px);\n"
    "  float dist = P.z * u.hand.x;\n"
    "  if (dist <= 0.0) return float4(1.0, 0.0, 1.0, 1.0);\n"
    /* the surface normal from the neighbors on the side nearer in depth (no smearing across edges) */
    "  float3 r = pos_at(u, dt, px + float2(1, 0)) - P, l = P - pos_at(u, dt, px - float2(1, 0));\n"
    "  float3 d = pos_at(u, dt, px + float2(0, 1)) - P, t = P - pos_at(u, dt, px - float2(0, 1));\n"
    "  float3 dx = abs(r.z) < abs(l.z) ? r : l, dy = abs(d.z) < abs(t.z) ? d : t;\n"
    "  float3 N = normalize(cross(dx, dy));\n"
    "  if (dot(N, P) > 0.0) N = -N;\n"
    "  const uchar BAYER[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };\n"
    "  int2 cell = int2(in.pos.xy) & 3;\n"
    "  float k = fract((float(BAYER[cell.y * 4 + cell.x]) + 0.5) / 16.0 + u.hist.y);\n"
    "  float sh = u.shadow.x > 0.0 && u.sun.w > 0.0 ? sun_shadow(u, dt, P, N, dist, k) : 1.0;\n"
    "  float mp = u.smap.x > 0.0 && u.sun.w > 0.0 ? sun_map(u, sm, smn, cmp, P, N, dist, k) : 1.0;\n"
    "  float rad = u.ao.x, rpx = min(rad * u.proj.y * 0.5 * u.vp.w / dist, u.ao.w);\n"
    "  if (u.ao.y <= 0.0 || rpx < 2.0) return float4(1.0, dist, mp, sh);\n"
    /* the spiral turned and scaled by one of 16 steps, a 4x4 ordered pattern over the pixels, the
     * same every frame; the blur averages exactly one 4x4 block (fx_blur), so each pixel ends up with
     * all 16 - even, and with no grain left to crawl as the camera moves. One pattern at every pixel
     * instead copies each occluder at the pattern's offsets: streaks and halos around characters. */
    "  const int NS = 20;\n"
    "  float sum = 0.0;\n"
    "  for (int i = 0; i < NS; ++i) {\n"
    "    float a = (float(i) + k) / float(NS);\n"
    "    float ang = float(i) * 2.3999632 + k * 6.2831853; /* the golden angle: an even spiral */\n"
    "    float2 q = px + float2(cos(ang), sin(ang)) * (a * rpx);\n"
    /* off the screen: nothing known there (clamped to the edge, the edge's own pixels would shade
     * it, and move with the camera) */
    "    if (any(q < u.vp.xy) || any(q >= u.vp.xy + u.vp.zw)) continue;\n"
    "    float3 Q = pos_at(u, dt, q);\n"
    /* a surface far nearer the camera floats in front of this one (a leg before the floor or the
     * other leg): it hides it, it does not shade it */
    "    if (Q.z == 0.0 || dist - Q.z * u.hand.x > 0.5 * rad) continue;\n"
    "    float3 v = Q - P;\n"
    "    float vv = dot(v, v), vn = dot(v, N);\n"
    "    float q2 = vv / (rad * rad), fall = saturate(1.0 - q2 * q2);\n"
    "    sum += fall * max(vn * rsqrt(vv + 1e-6) - u.ao.z, 0.0);\n"
    "  }\n"
    /* none on surfaces seen edge-on: there the normal from depth is unreliable, and the surface
     * shades itself in bands along every silhouette */
    "  float facing = smoothstep(0.1, 0.4, dot(N, -P) / dist);\n"
    "  return float4(saturate(1.0 - 3.0 * facing * sum / float(NS)), dist, mp, sh);\n"
    "}\n"
    /* the occlusion (x) and sun (y) at a scene pixel from the four nearest occlusion texels, each
     * weighted by how near its distance is to this pixel's: an edge's occlusion stays on its own
     * side, however the low-resolution grid falls on it */
    "static float3 ao_at(constant FxU& u, texture2d<float> ao, float2 uv, float dist) {\n"
    "  if (dist <= 0.0) return float3(1.0);\n"
    "  float2 g = uv * u.size.zw - 0.5, f = fract(g);\n"
    "  int2 i0 = int2(floor(g)), hi = int2(u.size.zw) - 1;\n"
    "  float3 s = 0.0;\n"
    "  float w = 0.0;\n"
    "  for (int k = 0; k < 4; ++k) {\n"
    "    int2 o = int2(k & 1, k >> 1);\n"
    "    float4 t = ao.read(uint2(clamp(i0 + o, int2(0), hi)));\n"
    "    float bw = (o.x ? f.x : 1.0 - f.x) * (o.y ? f.y : 1.0 - f.y);\n"
    "    float dw = t.y > 0.0 ? 1.0 / (1e-3 + abs(t.y - dist) / dist) : 1e-3;\n"
    "    s += t.xzw * bw * dw, w += bw * dw;\n"
    "  }\n"
    "  return w > 0.0 ? s / w : float3(1.0);\n"
    "}\n"
    "fragment float4 fx_blur(FO in [[stage_in]], constant FxU& u [[buffer(0)]], constant int2& dir [[buffer(1)]],\n"
    "                        texture2d<float> a [[texture(0)]]) {\n"
    "  int2 p = int2(in.pos.xy), hi = int2(u.size.zw) - 1;\n"
    "  float4 c = a.read(uint2(p));\n"
    "  if (c.y <= 0.0) return c;\n"
    /* four pixels' worth, centered (the ends at half weight): one period of the 4x4 pattern */
    /* the occlusion over four pixels; the shadows only over their neighbours (their softness is
     * the maps' own, and the temporal pass evens their noise) */
    "  float3 s = c.xzw;\n"
    "  float w = 1.0, sw = 1.0;\n"
    "  float2 ss = c.zw;\n"
    "  for (int i = -2; i <= 2; ++i) {\n"
    "    if (i == 0) continue;\n"
    "    float4 t = a.read(uint2(clamp(p + dir * i, int2(0), hi)));\n"
    "    float k = (abs(i) == 2 ? 0.5 : 1.0) * saturate(1.0 - abs(t.y - c.y) / (0.03 * c.y));\n"
    "    s.x += t.x * k, w += k;\n"
    "    if (abs(i) == 1) ss += t.zw * (0.5 * k), sw += 0.5 * k;\n"
    "  }\n"
    "  return float4(s.x / w, c.y, ss / sw);\n"
    "}\n"
    /* the occlusion and shadows over time: each texel's world point found in the frame before (its
     * camera), and that frame's result blended in where its distance agrees (no smearing across what
     * moved or came into view). The sampling patterns turn every frame, so this averages them into
     * a result that holds still as the camera moves, and a frame gone wrong fades rather than
     * flashes. */
    "fragment float4 fx_temporal(FO in [[stage_in]], constant FxU& u [[buffer(0)]], texture2d<float> cur [[texture(0)]],\n"
    "                            texture2d<float> hist [[texture(1)]], sampler s [[sampler(0)]]) {\n"
    "  float4 c = cur.read(uint2(in.pos.xy));\n"
    "  if (c.y <= 0.0 || u.hist.x == 0.0) return c;\n"
    "  float2 px = u.vp.xy + in.uv * u.vp.zw;\n"
    "  float3 P = view_pos(u, px, c.y * u.hand.x);\n"
    "  float4 pc = u.reproj * float4(P, 1.0);\n"
    "  if (pc.w <= 1e-4) return c;\n"
    "  float2 puv = float2(pc.x / pc.w * 0.5 + 0.5, 0.5 - pc.y / pc.w * 0.5);\n"
    "  if (any(puv < 0.0) || any(puv > 1.0)) return c;\n"
    "  float4 h = hist.sample(s, puv);\n"
    "  if (!(h.y > 0.0) || abs(h.y - pc.w) > 0.04 * pc.w) return c;\n"
    "  float3 m = mix(c.xzw, h.xzw, u.hist.z);\n"
    "  return float4(m.x, c.y, m.y, m.z);\n"
    "}\n"
    /* bloom's source: the scene at a quarter size (four bilinear taps), what is over the threshold,
     * with a soft knee */
    "fragment float4 fx_bright(FO in [[stage_in]], constant FxU& u [[buffer(0)]], texture2d<float> src [[texture(0)]],\n"
    "                          sampler s [[sampler(0)]]) {\n"
    "  float2 uv = (u.vp.xy + in.uv * u.vp.zw) / u.size.xy, t = 1.0 / u.size.xy;\n"
    "  float3 c = 0.25 * (src.sample(s, uv + t * float2(-1, -1)).rgb + src.sample(s, uv + t * float2(1, -1)).rgb +\n"
    "                     src.sample(s, uv + t * float2(-1, 1)).rgb + src.sample(s, uv + t * float2(1, 1)).rgb);\n"
    "  float l = max(c.r, max(c.g, c.b)), k = u.bloom.z;\n"
    "  float soft = clamp(l - u.bloom.x + k, 0.0, 2.0 * k);\n"
    "  soft = soft * soft / (4.0 * k + 1e-5);\n"
    "  return float4(c * (max(soft, l - u.bloom.x) / max(l, 1e-5)), 1.0);\n"
    "}\n"
    /* half the size of the source: four bilinear taps */
    "fragment float4 fx_down(FO in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
    "  float2 tx = 1.0 / float2(t.get_width(), t.get_height());\n"
    "  return 0.25 * (t.sample(s, in.uv + tx * float2(-1, -1)) + t.sample(s, in.uv + tx * float2(1, -1)) +\n"
    "                 t.sample(s, in.uv + tx * float2(-1, 1)) + t.sample(s, in.uv + tx * float2(1, 1)));\n"
    "}\n"
    /* a 9-tap gaussian in five bilinear taps, dir texels apart */
    "fragment float4 fx_gauss(FO in [[stage_in]], constant int2& dir [[buffer(1)]], texture2d<float> t [[texture(0)]],\n"
    "                         sampler s [[sampler(0)]]) {\n"
    "  float2 tx = float2(dir) / float2(t.get_width(), t.get_height());\n"
    "  float4 c = t.sample(s, in.uv) * 0.2270270;\n"
    "  c += (t.sample(s, in.uv + tx * 1.3846154) + t.sample(s, in.uv - tx * 1.3846154)) * 0.3162162;\n"
    "  c += (t.sample(s, in.uv + tx * 3.2307692) + t.sample(s, in.uv - tx * 3.2307692)) * 0.0702703;\n"
    "  return c;\n"
    "}\n"
    /* what the god rays start from: the sky's bright pixels, more of them nearer the sun */
    "fragment float4 fx_raymask(FO in [[stage_in]], constant FxU& u [[buffer(0)]], texture2d<float> src [[texture(0)]],\n"
    "                           depth2d<float> dt [[texture(1)]], sampler s [[sampler(0)]]) {\n"
    "  float2 px = floor(u.vp.xy + in.uv * u.vp.zw) + 0.5;\n"
    "  if (view_z(u, dt.read(uint2(px))) != 0.0) return float4(0.0);\n"
    "  float3 c = src.sample(s, px / u.size.xy).rgb;\n"
    "  float2 d = (in.uv - u.sunuv.xy) * float2(u.proj.y / u.proj.x, 1.0);\n"
    "  float glow = saturate(1.0 - length(d) / 0.6);\n"
    "  return float4(c * smoothstep(0.35, 0.9, dot(c, LUMA)) * glow * glow, 1.0);\n"
    "}\n"
    /* the mask gathered along the line toward the sun, fading with each step (no dither: it would
     * shimmer from frame to frame) */
    "fragment float4 fx_rays(FO in [[stage_in]], constant FxU& u [[buffer(0)]], texture2d<float> m [[texture(0)]],\n"
    "                        sampler s [[sampler(0)]]) {\n"
    "  const int NS = 64;\n"
    "  float2 uv = in.uv, step = (in.uv - u.sunuv.xy) * (u.rays.z / float(NS));\n"
    "  float3 acc = float3(0.0);\n"
    "  float w = 1.0;\n"
    "  for (int i = 0; i < NS; ++i) { acc += m.sample(s, uv).rgb * w; w *= u.rays.y; uv -= step; }\n"
    "  return float4(acc * (4.0 / float(NS)), 1.0);\n"
    "}\n"
    "static float3 screen(float3 a, float3 b) { return 1.0 - (1.0 - saturate(a)) * (1.0 - saturate(b)); }\n"
    "fragment float4 fx_comp(FO in [[stage_in]], constant FxU& u [[buffer(0)]], texture2d<float> src [[texture(0)]],\n"
    "                        texture2d<float> ao [[texture(1)]], depth2d<float> dt [[texture(2)]],\n"
    "                        texture2d<float> b1 [[texture(3)]], texture2d<float> b2 [[texture(4)]],\n"
    "                        texture2d<float> ry [[texture(5)]], texture2d<float> rt [[texture(6)]],\n"
    "                        sampler s [[sampler(0)]]) {\n"
    "  float2 px = in.pos.xy;\n"
    "  float4 c = src.read(uint2(px));\n"
    "  int dbg = int(u.grade.w);\n"
    "  float3 os = u.ao.y > 0.0 || u.shadow.x > 0.0 || u.smap.x > 0.0 ? ao_at(u, ao, in.uv, view_z(u, dt.read(uint2(px))) * u.hand.x) : float3(1.0);\n"
    "  float o = os.x, sun = mix(1.0, os.y, u.smap.x) * mix(1.0, os.z, u.shadow.x);\n"
    "  if (dbg == 1) return float4(o, o, o, c.a);\n"
    "  if (dbg == 5) return float4(float3(sun), c.a);\n"
    "  if (dbg == 6) return float4(rt.sample(s, in.uv).rgb, c.a);\n"
    "  c.rgb *= mix(1.0, o, u.ao.y) * sun;\n"
    "  float f = 0.0;\n"
    "  if (u.fogc.a > 0.0) {\n"
    "    float z = view_z(u, dt.read(uint2(px)));\n"
    "    if (z != 0.0) {\n"
    "      float3 P = view_pos(u, px, z);\n"
    "      float d = length(P), bd = u.fogp.x * dot(P, u.up.xyz);\n"
    /* density a * exp(-b * height above the camera), integrated from the camera to P */
    "      float k = abs(bd) > 1e-4 ? (1.0 - exp(-bd)) / bd : 1.0;\n"
    "      f = min(1.0 - exp(-u.fogc.a * d * k), u.fogp.y);\n"
    /* the sun's light scattered toward the camera (Henyey-Greenstein, 1 looking straight at it) */
    "      float g = u.fogp.w, cs = dot(P / max(d, 1e-5), u.sun.xyz);\n"
    "      float sunk = u.sun.w * u.fogp.z * pow((1.0 - g) * (1.0 - g) / max(1.0 + g * g - 2.0 * g * cs, 1e-5), 1.5);\n"
    "      c.rgb = mix(c.rgb, u.fogc.rgb + u.suncol.rgb * sunk, f);\n"
    "    }\n"
    "  }\n"
    "  if (dbg == 2) return float4(f, f, f, c.a);\n"
    "  float3 add = float3(0.0);\n"
    "  if (u.bloom.y > 0.0) {\n"
    "    float3 bl = b1.sample(s, in.uv).rgb * 0.6 + b2.sample(s, in.uv).rgb * 0.8;\n"
    "    if (dbg == 3) return float4(bl, c.a);\n"
    "    add += bl * u.bloom.y;\n"
    "  }\n"
    "  if (u.rays.x > 0.0 && u.sunuv.z > 0.0) {\n"
    "    float3 r = ry.sample(s, in.uv).rgb * u.suncol.rgb * u.sunuv.z;\n"
    "    if (dbg == 4) return float4(r, c.a);\n"
    "    add += r * u.rays.x;\n"
    "  }\n"
    "  c.rgb = screen(c.rgb, add);\n"
    "  float3 x = mix(float3(dot(c.rgb, LUMA)), c.rgb, u.grade.y);\n"
    "  x = saturate(x);\n"
    "  x = mix(x, x * x * (3.0 - 2.0 * x), u.grade.z);\n"
    "  c.rgb = mix(c.rgb, x, u.grade.x);\n"
    "  return c;\n"
    "}\n";

typedef struct FxU
{
    float proj[4], zp[4], vp[4], size[4], ao[4], grade[4], hand[4], up[4], sun[4], suncol[4], sunuv[4], fogc[4], fogp[4],
        bloom[4], rays[4], shadow[4], lmat[16], smap[4], smap2[4], reproj[16], hist[4], lmatn[16], smapn[4], smapn2[4], vw[16], ivp[16], cam[4], sunw[4];
} FxU;

static struct
{
    int tried;
    id<MTLLibrary> lib;
    id<MTLRenderPipelineState> ao_pipe, blur_pipe, bright_pipe, down_pipe, gauss_pipe, raymask_pipe, rays_pipe, comp_pipe,
        temporal_pipe;
    MTLPixelFormat comp_fmt;
    id<MTLTexture> src, ao0, ao1, b1a, b1b, b2a, b2b, ra, rb;
    id<MTLTexture> hist[2]; /* the occlusion and shadows after the temporal pass: this frame's and the one before */
    int hist_at;            /* which of hist[] the frame before wrote */
    uint64_t hist_serial;   /* the frame it was written (0: none) */
    float prev_view[16], prev_proj[16], prev_cam[3];
    id<MTLSamplerState> samp, cmp;
    id<MTLTexture> smap, smapn, scol, sdummy; /* the sun's shadow maps (far, near), their (memoryless) color, a stand-in */
    id<MTLDepthStencilState> sdepth;
    /* what the fog and rays follow, eased from frame to frame (fx_ease): the game's values can
     * change between frames, and the effects should not pop with them */
    int eased;           /* ease (the last scene was the frame before); else take the new values */
    uint64_t eased_serial;
    float fog_on, fogc[3], up[3], sun[3], suncol[3];
    /* toward the sun in the world, as the last lit draw gave it, and the frame it was seen */
    float sunw[3];
    /* how much of the game's light is the sun's (its diffuse against the ambient), eased: weather and
     * clouds dim it, and the shadows fade with it */
    float direct;
    uint64_t sunw_seen;
    /* the shadows' profile (FFXI_PROFILE): frames, frames with their own sun, with a map, the
     * fewest and most casters */
    uint32_t st_frames, st_own, st_map, st_cmin, st_cmax, st_drawn_this, st_cached;
    float st_across;
    /* the trace (gfx_trace_dump): each scene done */
    uint32_t tr_live, tr_cached, tr_skipped, tr_depth;
    float tr_strength, tr_day, tr_dl, tr_al, tr_fog[3];
    int fogc_set;
    struct { uint64_t serial; uint32_t live, cached, skipped, own, depth; float fog[3], dl, al, strength, day, sun[3], cam[3], across; } trace[1200];
    uint32_t ntrace;
} g_fx;

/* a toward b by k; the first time, b */
static void fx_ease(float* a, const float* b, int n, float k)
{
    for (int i = 0; i < n; ++i)
        a[i] = g_fx.eased ? a[i] + (b[i] - a[i]) * k : b[i];
}

static void normalize3(float* v)
{
    float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 0.0f)
        v[0] /= l, v[1] /= l, v[2] /= l;
}

/* every setting: its key in the file, FFXI_FX_<KEY> in the environment, and its default */
static const struct
{
    const char* key;
    size_t at;
    float def;
} FX_SETTINGS[] = {
    { "fx", offsetof(__typeof__(g_fxs), fx), 0.0f },
    { "ao", offsetof(__typeof__(g_fxs), ao), 0.8f },
    { "radius", offsetof(__typeof__(g_fxs), radius), 1.0f },
    { "grade", offsetof(__typeof__(g_fxs), grade), 1.0f },
    { "sat", offsetof(__typeof__(g_fxs), sat), 1.12f },
    { "contrast", offsetof(__typeof__(g_fxs), contrast), 0.2f },
    { "sharpen", offsetof(__typeof__(g_fxs), sharpen), 0.3f },
    { "filter", offsetof(__typeof__(g_fxs), filter), 1.0f },
    { "aniso", offsetof(__typeof__(g_fxs), aniso), 16.0f },
    { "fog", offsetof(__typeof__(g_fxs), fog), 0.004f },
    { "fog_falloff", offsetof(__typeof__(g_fxs), fog_falloff), 0.08f },
    { "fog_height", offsetof(__typeof__(g_fxs), fog_height), 2.0f },
    { "fog_max", offsetof(__typeof__(g_fxs), fog_max), 0.5f },
    { "fog_sun", offsetof(__typeof__(g_fxs), fog_sun), 0.5f },
    { "fog_g", offsetof(__typeof__(g_fxs), fog_g), 0.6f },
    { "bloom", offsetof(__typeof__(g_fxs), bloom), 0.3f },
    { "threshold", offsetof(__typeof__(g_fxs), threshold), 0.75f },
    { "rays", offsetof(__typeof__(g_fxs), rays), 0.6f },
    { "rays_decay", offsetof(__typeof__(g_fxs), rays_decay), 0.965f },
    { "rays_length", offsetof(__typeof__(g_fxs), rays_length), 0.85f },
    { "light", offsetof(__typeof__(g_fxs), light), 1.0f },
    { "shadow", offsetof(__typeof__(g_fxs), shadow), 0.3f },
    { "shadow_length", offsetof(__typeof__(g_fxs), shadow_length), 0.6f },
    { "sun", offsetof(__typeof__(g_fxs), sun), 0.5f },
    { "sun_distance", offsetof(__typeof__(g_fxs), sun_distance), 40.0f },
    { "sun_soft", offsetof(__typeof__(g_fxs), sun_soft), 0.03f },
    { "sun_face", offsetof(__typeof__(g_fxs), sun_face), 0.0f },
    { "sun_min", offsetof(__typeof__(g_fxs), sun_min), 1.0f },
    { "sun_direct", offsetof(__typeof__(g_fxs), sun_direct), 0.3f },
    { "sun_casters", offsetof(__typeof__(g_fxs), sun_casters), 1.0f },
    { "sun_near", offsetof(__typeof__(g_fxs), sun_near), 15.0f },
    { "temporal", offsetof(__typeof__(g_fxs), temporal), 0.85f },
    { "debug", offsetof(__typeof__(g_fxs), debug), 0.0f },
    /* ray tracing through the scene's triangles (rt_frame): 0 off, 1 on */
    { "rt", offsetof(__typeof__(g_fxs), rt), 0.0f },
    /* how a fixed-function caster's triangles go back to the world: 1 through its own matrices (clip,
     * object, view, world), 0 through the scene's camera like the vertex shaders' */
    { "rt_proj", offsetof(__typeof__(g_fxs), rt_proj), 1.0f },
    /* a sun of the tester's own for the traced scene (debug = 6): its height over the horizon and
     * its bearing, in degrees; elevation 0 keeps the game's */
    { "rt_sun_elev", offsetof(__typeof__(g_fxs), rt_sun_elev), 0.0f },
    { "rt_sun_azim", offsetof(__typeof__(g_fxs), rt_sun_azim), 0.0f },
    /* the traced sun shadows, in the world's own draws: how much of a surface's colour the shadow
     * takes (0: none - and the sun's maps back), and how far out they reach, in world units */
    { "rt_shadow", offsetof(__typeof__(g_fxs), rt_shadow), 0.7f },
    { "rt_distance", offsetof(__typeof__(g_fxs), rt_distance), 500.0f },
    /* how far the near shadow map reaches (the finer one, round the camera), in world units */
    { "rt_near", offsetof(__typeof__(g_fxs), rt_near), 25.0f },
    { "rt_debug", offsetof(__typeof__(g_fxs), rt_debug), 0.0f },
    { "hide_ui", offsetof(__typeof__(g_fxs), hide_ui), 0.0f },
    { "rt_sm_keep", offsetof(__typeof__(g_fxs), rt_sm_keep), 0.0f },
    /* not effects: the host's draw distances (host64 --draw-distance), live while tuning; 0 leaves them */
    { "draw", offsetof(__typeof__(g_fxs), draw), 0.0f },
    { "draw_entities", offsetof(__typeof__(g_fxs), draw_entities), 0.0f },
    /* not an effect either: the frame-rate overlay, shown (1) or hidden (0) */
    { "fps", offsetof(__typeof__(g_fxs), fps), 1.0f },
};

static float* fx_setting(const char* key)
{
    for (size_t i = 0; i < sizeof FX_SETTINGS / sizeof FX_SETTINGS[0]; ++i)
        if (!strcmp(FX_SETTINGS[i].key, key))
            return (float*)((char*)&g_fxs + FX_SETTINGS[i].at);
    return NULL;
}

void gfx_fx_set(const char* key, float v)
{
    float* p = fx_setting(key);
    if (p)
        *p = v;
}

float gfx_fx_get(const char* key)
{
    float* p = fx_setting(key);
    return p ? *p : 0.0f;
}

static char g_fx_file[1024];
static struct timespec g_fx_mtime;

/* the settings file, when it changed since the last look (from Present, twice a second) */
static void fx_reload(void)
{
    static double last;
    double now = CACurrentMediaTime();
    if (!g_fx_file[0] || now - last < 0.5)
        return;
    last = now;
    struct stat st;
    if (stat(g_fx_file, &st) || (st.st_mtimespec.tv_sec == g_fx_mtime.tv_sec && st.st_mtimespec.tv_nsec == g_fx_mtime.tv_nsec))
        return;
    g_fx_mtime = st.st_mtimespec;
    FILE* f = fopen(g_fx_file, "r");
    if (!f)
        return;
    char line[256], key[64];
    float v, *p;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, " %63[a-z_] = %f", key, &v) == 2 && (p = fx_setting(key)))
            *p = v;
    fclose(f);
    fprintf(stderr, "[recomp] gfx: scene effects %s from %s\n", g_fxs.fx != 0.0f ? "on" : "off", g_fx_file);
}

static void fx_config(void)
{
    for (size_t i = 0; i < sizeof FX_SETTINGS / sizeof FX_SETTINGS[0]; ++i)
    {
        char name[64], *c;
        snprintf(name, sizeof name, i ? "FFXI_FX_%s" : "FFXI_FX", FX_SETTINGS[i].key);
        for (c = name; *c; ++c)
            if (*c >= 'a' && *c <= 'z')
                *c -= 32;
        const char* v = getenv(name);
        *(float*)((char*)&g_fxs + FX_SETTINGS[i].at) = v && *v ? (float)atof(v) : FX_SETTINGS[i].def;
    }
    const char* dbg = getenv("FFXI_FX_DEBUG"); /* also by name */
    if (dbg)
        g_fxs.debug = !strcmp(dbg, "ao") ? 1.0f : !strcmp(dbg, "fog") ? 2.0f : !strcmp(dbg, "bloom") ? 3.0f
            : !strcmp(dbg, "rays") ? 4.0f : !strcmp(dbg, "shadow") ? 5.0f
            : !strcmp(dbg, "rt") ? 6.0f : (float)atof(dbg);
    const char* file = getenv("FFXI_FX_FILE");
    if (file && *file)
        snprintf(g_fx_file, sizeof g_fx_file, "%s", file);
    else if (getenv("HOME"))
        snprintf(g_fx_file, sizeof g_fx_file, "%s/Library/Caches/FFXI/fx.txt", getenv("HOME"));
}

static id<MTLRenderPipelineState> fx_pipeline(NSString* frag, MTLPixelFormat fmt)
{
    MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
    id<MTLFunction> vf = [g_fx.lib newFunctionWithName:@"fx_vs"], ff = [g_fx.lib newFunctionWithName:frag];
    pd.vertexFunction = vf;
    pd.fragmentFunction = ff;
    pd.colorAttachments[0].pixelFormat = fmt;
    NSError* err = nil;
    id<MTLRenderPipelineState> p = [g_dev newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!p)
        __atomic_fetch_add(&g_failures, 1, __ATOMIC_RELAXED),
            fprintf(stderr, "[recomp] gfx: scene effect %s failed: %s\n", [frag UTF8String], err ? [[err localizedDescription] UTF8String] : "?");
    [vf release];
    [ff release];
    [pd release];
    return p;
}

static int fx_init(void)
{
    if (g_fx.tried)
        return g_fx.ao_pipe != nil;
    g_fx.tried = 1;
    g_fx.lib = compile(FX_MSL);
    if (!g_fx.lib)
        return 0;
    g_fx.ao_pipe = fx_pipeline(@"fx_ao", MTLPixelFormatRGBA16Float);
    g_fx.blur_pipe = fx_pipeline(@"fx_blur", MTLPixelFormatRGBA16Float);
    g_fx.bright_pipe = fx_pipeline(@"fx_bright", MTLPixelFormatRGBA16Float);
    g_fx.down_pipe = fx_pipeline(@"fx_down", MTLPixelFormatRGBA16Float);
    g_fx.gauss_pipe = fx_pipeline(@"fx_gauss", MTLPixelFormatRGBA16Float);
    g_fx.raymask_pipe = fx_pipeline(@"fx_raymask", MTLPixelFormatRGBA16Float);
    g_fx.rays_pipe = fx_pipeline(@"fx_rays", MTLPixelFormatRGBA16Float);
    g_fx.temporal_pipe = fx_pipeline(@"fx_temporal", MTLPixelFormatRGBA16Float);
    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    g_fx.samp = [g_dev newSamplerStateWithDescriptor:sd];
    sd.compareFunction = MTLCompareFunctionLessEqual; /* 1 where the point is no deeper than the map */
    g_fx.cmp = [g_dev newSamplerStateWithDescriptor:sd];
    [sd release];
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                  width:1 height:1 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModePrivate;
    g_fx.sdummy = [g_dev newTextureWithDescriptor:td];
    MTLDepthStencilDescriptor* dd = [[MTLDepthStencilDescriptor alloc] init];
    dd.depthCompareFunction = MTLCompareFunctionLess;
    dd.depthWriteEnabled = YES;
    g_fx.sdepth = [g_dev newDepthStencilStateWithDescriptor:dd];
    [dd release];
    if (!g_fx.ao_pipe || !g_fx.blur_pipe || !g_fx.bright_pipe || !g_fx.down_pipe || !g_fx.gauss_pipe || !g_fx.raymask_pipe ||
        !g_fx.rays_pipe || !g_fx.temporal_pipe)
    {
        [g_fx.ao_pipe release], g_fx.ao_pipe = nil;
        return 0;
    }
    fprintf(stderr, "[recomp] gfx: scene effects ready\n");
    return 1;
}

/* a private texture of this size and format in *t, made again when either changes */
static id<MTLTexture> fx_tex(id<MTLTexture>* t, MTLPixelFormat fmt, NSUInteger w, NSUInteger h)
{
    if (*t && (*t).width == w && (*t).height == h && (*t).pixelFormat == fmt)
        return *t;
    [*t release];
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:w height:h mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    *t = [g_dev newTextureWithDescriptor:d];
    return *t;
}

/* one full-screen triangle into target, reading tex[0..n) */
static void fx_pass(id<MTLTexture> target, MTLLoadAction load, id<MTLRenderPipelineState> p, MTLViewport vp, const FxU* u,
    id<MTLTexture> const* tex, int n, const int32_t* dir)
{
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = target;
    rp.colorAttachments[0].loadAction = load;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> e = [cmd() renderCommandEncoderWithDescriptor:rp];
    [e setRenderPipelineState:p];
    [e setViewport:vp];
    [e setFragmentBytes:u length:sizeof *u atIndex:0];
    if (dir)
        [e setFragmentBytes:dir length:8 atIndex:1];
    for (int i = 0; i < n; ++i)
        [e setFragmentTexture:tex[i] atIndex:(NSUInteger)i];
    [e setFragmentSamplerState:g_fx.samp atIndex:0];
    [e setFragmentSamplerState:g_fx.cmp atIndex:1];
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [e endEncoding];
}

static MTLViewport fx_full(id<MTLTexture> t) { return (MTLViewport){ 0, 0, (double)t.width, (double)t.height, 0, 1 }; }

/* the inverse of a 4x4 matrix (row-major); 0 when it has none */
static int mat_inverse(float* out, const float* m)
{
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (det == 0.0f)
        return 0;
    for (int i = 0; i < 16; ++i)
        out[i] = inv[i] / det;
    return 1;
}

static void mat_mul(float* o, const float* a, const float* b)
{
    float t[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            t[i * 4 + j] = a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j] + a[i * 4 + 2] * b[8 + j] + a[i * 4 + 3] * b[12 + j];
    memcpy(o, t, sizeof t);
}

enum { SUN_MAP = 4096, SUN_CACHE_FRAMES = 60 * 30 };

/* The zone's casters, kept after they leave the view: the game draws only what the camera sees, but
 * a low sun throws the shadows of what is behind and beside the camera into it. Each caster drawn
 * from buffers the game keeps is kept with a copy of its uniforms and that frame's camera (clip
 * space back to the world), and drawn into the map from there while it is out of view - for 30
 * seconds, or until the camera jumps (a new zone). */
typedef struct CacheKey
{
    void* vb[GFX_NSTREAMS];
    NSUInteger voff[GFX_NSTREAMS];
    void* ib;
    NSUInteger ioff;
    uint32_t n, vstart;
    LibKey lib;
} CacheKey;

typedef struct Cached
{
    Caster c; /* retains its buffers; c.ub is the cache's own copy of the uniforms */
    float clip_world[16];
    uint64_t seen, replayed;
    int dead; /* a buffer it draws from was destroyed (the game let go of it: a zone left behind) */
} Cached;

static Cached* g_cache;
static uint32_t g_ncache, g_cache_cap;
static Map g_cache_map; /* CacheKey -> index + 1 */
static float g_cache_cam[3];

static void cache_key(CacheKey* k, const Caster* c)
{
    memset(k, 0, sizeof *k);
    for (int s = 0; s < GFX_NSTREAMS; ++s)
        k->vb[s] = (void*)c->vb[s], k->voff[s] = c->voff[s];
    k->ib = (void*)c->ib, k->ioff = c->ioff, k->n = c->n, k->vstart = c->vstart, k->lib = c->lib;
}

static void cache_map_free(void)
{
    for (uint32_t i = 0; i < g_cache_map.cap; ++i)
        if (g_cache_map.e[i].hash)
            free(g_cache_map.e[i].key);
    free(g_cache_map.e);
    memset(&g_cache_map, 0, sizeof g_cache_map);
}

static void rt_forget(id<MTLBuffer> buf);

static void cached_release(Cached* ce)
{
    Caster* c = &ce->c;
    for (int s = 0; s < GFX_NSTREAMS; ++s)
        [c->vb[s] release];
    [c->ub release];
    [c->ib release];
    for (int t = 0; t < 8; ++t)
        [c->tex[t] release];
}

/* a buffer the game destroyed: what the cache draws from it goes */
static void sun_cache_forget(id<MTLBuffer> buf)
{
    if (!buf)
        return;
    rt_forget(buf);
    for (uint32_t i = 0; i < g_ncache; ++i)
    {
        Caster* c = &g_cache[i].c;
        int hit = c->ib == buf;
        for (int s = 0; s < GFX_NSTREAMS; ++s)
            hit |= c->vb[s] == buf;
        if (hit)
            g_cache[i].dead = 1;
    }
}

/* drops what has not been seen for SUN_CACHE_FRAMES (all of it when all is true) and rebuilds the map */
static void sun_cache_trim(int all)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < g_ncache; ++i)
    {
        if (all || g_cache[i].dead || g_cache[i].seen + SUN_CACHE_FRAMES < g_serial)
            cached_release(&g_cache[i]);
        else
            g_cache[n++] = g_cache[i];
    }
    g_ncache = n;
    cache_map_free();
    for (uint32_t i = 0; i < n; ++i)
    {
        CacheKey k;
        cache_key(&k, &g_cache[i].c);
        map_put(&g_cache_map, &k, sizeof k, (id)(uintptr_t)(i + 1));
    }
}

/* this frame's fixed casters into the cache (new ones added, seen ones brought up to date) */
static void sun_cache_update(const float* clip_world, const float* cam)
{
    float dx = cam[0] - g_cache_cam[0], dy = cam[1] - g_cache_cam[1], dz = cam[2] - g_cache_cam[2];
    if (dx * dx + dy * dy + dz * dz > 50.0f * 50.0f)
    {
        if (gfx_profiling)
            fprintf(stderr, "[recomp] gfx: shadows: the camera jumped %.0f units (frame %llu): cache cleared\n",
                sqrtf(dx * dx + dy * dy + dz * dz), (unsigned long long)g_serial);
        sun_cache_trim(1);
        g_fx.fogc_set = 0;
    }
    memcpy(g_cache_cam, cam, 12);
    if (!(g_serial & 255))
        sun_cache_trim(0);
    for (uint32_t i = 0; i < g_ncasters; ++i)
    {
        const Caster* c = &g_casters[i];
        if (!c->fixed)
            continue;
        CacheKey k;
        cache_key(&k, c);
        uintptr_t at = (uintptr_t)map_get(&g_cache_map, &k, sizeof k);
        Cached* ce;
        if (!at)
        {
            if (g_ncache == g_cache_cap)
            {
                g_cache_cap = g_cache_cap ? g_cache_cap * 2 : 1024;
                g_cache = (Cached*)realloc(g_cache, g_cache_cap * sizeof(Cached));
            }
            ce = &g_cache[g_ncache++];
            memset(ce, 0, sizeof *ce);
            ce->c = *c;
            for (int s = 0; s < GFX_NSTREAMS; ++s)
                [ce->c.vb[s] retain];
            [ce->c.ib retain];
            for (int t = 0; t < 8; ++t)
                [ce->c.tex[t] retain];
            ce->c.ub = nil;
            map_put(&g_cache_map, &k, sizeof k, (id)(uintptr_t)g_ncache);
        }
        else
            ce = &g_cache[at - 1];
        /* the uniforms as of this frame: in place, unless a frame the GPU may still be drawing
         * read them */
        const void* src = (const uint8_t*)[c->ub contents] + c->uoff;
        if (ce->c.ub && ce->replayed + FRAMES < g_serial)
            memcpy([ce->c.ub contents], src, sizeof(GfxU));
        else
        {
            [ce->c.ub release];
            ce->c.ub = [g_dev newBufferWithBytes:src length:sizeof(GfxU) options:MTLResourceStorageModeShared];
        }
        ce->c.uoff = 0;
        memcpy(ce->clip_world, clip_world, 64);
        ce->seen = g_serial;
    }
}

/* The sun's shadow map for the scene: an orthographic view along the sun over a sphere around the
 * first sun_distance units the camera sees, its center snapped to whole texels (the shadows' edges
 * hold still as the camera moves), the casters drawn into it. Gives the matrix from the camera's view
 * space to the map (lmat) and a texel's size in world units; 0 when there is no map. */
/* One cascade of the sun's map: an orthographic view along the sun over a sphere around the slice
 * [t0, t1] of what the camera sees, its center snapped to whole texels (the shadows' edges hold still
 * as the camera moves). */
typedef struct SunCascade
{
    float S[16];    /* the world to the map */
    float lmat[16]; /* view space to the map: x, y -1..1, z 0..1 */
    float texel, bias, soft, slope, range, across;
} SunCascade;

static void sun_fit(const GfxScene* s, const float* invV, const float* L, float t0, float t1, SunCascade* k)
{
    float hand = s->proj[11] < 0.0f ? -1.0f : 1.0f, p[8][3], c[3] = { 0, 0, 0 };
    for (int i = 0; i < 8; ++i)
    {
        float t = i < 4 ? t0 : t1, z = t * hand, nx = (i & 1) ? 1.0f : -1.0f, ny = (i & 2) ? 1.0f : -1.0f;
        float v[4] = { (nx * t - s->proj[8] * z) / s->proj[0], (ny * t - s->proj[9] * z) / s->proj[5], z, 1.0f };
        for (int j = 0; j < 3; ++j)
            p[i][j] = v[0] * invV[j] + v[1] * invV[4 + j] + v[2] * invV[8 + j] + invV[12 + j], c[j] += p[i][j] / 8.0f;
    }
    float R = 0.0f;
    for (int i = 0; i < 8; ++i)
    {
        float dx = p[i][0] - c[0], dy = p[i][1] - c[1], dz = p[i][2] - c[2];
        R = fmaxf(R, sqrtf(dx * dx + dy * dy + dz * dz));
    }
    R = ceilf(R); /* whole units: the texel's size holds still */
    /* the sun's axes: f the way its light goes, r and u across */
    float f[3] = { -L[0], -L[1], -L[2] }, up[3] = { 0, 1, 0 };
    if (fabsf(f[1]) > 0.9f)
        up[0] = 1, up[1] = 0;
    float r[3] = { up[1] * f[2] - up[2] * f[1], up[2] * f[0] - up[0] * f[2], up[0] * f[1] - up[1] * f[0] };
    normalize3(r);
    float u[3] = { f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0] };
    float tx = 2.0f * R / SUN_MAP;
    float cx = floorf((c[0] * r[0] + c[1] * r[1] + c[2] * r[2]) / tx) * tx;
    float cy = floorf((c[0] * u[0] + c[1] * u[1] + c[2] * u[2]) / tx) * tx;
    float cz = c[0] * f[0] + c[1] * f[1] + c[2] * f[2];
    /* casters stand up to 200 units sunward of the sphere (a cliff over the camera) */
    float back = 200.0f, range = 2.0f * R + back, z0 = cz - R - back;
    float S[16] = {
        r[0] / R, u[0] / R, f[0] / range, 0,
        r[1] / R, u[1] / R, f[1] / range, 0,
        r[2] / R, u[2] / R, f[2] / range, 0,
        -cx / R, -cy / R, -z0 / range, 1,
    };
    memcpy(k->S, S, sizeof S);
    mat_mul(k->lmat, invV, S);
    k->texel = tx, k->bias = 0.03f / range, k->range = range, k->across = 2.0f * R;
    /* the penumbra's radius grows by sun_soft for each unit from the caster: in the map's width
     * (2R across) per unit of its depth (range deep) */
    k->soft = g_fxs.sun_soft * range / (2.0f * R);
    k->slope = 2.0f * R / range;
}

static id<MTLTexture> sun_target(id<MTLTexture>* t)
{
    if (!*t)
    {
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                      width:SUN_MAP height:SUN_MAP mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModePrivate;
        *t = [g_dev newTextureWithDescriptor:td];
    }
    if (!g_fx.scol)
    {
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                      width:SUN_MAP height:SUN_MAP mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModeMemoryless;
        g_fx.scol = [g_dev newTextureWithDescriptor:td];
    }
    return *t && g_fx.scol ? *t : nil;
}

/* the casters into one cascade's map: this frame's, and the zone's kept from before (cache) */
static uint32_t sun_draw(id<MTLTexture> target, const float* invP, const float* invV, const SunCascade* k, int cache)
{
    float clip_world[16], M[16];
    mat_mul(clip_world, invP, invV);
    mat_mul(M, clip_world, k->S); /* the camera's clip space -> the map */
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.depthAttachment.texture = target;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    rp.depthAttachment.clearDepth = 1.0;
    rp.depthAttachment.storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].texture = g_fx.scol;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionDontCare;
    id<MTLRenderCommandEncoder> e = [cmd() renderCommandEncoderWithDescriptor:rp];
    [e setViewport:(MTLViewport){ 0, 0, SUN_MAP, SUN_MAP, 0, 1 }];
    [e setDepthStencilState:g_fx.sdepth];
    [e setCullMode:MTLCullModeNone];
    [e setDepthBias:0 slopeScale:1.5f clamp:0];
    uint32_t drawn = 0, total = g_ncasters + (cache ? g_ncache : 0);
    for (uint32_t i = 0; i < total; ++i)
    {
        const Caster* cs;
        if (i < g_ncasters)
        {
            cs = &g_casters[i];
            if (i == 0)
                [e setVertexBytes:M length:64 atIndex:5];
            /* sun_casters 1: characters alone cast - the zone's shadows are baked into its colours
             * already, and the game tints them for the hour and the weather */
            if (g_fxs.sun_casters == 1.0f && cs->fixed)
                continue;
        }
        else
        {
            if (g_fxs.sun_casters == 1.0f)
                break;
            /* the zone out of view: as it was drawn when last seen, through that frame's camera */
            Cached* ce = &g_cache[i - g_ncasters];
            if (ce->dead || ce->seen == g_serial || ce->seen + SUN_CACHE_FRAMES < g_serial)
                continue;
            float m[16];
            mat_mul(m, ce->clip_world, k->S);
            [e setVertexBytes:m length:64 atIndex:5];
            ce->replayed = g_serial;
            cs = &ce->c;
        }
        PipeKey pk;
        memset(&pk, 0, sizeof pk);
        pk.lib = cs->lib;
        pk.lib.vs.shadow = 1, pk.lib.vs.pixel = 0;
        int at = alpha_tested(&cs->lib.fs);
        if (!at)
        {
            /* the position alone: one pipeline serves every draw with the same vertex layout */
            GfxVsKey* v = &pk.lib.vs;
            v->lighting = v->normalize = v->localviewer = v->specular = 0;
            v->src_diffuse = v->src_specular = v->src_ambient = v->src_emissive = 0;
            v->nlights = 0, memset(v->light_type, 0, sizeof v->light_type);
            v->fog_vertex = v->range_fog = 0, v->ntex = 0, v->flat = 0;
            memset(v->tci, 0, sizeof v->tci), memset(v->ttf, 0, sizeof v->ttf);
            memset(&pk.lib.fs, 0, sizeof pk.lib.fs);
        }
        pk.color = (uint32_t)MTLPixelFormatR8Unorm, pk.depth = (uint32_t)MTLPixelFormatDepth32Float;
        id<MTLRenderPipelineState> ps = pipeline_for(&pk, cs->vs, cs->ps);
        if (!ps)
        {
            g_fx.tr_skipped++;
            continue;
        }
        if (i < g_ncasters)
            g_fx.tr_live++;
        else
            g_fx.tr_cached++;
        [e setRenderPipelineState:ps];
        for (int st = 0; st < GFX_NSTREAMS; ++st)
            [e setVertexBuffer:cs->vb[st] offset:cs->voff[st] atIndex:(NSUInteger)st];
        [e setVertexBuffer:cs->ub offset:cs->uoff atIndex:4];
        if (at)
        {
            [e setFragmentBuffer:cs->ub offset:cs->uoff atIndex:4];
            for (int t = 0; t < 8; ++t)
                if (cs->tex[t])
                {
                    GfxSampler sk = cs->samp[t];
                    [e setFragmentTexture:cs->tex[t] atIndex:(NSUInteger)t];
                    [e setFragmentSamplerState:sampler(&sk) atIndex:(NSUInteger)t];
                }
        }
        if (cs->itype)
            [e drawIndexedPrimitives:cs->prim indexCount:cs->n
                           indexType:cs->itype == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                         indexBuffer:cs->ib indexBufferOffset:cs->ioff];
        else
            [e drawPrimitives:cs->prim vertexStart:cs->vstart vertexCount:cs->n];
        drawn++;
    }
    [e endEncoding];
    return drawn;
}

/* The sun's maps for the scene: a near cascade over the first sun_near units the camera sees, where
 * characters stand and shadows are looked at closely (a 4096 map over some 40 units: texels of a
 * centimetre), and a far one out to sun_distance. Fills the effects' uniforms for both; 0 when
 * nothing was drawn. */
static int sun_map(const GfxScene* s, const float* L, FxU* u)
{
    if (!g_ncasters && !g_ncache)
        return 0;
    float invP[16], invV[16];
    if (!mat_inverse(invP, s->proj) || !mat_inverse(invV, s->view))
        return 0;
    float clip_world[16];
    mat_mul(clip_world, invP, invV);
    sun_cache_update(clip_world, invV + 12);
    float dfar = fmaxf(g_fxs.sun_distance, 4.0f), dnear = fminf(fmaxf(g_fxs.sun_near, 0.0f), dfar);
    SunCascade far, near;
    sun_fit(s, invV, L, 0.5f, dfar, &far);
    if (!sun_target(&g_fx.smap))
        return 0;
    uint32_t drawn = sun_draw(g_fx.smap, invP, invV, &far, 1);
    memcpy(u->lmat, far.lmat, 64);
    u->smap[1] = far.texel, u->smap[2] = far.bias, u->smap[3] = far.soft;
    u->smap2[0] = far.slope, u->smap2[3] = far.range;
    u->smapn2[1] = 0.0f;
    if (dnear >= 2.0f && dnear < dfar && sun_target(&g_fx.smapn))
    {
        sun_fit(s, invV, L, 0.5f, dnear, &near);
        sun_draw(g_fx.smapn, invP, invV, &near, 1);
        memcpy(u->lmatn, near.lmat, 64);
        u->smapn[0] = near.texel, u->smapn[1] = near.bias, u->smapn[2] = near.soft, u->smapn[3] = near.slope;
        u->smapn2[0] = near.range, u->smapn2[1] = 1.0f;
    }
    g_fx.st_across = far.across;
    g_fx.st_cached = g_ncache;
    return drawn != 0;
}

/* --- ray tracing (rt = 1) -------------------------------------------------------------------------------------
 * The scene's casters as triangles, in acceleration structures the scene effects trace rays through
 * (Metal ray tracing). A caster is run once more through its own vertex function with nothing drawn
 * (the capture key, gfx_msl_vs_open): the position of each of its triangles' vertices goes out to a
 * buffer, three a triangle, through a matrix from its clip space.
 *
 * A mesh the game keeps (a fixed caster whose transform is one matrix: the fixed function's wvp, or the
 * four constants a vertex shader places its vertices with, gfx_msl_vs1_pos_consts) is captured once, in
 * its own space, into a structure of its own. Each place it is drawn - that matrix through the camera's
 * inverse, object to world - is an instance of it, kept for RT_KEEP frames after it was last drawn: the
 * game draws only what the camera sees, and the zone behind the camera still casts. A zone's meshes are
 * drawn many times each (rocks, trees, tiles of ground), and a door is one place that moves. A place the
 * camera looks straight at that is no longer drawn is dropped (the game drew a nearer or farther detail
 * of the mesh instead); a mesh whose places move on most frames (grass swaying) is placed anew each frame
 * and kept nowhere. The rest (characters, drawn from the frame's ring) is captured in the world every
 * frame, into one structure.
 * The instance structure over all of it is built every frame.
 *
 * debug = 6 shows the traced scene against the drawn one: clay, lit by the scene's sun with a shadow
 * ray toward it from each hit, where the hit agrees with the depth drawn; red where the ray hits nearer (a mesh kept where
 * nothing is now), blue where it hits farther or nothing (a surface drawn that the structures lack),
 * yellow where it hits something in the sky. */
enum { RT_BUILDS = 256, RT_BUILD_TRIS = 400000 };

static const char RT_MSL[] =
    "#include <metal_raytracing>\n"
    "using namespace raytracing;\n"

    "kernel void rt_view(uint2 id [[thread_position_in_grid]], constant FxU& u [[buffer(0)]],\n"
    "                    instance_acceleration_structure as [[buffer(1)]], device const ulong* vtx [[buffer(2)]],\n"
    "                    depth2d<float> dt [[texture(0)]], texture2d<float, access::write> out [[texture(1)]]) {\n"
    "  uint2 sz = uint2(out.get_width(), out.get_height());\n"
    "  if (id.x >= sz.x || id.y >= sz.y) return;\n"
    "  float2 px = floor(u.vp.xy + (float2(id) + 0.5) * u.vp.zw / float2(sz)) + 0.5;\n"
    /* clip space back to the world, less the camera's place: the pixel at the near plane, halfway, and
     * at the depth drawn there */
    "  float2 ndc = float2((px.x - u.vp.x) / u.vp.z * 2.0 - 1.0, 1.0 - (px.y - u.vp.y) / u.vp.w * 2.0);\n"
    "  float4 a = u.ivp * float4(ndc, 0.0, 1.0), b = u.ivp * float4(ndc, 0.5, 1.0);\n"
    "  float d = (dt.read(uint2(px)) - u.zp.z) / max(u.zp.w - u.zp.z, 1e-6);\n"
    "  float4 q = u.ivp * float4(ndc, d, 1.0);\n"
    "  float dd = d < 0.999999 ? length(q.xyz / q.w) : 0.0;\n"
    "  ray r;\n"
    "  r.origin = u.cam.xyz;\n"
    "  r.direction = normalize(b.xyz / b.w - a.xyz / a.w);\n"
    "  r.min_distance = 0.0;\n"
    "  r.max_distance = 1e5;\n"
    "  intersector<triangle_data, instancing, world_space_data> x;\n"
    "  x.assume_geometry_type(geometry_type::triangle);\n"
    "  x.force_opacity(forced_opacity::opaque);\n"
    "  intersector<triangle_data, instancing, world_space_data>::result_type h = x.intersect(r, as);\n"
    /* clay where the hit agrees with the depth drawn: the face hit (its triangle, each instance's
     * vertices at vtx[instance], three a triangle), lit from above, darker with distance; red nearer,
     * blue farther or missing, yellow in the sky - only past 5% and 0.3 units (a pixel's edge is
     * neither) */
    "  float3 c = float3(0.0);\n"
    "  if (h.type == intersection_type::none) c = dd > 0.0 ? float3(0.1, 0.3, 1.0) : float3(0.0);\n"
    "  else if (dd <= 0.0) c = float3(1.0, 0.9, 0.1);\n"
    "  else {\n"
    "    float e = h.distance - dd;\n"
    "    if (abs(e) > max(0.05 * dd, 0.3)) c = e < 0.0 ? float3(1.0, 0.15, 0.1) : float3(0.1, 0.3, 1.0);\n"
    "    else {\n"
    /* each vertex a position, a normal and texture coordinates; the normals, where the vertices came
     * with them (the low bit of the address), blended across the triangle: smooth; else the face's */
    "      ulong at = vtx[h.instance_id];\n"
    "      device const float4* v = (device const float4*)(at & ~1ul);\n"
    "      uint p = h.primitive_id * 9u;\n"
    "      float3 f = cross(v[p + 3].xyz - v[p].xyz, v[p + 6].xyz - v[p].xyz), n = f;\n"
    "      if (at & 1ul) {\n"
    "        float2 bc = h.triangle_barycentric_coord;\n"
    "        float3 s = v[p + 1].xyz * (1.0 - bc.x - bc.y) + v[p + 4].xyz * bc.x + v[p + 7].xyz * bc.y;\n"
    "        if (dot(s, s) > 1e-8) n = dot(s, f) < 0.0 ? -s : s;\n"
    "      }\n"
    "      n = normalize(h.object_to_world_transform * float4(n, 0.0));\n"
    "      if (dot(n, r.direction) > 0.0) n = -n;\n"
    /* lit by the scene's sun (or moon), and shaded where a ray toward it hits anything: a shadow
     * from anything kept, in view or not; with no sun, from above (FFXI's world y points down) */
    "      float3 L = u.sunw.w > 0.0 ? normalize(u.sunw.xyz) : normalize(float3(0.35, -0.8, 0.45));\n"
    "      float nl = saturate(dot(n, L)), vis = 1.0;\n"
    "      if (u.sunw.w > 0.0 && nl > 0.0) {\n"
    "        ray sr;\n"
    "        sr.origin = r.origin + r.direction * h.distance + n * max(0.002 * h.distance, 0.01);\n"
    "        sr.direction = L;\n"
    "        sr.min_distance = 0.0;\n"
    "        sr.max_distance = 1e4;\n"
    "        intersector<triangle_data, instancing> sx;\n"
    "        sx.assume_geometry_type(geometry_type::triangle);\n"
    "        sx.force_opacity(forced_opacity::opaque);\n"
    "        sx.accept_any_intersection(true);\n"
    "        if (sx.intersect(sr, as, 3u).type != intersection_type::none) vis = 0.0;\n"
    "      }\n"
    "      float lit = 0.25 + 0.75 * nl * vis;\n"
    "      c = float3(0.85, 0.82, 0.78) * lit * mix(1.0, 0.5, saturate(h.distance / 200.0));\n"
    "    }\n"
    "  }\n"
    "  out.write(float4(c, 1.0), id);\n"
    "}\n";

/* a mesh the game keeps (drawn from its own buffers): its triangles once, in its own space */
typedef struct RtMesh
{
    Caster c; /* its buffers (retained); c.ub nil: the uniforms are a draw's, at its capture */
    id<MTLBuffer> rtv;
    id<MTLAccelerationStructure> blas;
    uint32_t tris;
    uint32_t nplace; /* the places it is drawn at */
    uint64_t seen, tried;
    int dead; /* a buffer it draws from changed or went: captured again when drawn again */
    int none; /* no triangles to trace (lines, points) */
    uint32_t at; /* its index among this frame's instance structures */
    /* its draws since the last look (rt_trim): on a place kept as it was, or not (moved, new); unstable:
     * most were not - it moves every frame (swaying, animated), placed anew each frame and kept nowhere */
    uint32_t n_same, n_move;
    int unstable;
    /* its box in its own space, read back once its capture is done (rt_bounds) */
    float lo[3], hi[3];
    uint64_t captured; /* the frame it was captured in */
    int bounds;
    int normals; /* its vertices came with normals (smooth shading); else the faces' */
    /* alpha-tested: its first texture, how it is sampled and the test's reference (the shadow maps let
     * through what the test lets through) */
    int alpha;
    id<MTLTexture> tex;
    GfxSampler samp;
    float aref;
    uint32_t sm_count, sm_first; /* its places in this frame's shadow maps' instance list */
    /* this frame's draws of it, and the last one's object to world (rt_anchor) */
    uint64_t drawn;
    uint32_t ndrawn, order; /* and the first of them among the frame's draws */
    float clip[16][16], clip_prev[16][16]; /* object to clip of its first draws in a frame traced, and the one before's */
    uint64_t clip_serial, clip_prev_serial;
    uint32_t nclip, nclip_prev;
    uint64_t dt_serial; /* its draws so far this frame (rt_frame_camera) */
    uint32_t dt_n;
    double wd[16];
} RtMesh;

/* one place a mesh is drawn: its object space to the world (row vectors, as D3D) */
typedef struct RtPlace
{
    uint32_t mesh;
    float w[16];
    uint64_t seen;
    int32_t next; /* the next in its hash cell (rt_hash), -1 */
    uint8_t missed; /* frames traced since, in the camera's view but not drawn */
    uint8_t gone;   /* not drawn where the camera sees it: replaced (a nearer or farther detail of it), dropped */
} RtPlace;

static struct
{
    int tried, ok;
    id<MTLLibrary> lib;
    id<MTLComputePipelineState> view_pipe;
    id<MTLAccelerationStructure> tlas, dyn; /* this frame's */
    NSArray* blas;                          /* what the instances point at (the tracing passes' useResources) */
    id<MTLTexture> out;                     /* debug = 6 */
    id<MTLBuffer> vtx;                      /* each instance's triangles' vertices, by GPU address */
    NSUInteger vtx_off;
    NSMutableArray* res;                    /* what the rays read besides: the vertices (useResources) */
    RtMesh* mesh;
    uint32_t nmesh, cap_mesh;
    Map map; /* CacheKey -> mesh index + 1 */
    RtPlace* place;
    uint32_t nplace, cap_place;
    int32_t* head; /* places by mesh and whole world unit (rt_hash): the first in each cell, -1 */
    uint32_t nhead;
    uint64_t prev; /* the frame traced before this one (the world is not drawn every frame) */
    float cam[3];
    /* the vertex shaders' position transforms (gfx_msl_vs1_pos_consts), by shader */
    uint32_t pos_prog[64];
    int pos_k[64];
    uint32_t npos;
    /* the profile (FFXI_PROFILE) */
    uint32_t nkept, kept_tris, ninst, moving_tris, built, waiting;
    uint32_t pl_same, pl_moved, pl_new, pl_samefr; /* places matched as kept, moved, new; new beside one drawn this frame */
    float pl_nd, pl_nt; /* a new place's nearest kept one: largest rotation difference, translation distance */
    /* this frame's places of the unstable meshes: mesh, object to world */
    struct { uint32_t mesh; float w[16]; } *now;
    uint32_t nnow, cap_now;
    uint32_t pl_gone; /* places dropped: in view, not drawn */
    /* the anchors: meshes drawn once a frame, where they stand (object to world), from which each frame's
     * camera is found (rt_camera) */
    struct { uint32_t mesh; double w[16]; uint64_t seen; CacheKey key; } anchor[8];
    uint32_t nanchor, anchored, anchor_fallback;
    /* this frame's camera for the rays: its place in the world, and clip space to the world less that
     * place (small numbers: precise in float) */
    float ray_cam[3], ray_ivp[16];
    float late_cw[16], late_vw[16]; /* the camera of the frame traced before (its draws after the effects) */
    uint32_t late_used;             /* the profile: their triangles traced */
    uint32_t zones;                 /* the profile: new zones (everything kept let go) */
    /* the world's draws traced (rt_draw_bind): the camera of the frame traced last (world to clip), and
     * this frame's as the draws find it - the last one's until an anchor is drawn, then that one's */
    int render_ok;                  /* rays from the draws' own functions (supportsRaytracingFromRender) */
    double vp_last[16], vp_prev[16]; /* and the one of the frame before it */
    int have_vp_last, have_vp_prev;
    uint64_t vp_last_serial;
    double fvp[16];
    float fvp_m[16], fvp_cam[4];
    uint64_t fvp_serial;
    int fvp_ok, fvp_exact;
    uint32_t cam_draws;             /* draws given a camera this frame (the trace) */
    double fvp_vp[16];              /* the last exact camera the draws had (world to clip), and its frame */
    uint64_t fvp_vp_serial;
    uint32_t cam_from_draws;        /* the profile: frames whose scene took the draws' camera (no anchor drawn) */
    uint32_t cam_chain;             /* and the last one's, moved on by the meshes drawn in both */
    uint32_t cam_guessed;           /* and the last one's moved on as it moved (nothing else to go by) */
    uint32_t cam_fresh;             /* the scene's own, found this frame */
    uint32_t ch_calls, ch_cand, ch_most, ch_seen, ch_samecount, ch_prev; /* the profile: rt_chain's inputs */
    const GfxTex* fvp_target;       /* the target the camera above is for: each has its own (a reflection's) */
    struct { const GfxTex* t; uint64_t serial; int ok, exact, have_pend; float m[16], cam[4], pm[16], pcam[4]; } fvp_slot[4];
    uint64_t solo_serial;
    int32_t* solo;                  /* this frame: each mesh's one live place, else -1 (rt_frame_camera) */
    uint32_t cap_solo;
    int have_pend;                  /* a camera from a solo place too far from the guess, until another agrees */
    float pend_m[16], pend_cam[4];
    double pend_vp[16];
    uint32_t dr_solo, dr_solo_far;  /* the profile: frames whose camera came from a solo place; ones held */
    float sun_now[4], k_now[4];     /* toward the sun and the shadow's settings, as of the last scene */
    float day;                      /* the sun's share of the light at the last scene (0: none, night) */
    uint64_t enc_pass;              /* the render pass given the structures (useResources) */
    uint32_t dr_exact, dr_pred, dr_ffp, dr_none; /* the profile: traced draws, by the camera they had */
    double cam_spread, cam_scene; /* the profile: the anchors' spread, the most the scene's camera is off */
    uint32_t cam_agree;
    uint32_t pl_far, pl_full; /* draws not placed: out of the world (far off, not finite), their mesh at RT_PLACES */
} g_rtx;

/* released once the GPU is past everything recorded so far (a structure an instance structure in flight
 * may still point at) */
static void rt_release_later(id o)
{
    if (!o)
        return;
    [cmd() addCompletedHandler:^(id<MTLCommandBuffer> done) {
        (void)done;
        [o release];
    }];
}

static int rt_init(void)
{
    if (g_rtx.tried)
        return g_rtx.ok;
    g_rtx.tried = 1;
    if (![g_dev supportsRaytracing])
    {
        fprintf(stderr, "[recomp] gfx: ray tracing: not on this GPU\n");
        return 0;
    }
    size_t a = strlen(FX_MSL), b = strlen(RT_MSL);
    char* src = (char*)malloc(a + b + 1);
    memcpy(src, FX_MSL, a), memcpy(src + a, RT_MSL, b + 1);
    g_rtx.lib = compile(src);
    free(src);
    if (!g_rtx.lib)
        return 0;
    id<MTLFunction> f = [g_rtx.lib newFunctionWithName:@"rt_view"];
    NSError* err = nil;
    g_rtx.view_pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    [f release];
    if (!g_rtx.view_pipe)
    {
        fprintf(stderr, "[recomp] gfx: ray tracing failed: %s\n", err ? [[err localizedDescription] UTF8String] : "?");
        return 0;
    }
    g_rtx.ok = 1;
    g_rtx.render_ok = [g_dev supportsRaytracingFromRender];
    fprintf(stderr, "[recomp] gfx: ray tracing ready (%s%s)\n", [[g_dev name] UTF8String],
        g_rtx.render_ok ? "" : "; not from the draws: no traced shadows");
    return 1;
}

/* the vertex ids of a caster's triangles, three each (a strip's joins left out), into the frame's ring;
 * how many (0: no triangles) */
static uint32_t rt_ids(const Caster* c, id<MTLBuffer>* buf, NSUInteger* off)
{
    uint32_t ntri = c->prim == MTLPrimitiveTypeTriangle ? c->n / 3
        : c->prim == MTLPrimitiveTypeTriangleStrip && c->n >= 3 ? c->n - 2 : 0;
    if (!ntri)
        return 0;
    const uint8_t* ib = c->itype ? (const uint8_t*)[c->ib contents] + c->ioff : NULL;
    uint32_t* ids = (uint32_t*)ring((size_t)ntri * 12, 16, buf, off);
    uint32_t m = 0;
    for (uint32_t t = 0; t < ntri; ++t)
    {
        uint32_t k[3];
        for (int j = 0; j < 3; ++j)
        {
            uint32_t at = c->prim == MTLPrimitiveTypeTriangle ? 3 * t + j : t + j;
            k[j] = !ib ? c->vstart + at : c->itype == 2 ? ((const uint16_t*)ib)[at] : ((const uint32_t*)ib)[at];
        }
        if (k[0] == k[1] || k[1] == k[2] || k[0] == k[2])
            continue;
        ids[m++] = k[0], ids[m++] = k[1], ids[m++] = k[2];
    }
    return m;
}

/* the caster's capture function: the position alone, as in the sun's map, so draws with one vertex
 * layout share it; nil while it builds */
static id<MTLRenderPipelineState> rt_pipeline(const Caster* cs)
{
    PipeKey pk;
    memset(&pk, 0, sizeof pk);
    pk.lib.vs = cs->lib.vs;
    GfxVsKey* v = &pk.lib.vs;
    v->capture = 1, v->shadow = 0, v->pixel = 0;
    v->lighting = v->normalize = v->localviewer = v->specular = 0;
    v->src_diffuse = v->src_specular = v->src_ambient = v->src_emissive = 0;
    v->nlights = 0, memset(v->light_type, 0, sizeof v->light_type);
    v->fog_vertex = v->range_fog = 0, v->flat = 0;
    if (!alpha_tested(&cs->lib.fs)) /* the texture coordinates only where the texture decides what is there */
        v->ntex = 0, memset(v->tci, 0, sizeof v->tci), memset(v->ttf, 0, sizeof v->ttf);
    return pipeline_for(&pk, cs->vs, NULL);
}

static void rt_capture(id<MTLRenderCommandEncoder> e, id<MTLRenderPipelineState> p, const Caster* cs, const float* clip_world,
    id<MTLBuffer> ids, NSUInteger ids_off, uint32_t m, id<MTLBuffer> out, NSUInteger out_off)
{
    [e setRenderPipelineState:p];
    for (int st = 0; st < GFX_NSTREAMS; ++st)
        [e setVertexBuffer:cs->vb[st] offset:cs->voff[st] atIndex:(NSUInteger)st];
    [e setVertexBuffer:cs->ub offset:cs->uoff atIndex:4];
    [e setVertexBytes:clip_world length:64 atIndex:5];
    [e setVertexBuffer:out offset:out_off atIndex:6];
    [e setVertexBuffer:ids offset:ids_off atIndex:7];
    [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:m];
}

/* what takes a caster's clip-space positions to the world: for the fixed function (rt_proj 1) its own
 * matrices, clip space back through wvp to the object, then out through wv and the camera to the world -
 * whatever projection it was drawn with; else the scene camera's inverse (clip_world) */
static void rt_matrix(const Caster* c, const float* clip_world, const float* view_world, float* out)
{
    memcpy(out, clip_world, 64);
    if (!c->lib.vs.prog && c->has_cw) /* its own view and projection's */
    {
        memcpy(out, c->cw, 64);
        return;
    }
    if (c->lib.vs.prog || g_fxs.rt_proj == 0.0f || !c->ub)
        return;
    const GfxU* u = (const GfxU*)((const uint8_t*)[c->ub contents] + c->uoff);
    float iwvp[16], t[16];
    if (mat_inverse(iwvp, u->wvp))
        mat_mul(t, iwvp, u->wv), mat_mul(out, t, view_world);
}

/* the profile: the projections the fixed-function casters were drawn with, against the scene's */
static void rt_projections(const float* proj)
{
    static double last;
    double now = CACurrentMediaTime();
    if (now - last < 2.0)
        return;
    last = now;
    float seen[8][4];
    uint32_t count[8] = { 0 }, nseen = 0, prog = 0, fixed = 0, moving = 0;
    for (uint32_t i = 0; i < g_ncasters; ++i)
    {
        const Caster* c = &g_casters[i];
        c->fixed ? fixed++ : moving++;
        if (c->lib.vs.prog)
        {
            prog++;
            continue;
        }
        const GfxU* u = (const GfxU*)((const uint8_t*)[c->ub contents] + c->uoff);
        float iwv[16], P[16];
        if (!mat_inverse(iwv, u->wv))
            continue;
        mat_mul(P, iwv, u->wvp);
        float k[4] = { P[0], P[5], P[10], P[14] };
        uint32_t j = 0;
        for (; j < nseen; ++j)
            if (fabsf(seen[j][0] - k[0]) < 1e-3f && fabsf(seen[j][1] - k[1]) < 1e-3f && fabsf(seen[j][2] - k[2]) < 1e-3f &&
                fabsf(seen[j][3] - k[3]) < 1e-3f)
                break;
        if (j == nseen && nseen < 8)
            memcpy(seen[nseen++], k, 16);
        if (j < 8)
            count[j]++;
    }
    fprintf(stderr, "[recomp] gfx: rt: scene projection P00 %.4f P11 %.4f P22 %.4f P32 %.4f (P20 %.4f P21 %.4f P23 %.1f); casters %u "
        "(%u fixed, %u moving, %u vertex shaders)\n", proj[0], proj[5], proj[10], proj[14], proj[8], proj[9], proj[11],
        g_ncasters, fixed, moving, prog);
    for (uint32_t j = 0; j < nseen; ++j)
        fprintf(stderr, "[recomp] gfx: rt:   drawn with P00 %.4f P11 %.4f P22 %.4f P32 %.4f: %u casters\n", seen[j][0], seen[j][1],
            seen[j][2], seen[j][3], count[j]);
}

static MTLPrimitiveAccelerationStructureDescriptor* rt_blas_desc(id<MTLBuffer> v, NSUInteger off, uint32_t ntri)
{
    MTLAccelerationStructureTriangleGeometryDescriptor* g = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    g.vertexBuffer = v, g.vertexBufferOffset = off, g.vertexStride = 48, g.triangleCount = ntri, g.opaque = YES; /* position, normal, uv */
    MTLPrimitiveAccelerationStructureDescriptor* d = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    d.geometryDescriptors = @[ g ];
    return d;
}

enum
{
    RT_KEEP = 60 * 30,  /* frames a place is kept after it was last drawn */
    RT_PLACES = 4096,   /* places one mesh may have */
    RT_WORLD = 100000,  /* how far from the origin a place may be, in world units */
};

static void rt_map_free(void)
{
    for (uint32_t i = 0; i < g_rtx.map.cap; ++i)
        if (g_rtx.map.e[i].hash)
            free(g_rtx.map.e[i].key);
    free(g_rtx.map.e);
    memset(&g_rtx.map, 0, sizeof g_rtx.map);
}

static void rt_mesh_release(RtMesh* me)
{
    for (int s = 0; s < GFX_NSTREAMS; ++s)
        [me->c.vb[s] release];
    [me->c.ib release];
    rt_release_later(me->rtv), rt_release_later(me->blas);
    [me->tex release];
}

/* a buffer the game changed or let go of: the meshes drawn from it are captured again if drawn again */
static void rt_forget(id<MTLBuffer> buf)
{
    for (uint32_t i = 0; i < g_rtx.nmesh; ++i)
    {
        Caster* c = &g_rtx.mesh[i].c;
        int hit = c->ib == buf;
        for (int s = 0; s < GFX_NSTREAMS; ++s)
            hit |= c->vb[s] == buf;
        if (hit)
            g_rtx.mesh[i].dead = 1;
    }
}

/* keeps the places drawn in the last RT_KEEP frames (none when all) and the meshes they use, and
 * builds the lookups again */
static void rt_trim(int all)
{
    uint32_t* remap = (uint32_t*)malloc((g_rtx.nmesh + 1) * sizeof *remap);
    /* which meshes hold still: most of their draws since the last look on a place kept as it was */
    uint32_t top[5] = { 0 }, ntop = 0;
    for (uint32_t i = 0; i < g_rtx.nmesh; ++i)
    {
        RtMesh* me = &g_rtx.mesh[i];
        if (me->n_move + me->n_same >= 32)
            me->unstable = me->n_move > me->n_same;
        if (gfx_profiling && me->n_move)
        {
            uint32_t k = ntop < 5 ? ntop++ : 4;
            if (k == 4 && ntop == 5 && g_rtx.mesh[top[4]].n_move >= me->n_move)
                continue;
            top[k] = i;
            for (; k > 0 && g_rtx.mesh[top[k - 1]].n_move < g_rtx.mesh[top[k]].n_move; --k)
            {
                uint32_t t = top[k];
                top[k] = top[k - 1], top[k - 1] = t;
            }
        }
    }
    if (gfx_profiling && ntop && !all)
    {
        fprintf(stderr, "[recomp] gfx: rt: the meshes placed anew most (draws kept as they were / not, places, triangles, shader):");
        for (uint32_t k = 0; k < ntop; ++k)
        {
            RtMesh* me = &g_rtx.mesh[top[k]];
            fprintf(stderr, " %u/%u %u %u %08x%s;", me->n_same, me->n_move, me->nplace, me->tris, me->c.lib.vs.prog,
                me->unstable ? " unstable" : "");
        }
        fprintf(stderr, "\n");
    }
    for (uint32_t i = 0; i < g_rtx.nmesh; ++i)
        g_rtx.mesh[i].nplace = 0, g_rtx.mesh[i].n_move = g_rtx.mesh[i].n_same = 0;
    uint32_t np = 0;
    for (uint32_t i = 0; i < g_rtx.nplace; ++i)
    {
        RtPlace* p = &g_rtx.place[i];
        if (!all && !p->gone && p->seen + RT_KEEP >= g_serial && !g_rtx.mesh[p->mesh].unstable)
            g_rtx.place[np++] = *p, g_rtx.mesh[p->mesh].nplace++;
    }
    g_rtx.nplace = np;
    uint32_t nm = 0;
    for (uint32_t i = 0; i < g_rtx.nmesh; ++i)
    {
        RtMesh* me = &g_rtx.mesh[i];
        if (all || (!me->nplace && me->seen + RT_KEEP < g_serial))
        {
            rt_mesh_release(me);
            remap[i] = UINT32_MAX;
            continue;
        }
        remap[i] = nm;
        g_rtx.mesh[nm++] = *me;
    }
    g_rtx.nmesh = nm;
    uint32_t na = 0;
    for (uint32_t a = 0; a < g_rtx.nanchor; ++a)
        if (!all && remap[g_rtx.anchor[a].mesh] != UINT32_MAX)
            g_rtx.anchor[na] = g_rtx.anchor[a], g_rtx.anchor[na++].mesh = remap[g_rtx.anchor[a].mesh];
    g_rtx.nanchor = na;
    rt_map_free();
    for (uint32_t i = 0; i < nm; ++i)
    {
        RtMesh* me = &g_rtx.mesh[i];
        me->nplace = 0;
        CacheKey k;
        cache_key(&k, &me->c);
        map_put(&g_rtx.map, &k, sizeof k, (id)(uintptr_t)(i + 1));
    }
    for (uint32_t i = 0; i < np; ++i)
    {
        RtPlace* p = &g_rtx.place[i];
        p->mesh = remap[p->mesh];
        g_rtx.mesh[p->mesh].nplace++;
    }
    free(remap);
}

/* the caster's object space to its clip space (row vectors): the fixed function's wvp, or the four
 * constants a vertex shader's position transform reads; 0 when it is not one matrix */
static int rt_wvp_u(const GfxU* u, const GfxVsKey* vs, const uint32_t* tokens, float* out);

static int rt_wvp(const Caster* c, float* out)
{
    return rt_wvp_u((const GfxU*)((const uint8_t*)[c->ub contents] + c->uoff), &c->lib.vs, c->vs, out);
}

static int rt_wvp_u(const GfxU* u, const GfxVsKey* vs, const uint32_t* tokens, float* out)
{
    if (!vs->prog)
    {
        memcpy(out, u->wvp, 64);
        return 1;
    }
    int k = -1;
    uint32_t i = 0;
    for (; i < g_rtx.npos; ++i)
        if (g_rtx.pos_prog[i] == vs->prog)
            break;
    if (i < g_rtx.npos)
        k = g_rtx.pos_k[i];
    else
    {
        k = gfx_msl_vs1_pos_consts(tokens);
        if (g_rtx.npos < 64)
            g_rtx.pos_prog[g_rtx.npos] = vs->prog, g_rtx.pos_k[g_rtx.npos++] = k;
        fprintf(stderr, "[recomp] gfx: rt: vertex shader %08x places its vertices %s%d\n", vs->prog,
            k >= 0 ? "through c" : "some other way ", k);
    }
    if (k < 0 || k + 3 >= GFX_NVSC)
        return 0;
    for (int r = 0; r < 4; ++r)
        for (int col = 0; col < 4; ++col)
            out[r * 4 + col] = u->vsc[k + col][r];
    return 1;
}

static int rt_unit(float v) { return (int)floorf(fminf(fmaxf(v, -(float)RT_WORLD), (float)RT_WORLD)); }

static uint32_t rt_cell(uint32_t m, int x, int y, int z)
{
    uint32_t h = m * 0x9E3779B1u ^ (uint32_t)x * 0x85EBCA77u ^ (uint32_t)y * 0xC2B2AE3Du ^ (uint32_t)z * 0x27D4EB2Fu;
    return (h ^ (h >> 15)) & (g_rtx.nhead - 1);
}

static void rt_hash_add(uint32_t i)
{
    RtPlace* p = &g_rtx.place[i];
    uint32_t c = rt_cell(p->mesh, rt_unit(p->w[12]), rt_unit(p->w[13]), rt_unit(p->w[14]));
    p->next = g_rtx.head[c], g_rtx.head[c] = (int32_t)i;
}

/* the places by cell, again (a place that moved is found in its new one); room for twice as many */
static void rt_hash(void)
{
    uint32_t want = 1024;
    while (want < 4 * (g_rtx.nplace + 4096))
        want *= 2;
    if (want != g_rtx.nhead)
        g_rtx.nhead = want, g_rtx.head = (int32_t*)realloc(g_rtx.head, want * sizeof *g_rtx.head);
    memset(g_rtx.head, 0xFF, g_rtx.nhead * sizeof *g_rtx.head);
    for (uint32_t i = 0; i < g_rtx.nplace; ++i)
        rt_hash_add(i);
}

/* where this frame's draw of mesh m (object to world w) is: a place already kept there, one drawn the
 * frame traced before that has since moved a little (a door opening), or a new one */
static int32_t rt_place(uint32_t m, const float* w)
{
    RtMesh* me = &g_rtx.mesh[m];
    int moved = -1;
    float best = 0.25f, bestd = 0.0f, nd = 1e9f, nt = 1e9f;
    int cx = rt_unit(w[12]), cy = rt_unit(w[13]), cz = rt_unit(w[14]);
    for (int k = 0; k < 27; ++k)
    {
        int x = cx + k % 3 - 1, y = cy + (k / 3) % 3 - 1, z = cz + k / 9 - 1;
        uint32_t c = rt_cell(m, x, y, z);
        for (int32_t i = g_rtx.head[c]; i >= 0; i = g_rtx.place[i].next)
        {
            RtPlace* p = &g_rtx.place[i];
            if (p->mesh != m || p->gone)
                continue;
            float d = 0.0f;
            for (int j = 0; j < 12; ++j)
                d = fmaxf(d, fabsf(p->w[j] - w[j]) * ((j & 3) == 3 ? 0.0f : 1.0f));
            float dx = p->w[12] - w[12], dy = p->w[13] - w[13], dz = p->w[14] - w[14], t = dx * dx + dy * dy + dz * dz;
            if (t < nt)
                nt = t, nd = d;
            /* placed already this frame, near enough: the same mesh drawn again there (another pass, a
             * second target drawn with the camera a moment apart) */
            if (p->seen == g_serial)
            {
                if (d < 1e-2f && t < 0.25f)
                {
                    g_rtx.pl_samefr++;
                    return -1;
                }
                continue;
            }
            if (d < 1e-2f && t < best)
                best = t, moved = i, bestd = d;
        }
    }
    /* the nearest place kept: as it was when within the rounding of the game's float transforms (a
     * hundredth of a unit), else moved there with it (a door, a branch swaying) */
    if (moved >= 0)
    {
        RtPlace* p = &g_rtx.place[moved];
        p->seen = g_serial, p->missed = 0;
        if (best > 1e-4f || bestd > 1e-4f)
            memcpy(p->w, w, 64), g_rtx.pl_moved++, me->n_move++;
        else
            g_rtx.pl_same++, me->n_same++;
        return moved;
    }
    if (me->nplace >= RT_PLACES)
    {
        g_rtx.pl_full++;
        return -1;
    }
    g_rtx.pl_new++, me->n_move++;
    if (nt < 1e9f)
        g_rtx.pl_nd = nd, g_rtx.pl_nt = sqrtf(nt);
    if (g_rtx.nplace == g_rtx.cap_place)
        g_rtx.cap_place = g_rtx.cap_place ? g_rtx.cap_place * 2 : 1024,
        g_rtx.place = (RtPlace*)realloc(g_rtx.place, g_rtx.cap_place * sizeof *g_rtx.place);
    RtPlace* p = &g_rtx.place[g_rtx.nplace];
    p->mesh = m, p->seen = g_serial, p->missed = 0, p->gone = 0;
    memcpy(p->w, w, 64);
    me->nplace++;
    int32_t at = (int32_t)g_rtx.nplace;
    if (4 * (g_rtx.nplace + 1) > g_rtx.nhead) /* the table's too full: bigger, with every place again */
        g_rtx.nplace++, rt_hash();
    else
        rt_hash_add(g_rtx.nplace++);
    return at;
}

/* The same in double precision, for the places: a draw's transform through the inverse of the camera's
 * (the zone's coordinates run to thousands of units, and a projection inverted in float loses tenths
 * of a unit there - a place would move a little with every turn of the camera) */
static int matd_inverse(double* out, const double* m)
{
    double inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (det == 0.0)
        return 0;
    for (int i = 0; i < 16; ++i)
        out[i] = inv[i] / det;
    return 1;
}

static void matd_mul(double* o, const double* a, const double* b)
{
    double t[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            t[i * 4 + j] = a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j] + a[i * 4 + 2] * b[8 + j] + a[i * 4 + 3] * b[12 + j];
    memcpy(o, t, sizeof t);
}

/* the mesh a fixed caster draws, kept from before or new */
static uint32_t rt_mesh(const Caster* c)
{
    CacheKey k;
    cache_key(&k, c);
    uintptr_t at = (uintptr_t)map_get(&g_rtx.map, &k, sizeof k);
    if (at)
    {
        RtMesh* me = &g_rtx.mesh[at - 1];
        if (me->dead) /* new contents: captured again */
        {
            rt_release_later(me->rtv), rt_release_later(me->blas);
            me->rtv = nil, me->blas = nil, me->dead = 0, me->none = 0;
        }
        return (uint32_t)(at - 1);
    }
    if (g_rtx.nmesh == g_rtx.cap_mesh)
        g_rtx.cap_mesh = g_rtx.cap_mesh ? g_rtx.cap_mesh * 2 : 1024,
        g_rtx.mesh = (RtMesh*)realloc(g_rtx.mesh, g_rtx.cap_mesh * sizeof *g_rtx.mesh);
    RtMesh* me = &g_rtx.mesh[g_rtx.nmesh];
    memset(me, 0, sizeof *me);
    me->c = *c;
    for (int s = 0; s < GFX_NSTREAMS; ++s)
        [me->c.vb[s] retain];
    [me->c.ib retain];
    me->c.ub = nil;
    memset(me->c.tex, 0, sizeof me->c.tex);
    map_put(&g_rtx.map, &k, sizeof k, (id)(uintptr_t)(g_rtx.nmesh + 1));
    return g_rtx.nmesh++;
}

/* the mesh's box in its own space, once the GPU has written its triangles; 0 until then */
static int rt_bounds(RtMesh* me)
{
    if (me->bounds || !me->rtv || !me->blas || me->captured > atomic_load(&g_completed))
        return me->bounds;
    const float* v = (const float*)[me->rtv contents];
    uint32_t n = me->tris * 3;
    for (int k = 0; k < 3; ++k)
        me->lo[k] = 1e30f, me->hi[k] = -1e30f;
    for (uint32_t i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k)
            me->lo[k] = fminf(me->lo[k], v[12 * i + k]), me->hi[k] = fmaxf(me->hi[k], v[12 * i + k]);
    me->bounds = n && me->lo[0] <= me->hi[0];
    return me->bounds;
}

/* whether a mesh's box, placed by w, is in the camera's view (vp: world to clip, row vectors) - well
 * inside it, so that the game, culling by bounds of its own, surely draws what is there */
static int rt_in_view(const RtMesh* me, const float* w, const float* vp)
{
    float c[8][4];
    for (int i = 0; i < 8; ++i)
    {
        float o[3] = { (i & 1) ? me->hi[0] : me->lo[0], (i & 2) ? me->hi[1] : me->lo[1], (i & 4) ? me->hi[2] : me->lo[2] }, p[3];
        for (int j = 0; j < 3; ++j)
            p[j] = o[0] * w[j] + o[1] * w[4 + j] + o[2] * w[8 + j] + w[12 + j];
        for (int j = 0; j < 4; ++j)
            c[i][j] = p[0] * vp[j] + p[1] * vp[4 + j] + p[2] * vp[8 + j] + vp[12 + j];
    }
    /* out when every corner is beyond one plane: x, y within 0.8 w, w at least 1 */
    for (int pl = 0; pl < 5; ++pl)
    {
        int out = 1;
        for (int i = 0; i < 8 && out; ++i)
        {
            float x = c[i][0], y = c[i][1], ww = c[i][3];
            out = pl == 0 ? x < -0.8f * ww : pl == 1 ? x > 0.8f * ww : pl == 2 ? y < -0.8f * ww : pl == 3 ? y > 0.8f * ww : ww < 1.0f;
        }
        if (out)
            return 0;
    }
    return 1;
}

/* This frame's camera as the zone's shaders drew with it (vpinv: clip space to the world): each anchor
 * drawn once gives one - its transform through the inverse of where it stands - and the one most of
 * them agree on is taken (an anchor that moved, a door, disagrees and is let go; one not drawn for ten
 * frames gives way to what is drawn now, in the same reference). The scene's camera
 * (the first fogged fixed-function draw's view) is not quite it: placed through that, the zone shifted
 * a little with every turn of the camera. 0 when no anchor was drawn: vpinv is left as it was. */
static int rt_camera(double* vpinv)
{
    double cand[8][16], cam[8][3];
    uint32_t idx[8], n = 0;
    for (uint32_t a = 0; a < g_rtx.nanchor; ++a)
    {
        RtMesh* me = &g_rtx.mesh[g_rtx.anchor[a].mesh];
        double iw[16], vpc[16];
        if (me->drawn != g_serial || me->ndrawn != 1 || !matd_inverse(iw, g_rtx.anchor[a].w))
            continue;
        /* its transform this frame: object to clip = (object to world) (world to clip) */
        double wvp[16];
        memcpy(wvp, me->wd, sizeof wvp); /* me->wd holds this frame's object to clip (rt_frame's first pass) */
        matd_mul(vpc, iw, wvp);
        if (!matd_inverse(cand[n], vpc) || fabs(cand[n][11]) < 1e-12)
            continue;
        for (int k = 0; k < 3; ++k)
            cam[n][k] = cand[n][8 + k] / cand[n][11];
        idx[n++] = a;
        g_rtx.anchor[a].seen = g_serial;
    }
    if (!n)
    {
        /* none drawn: the ones not drawn for ten frames give way (new ones are found this frame) */
        uint32_t keep = 0;
        for (uint32_t a = 0; a < g_rtx.nanchor; ++a)
            if (g_rtx.anchor[a].seen + 10 >= g_serial)
                g_rtx.anchor[keep++] = g_rtx.anchor[a];
        g_rtx.nanchor = keep;
        return 0;
    }
    uint32_t best = 0, most = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        uint32_t agree = 0;
        for (uint32_t j = 0; j < n; ++j)
        {
            double dx = cam[i][0] - cam[j][0], dy = cam[i][1] - cam[j][1], dz = cam[i][2] - cam[j][2];
            agree += dx * dx + dy * dy + dz * dz < 1e-4;
        }
        if (agree > most)
            most = agree, best = i;
    }
    /* the ones that disagree have moved: no longer anchors */
    uint32_t keep = 0;
    uint8_t drop[8] = { 0 };
    for (uint32_t j = 0; j < n; ++j)
    {
        double dx = cam[best][0] - cam[j][0], dy = cam[best][1] - cam[j][1], dz = cam[best][2] - cam[j][2];
        if (!(dx * dx + dy * dy + dz * dz < 1e-4) && n >= 3)
            drop[idx[j]] = 1;
    }
    for (uint32_t a = 0; a < g_rtx.nanchor; ++a)
        if (!drop[a] && g_rtx.anchor[a].seen + 10 >= g_serial) /* not drawn for a while: others in its stead */
            g_rtx.anchor[keep++] = g_rtx.anchor[a];
    g_rtx.nanchor = keep;
    /* the mean of the ones that agree (one alone jumps by its own rounding as the choice changes) */
    double sum[16] = { 0 }, spread = 0.0;
    uint32_t na = 0;
    for (uint32_t j = 0; j < n; ++j)
    {
        double dx = cam[best][0] - cam[j][0], dy = cam[best][1] - cam[j][1], dz = cam[best][2] - cam[j][2], d2 = dx * dx + dy * dy + dz * dz;
        if (!(d2 < 1e-4))
            continue;
        spread = d2 > spread ? d2 : spread;
        for (int k = 0; k < 16; ++k)
            sum[k] += cand[j][k] / cand[j][11]; /* each scaled alike: row 2's w is 1 */
        na++;
    }
    for (int k = 0; k < 16; ++k)
        vpinv[k] = sum[k] / na;
    /* the profile: how far apart the agreeing ones are, and from the scene's camera */
    g_rtx.cam_spread = fmax(g_rtx.cam_spread, sqrt(spread));
    g_rtx.cam_agree += na;
    return 1;
}

/* This frame's camera from the last traced one's (vpinv: clip to the world): a mesh drawn once in both,
 * still, went from clip_prev = W VP_last to clip = W VP, so VP = VP_last clip_prev^-1 clip whatever W
 * is. The one most of them agree on (what moved disagrees); 0 when fewer than two agree. */
static int rt_chain(double* vpinv)
{
    if (!g_rtx.have_vp_last || !g_rtx.prev)
        return 0;
    enum { N = 256 };
    static double cand[N][16], cam[N][3];
    uint32_t n = 0;
    /* a mesh drawn as many times (up to sixteen) in both: its draws paired in the order drawn (a pairing
     * that is wrong, or a draw that moved, disagrees with the rest) */
    g_rtx.ch_calls++;
    for (uint32_t i = 0; i < g_rtx.nmesh && n < N; ++i)
    {
        const RtMesh* me = &g_rtx.mesh[i];
        if (me->clip_serial == g_serial && me->nclip)
        {
            g_rtx.ch_seen++;
            g_rtx.ch_samecount += me->nclip == me->nclip_prev;
            g_rtx.ch_prev += me->clip_prev_serial == g_rtx.vp_last_serial;
        }
        /* (what the places call unstable too: a camera off makes them so, and this does not use them) */
        if (me->clip_serial != g_serial || !me->nclip || me->nclip > 16 || me->nclip != me->nclip_prev ||
            me->clip_prev_serial != g_rtx.vp_last_serial)
            continue;
        for (uint32_t d = 0; d < me->nclip && n < N; ++d)
        {
            double a[16], b[16], ia[16], step[16], vp[16];
            for (int j = 0; j < 16; ++j)
                a[j] = me->clip_prev[d][j], b[j] = me->clip[d][j];
            if (!matd_inverse(ia, a))
                continue;
            matd_mul(step, ia, b);
            matd_mul(vp, g_rtx.vp_last, step);
            if (!matd_inverse(cand[n], vp) || fabs(cand[n][11]) < 1e-12)
                continue;
            for (int k = 0; k < 3; ++k)
                cam[n][k] = cand[n][8 + k] / cand[n][11];
            n++;
        }
    }
    uint32_t best = 0, most = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        uint32_t agree = 0;
        for (uint32_t j = 0; j < n; ++j)
        {
            double dx = cam[i][0] - cam[j][0], dy = cam[i][1] - cam[j][1], dz = cam[i][2] - cam[j][2];
            agree += dx * dx + dy * dy + dz * dz < 1e-4;
        }
        if (agree > most)
            most = agree, best = i;
    }
    g_rtx.ch_cand += n, g_rtx.ch_most += most;
    if (most < 2)
        return 0;
    double sum[16] = { 0 };
    for (uint32_t j = 0; j < n; ++j)
    {
        double dx = cam[best][0] - cam[j][0], dy = cam[best][1] - cam[j][1], dz = cam[best][2] - cam[j][2];
        if (dx * dx + dy * dy + dz * dz < 1e-4)
            for (int k = 0; k < 16; ++k)
                sum[k] += cand[j][k] / cand[j][11];
    }
    for (int k = 0; k < 16; ++k)
        vpinv[k] = sum[k] / most;
    return 1;
}

/* the profile: the most places one mesh has */
static uint32_t rt_most_places(void)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < g_rtx.nmesh; ++i)
        n = g_rtx.mesh[i].nplace > n ? g_rtx.mesh[i].nplace : n;
    return n;
}

/* --- the sun's shadow maps, from the kept world (rt_sun_maps) --------------------------------------------
 * Each scene's end, the world the ray tracing keeps - every kept mesh at each of its places (in view or
 * not), what is placed anew each frame and the characters - is drawn from the sun into two depth maps:
 * a near one round the camera (rt_near units) and a far one (rt_distance), each a sphere round its
 * slice of the camera's view, its center snapped to whole texels (the edges hold still as the camera
 * moves). Everything goes camera-relative, in double precision until it is small: the zone's
 * coordinates run to thousands of units. Alpha-tested meshes let through what their test does. The
 * world's own draws of the next frame look themselves up in them (the rt keys, gfx_msl.c). */
enum { RT_MAP = 4096, RT_CASCADES = 3 };

static const char SM_MSL[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct SmO { float4 pos [[position]]; float2 uv; };\n"
    /* each vertex a position, a normal and texture coordinates (the capture's); w: the instance's object
     * to the world less the camera's place; s: that to the map */
    "vertex SmO sm_vs(uint vid [[vertex_id]], uint iid [[instance_id]], device const float4* v [[buffer(0)]],\n"
    "                 device const float4x4* w [[buffer(1)]], constant float4x4& s [[buffer(2)]]) {\n"
    "  SmO o;\n"
    "  o.pos = s * (w[iid] * float4(v[3 * vid].xyz, 1.0));\n"
    "  o.uv = v[3 * vid + 2].xy;\n"
    "  return o;\n"
    "}\n"
    "fragment void sm_fs(SmO in [[stage_in]], texture2d<float> t [[texture(0)]], sampler sp [[sampler(0)]],\n"
    "                    constant float& aref [[buffer(0)]]) {\n"
    "  if (t.sample(sp, in.uv).a < aref) discard_fragment();\n"
    "}\n";

typedef struct SmCascade
{
    float m[16];         /* the world less the camera's place to the map: x, y -1..1, z 0..1 */
    double texel, range; /* a texel in world units; world units over the map's depth */
    double reach;        /* how far from the camera it is used */
} SmCascade;

static struct
{
    int tried, ok;
    id<MTLRenderPipelineState> solid, alpha;
    id<MTLTexture> map;    /* a layer a cascade, nearest first */
    id<MTLTexture> scol;   /* the colour the casters' pipelines are built with (never kept) */
    SmCascade c[RT_CASCADES];
    int nc;                /* the cascades of the last maps */
    double cam[3]; /* the camera the maps' matrices are relative to (their scene's) */
    uint32_t drawn, instances; /* the profile: draws and instances of the last maps */
} g_sm;

static int sm_init(void)
{
    if (g_sm.tried)
        return g_sm.ok || g_sm.solid;
    g_sm.tried = 1;
    id<MTLLibrary> lib = compile(SM_MSL);
    if (!lib)
        return 0;
    id<MTLFunction> vf = [lib newFunctionWithName:@"sm_vs"], ff = [lib newFunctionWithName:@"sm_fs"];
    MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vf;
    pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    NSError* err = nil;
    g_sm.solid = [g_dev newRenderPipelineStateWithDescriptor:pd error:&err];
    pd.fragmentFunction = ff;
    g_sm.alpha = [g_dev newRenderPipelineStateWithDescriptor:pd error:&err];
    [pd release], [vf release], [ff release], [lib release];
    {
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                      width:RT_MAP height:RT_MAP mipmapped:NO];
        td.textureType = MTLTextureType2DArray;
        td.arrayLength = RT_CASCADES;
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModePrivate;
        g_sm.map = [g_dev newTextureWithDescriptor:td];
        MTLTextureDescriptor* cd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                      width:RT_MAP height:RT_MAP mipmapped:NO];
        cd.usage = MTLTextureUsageRenderTarget;
        cd.storageMode = MTLStorageModeMemoryless;
        g_sm.scol = [g_dev newTextureWithDescriptor:cd];
    }
    if (!g_sm.solid || !g_sm.alpha || !g_sm.map || !g_sm.scol)
    {
        fprintf(stderr, "[recomp] gfx: rt: shadow maps failed: %s\n", err ? [[err localizedDescription] UTF8String] : "?");
        [g_sm.solid release], g_sm.solid = nil;
        return 0;
    }
    return 1;
}

/* clip space (x, y, z) through clip-to-world vpinv to the world */
static int rt_unproject(const double* vpinv, double x, double y, double z, double* o)
{
    double h[4];
    for (int j = 0; j < 4; ++j)
        h[j] = x * vpinv[j] + y * vpinv[4 + j] + z * vpinv[8 + j] + vpinv[12 + j];
    if (fabs(h[3]) < 1e-12)
        return 0;
    for (int j = 0; j < 3; ++j)
        o[j] = h[j] / h[3];
    return 1;
}

static void v3_normalize(double* v)
{
    double l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 0.0)
        v[0] /= l, v[1] /= l, v[2] /= l;
}

/* a map over the slice [t0, t1] of the camera's view (units ahead of it): the sphere round the slice,
 * seen along the sun (toward it: L), its center on whole texels; 200 units sunward of the sphere
 * still cast into it (a cliff above the camera) */
static int rt_cascade(const double* vpinv, const double* cam, const double* L, double t0, double t1, SmCascade* k)
{
    double c0[3], f[3], p[8][3], c[3] = { 0, 0, 0 };
    if (!rt_unproject(vpinv, 0, 0, 0.5, c0))
        return 0;
    for (int j = 0; j < 3; ++j)
        f[j] = c0[j] - cam[j];
    v3_normalize(f);
    for (int i = 0; i < 4; ++i)
    {
        double a[3], d[3];
        if (!rt_unproject(vpinv, (i & 1) ? 1.0 : -1.0, (i & 2) ? 1.0 : -1.0, 0.5, a))
            return 0;
        for (int j = 0; j < 3; ++j)
            d[j] = a[j] - cam[j];
        v3_normalize(d);
        double sc = 1.0 / fmax(d[0] * f[0] + d[1] * f[1] + d[2] * f[2], 1e-3);
        for (int j = 0; j < 3; ++j)
        {
            p[i][j] = cam[j] + d[j] * t0 * sc, p[4 + i][j] = cam[j] + d[j] * t1 * sc;
            c[j] += (p[i][j] + p[4 + i][j]) / 8.0;
        }
    }
    double R = 0.0;
    for (int i = 0; i < 8; ++i)
    {
        double dx = p[i][0] - c[0], dy = p[i][1] - c[1], dz = p[i][2] - c[2];
        R = fmax(R, sqrt(dx * dx + dy * dy + dz * dz));
    }
    R = ceil(R); /* whole units: the texel's size holds still */
    double fl[3] = { -L[0], -L[1], -L[2] }, up[3] = { 0, 1, 0 };
    if (fabs(fl[1]) > 0.9)
        up[0] = 1, up[1] = 0;
    double r[3] = { up[1] * fl[2] - up[2] * fl[1], up[2] * fl[0] - up[0] * fl[2], up[0] * fl[1] - up[1] * fl[0] };
    v3_normalize(r);
    double u[3] = { fl[1] * r[2] - fl[2] * r[1], fl[2] * r[0] - fl[0] * r[2], fl[0] * r[1] - fl[1] * r[0] };
    double tx = 2.0 * R / RT_MAP;
    double cx = floor((c[0] * r[0] + c[1] * r[1] + c[2] * r[2]) / tx) * tx;
    double cy = floor((c[0] * u[0] + c[1] * u[1] + c[2] * u[2]) / tx) * tx;
    double cz = c[0] * fl[0] + c[1] * fl[1] + c[2] * fl[2];
    double back = 200.0, range = 2.0 * R + back, z0 = cz - R - back;
    float m[16] = {
        (float)(r[0] / R), (float)(u[0] / R), (float)(fl[0] / range), 0,
        (float)(r[1] / R), (float)(u[1] / R), (float)(fl[1] / range), 0,
        (float)(r[2] / R), (float)(u[2] / R), (float)(fl[2] / range), 0,
        (float)(((cam[0] * r[0] + cam[1] * r[1] + cam[2] * r[2]) - cx) / R),
        (float)(((cam[0] * u[0] + cam[1] * u[1] + cam[2] * u[2]) - cy) / R),
        (float)(((cam[0] * fl[0] + cam[1] * fl[1] + cam[2] * fl[2]) - z0) / range), 1,
    };
    memcpy(k->m, m, sizeof m);
    k->texel = tx, k->range = range;
    return 1;
}

/* the frame's casters into one cascade's map (rel: its camera's clip space to the world less the
 * camera's place cam) */
static void rt_map_casters(uint32_t layer, const SmCascade* k, const double* rel, const double* cam)
{
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.depthAttachment.texture = g_sm.map;
    rp.depthAttachment.slice = layer;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    rp.depthAttachment.clearDepth = 1.0;
    rp.depthAttachment.storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].texture = g_sm.scol;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionDontCare;
    id<MTLRenderCommandEncoder> e = [cmd() renderCommandEncoderWithDescriptor:rp];
    [e setViewport:(MTLViewport){ 0, 0, RT_MAP, RT_MAP, 0, 1 }];
    [e setDepthStencilState:g_fx.sdepth];
    [e setCullMode:MTLCullModeNone];
    [e setDepthBias:0 slopeScale:1.5f clamp:0];
    double km[16], md[16];
    for (int j = 0; j < 16; ++j)
        km[j] = k->m[j];
    float M[16];
    matd_mul(md, rel, km); /* the frame's clip space -> the map */
    for (int j = 0; j < 16; ++j)
        M[j] = (float)md[j];
    int late = g_late_serial == g_serial;
    uint32_t total = g_ncasters + (late ? g_nlate : 0);
    for (uint32_t i = 0; i < total; ++i)
    {
        const Caster* cs = i < g_ncasters ? &g_casters[i] : &g_late[i - g_ncasters];
        if (cs->has_cw) /* fixed function: its own view back to the world */
        {
            double cw[16], m2[16];
            for (int j = 0; j < 16; ++j)
                cw[j] = cs->cw[j];
            for (int r = 0; r < 4; ++r)
                for (int j = 0; j < 3; ++j)
                    cw[r * 4 + j] -= cam[j] * cw[r * 4 + 3];
            matd_mul(m2, cw, km);
            float m[16];
            for (int j = 0; j < 16; ++j)
                m[j] = (float)m2[j];
            [e setVertexBytes:m length:64 atIndex:5];
        }
        else
            [e setVertexBytes:M length:64 atIndex:5];
        PipeKey pk;
        memset(&pk, 0, sizeof pk);
        pk.lib = cs->lib;
        pk.lib.vs.shadow = 1, pk.lib.vs.pixel = 0, pk.lib.vs.rt = 0, pk.lib.fs.rt = 0;
        int at = alpha_tested(&cs->lib.fs);
        if (!at)
        {
            /* the position alone: one pipeline serves every draw with the same vertex layout */
            GfxVsKey* v = &pk.lib.vs;
            v->lighting = v->normalize = v->localviewer = v->specular = 0;
            v->src_diffuse = v->src_specular = v->src_ambient = v->src_emissive = 0;
            v->nlights = 0, memset(v->light_type, 0, sizeof v->light_type);
            v->fog_vertex = v->range_fog = 0, v->ntex = 0, v->flat = 0;
            memset(v->tci, 0, sizeof v->tci), memset(v->ttf, 0, sizeof v->ttf);
            memset(&pk.lib.fs, 0, sizeof pk.lib.fs);
        }
        pk.color = (uint32_t)MTLPixelFormatR8Unorm, pk.depth = (uint32_t)MTLPixelFormatDepth32Float;
        id<MTLRenderPipelineState> ps = pipeline_for(&pk, cs->vs, cs->ps);
        if (!ps)
            continue;
        [e setRenderPipelineState:ps];
        for (int st = 0; st < GFX_NSTREAMS; ++st)
            [e setVertexBuffer:cs->vb[st] offset:cs->voff[st] atIndex:(NSUInteger)st];
        [e setVertexBuffer:cs->ub offset:cs->uoff atIndex:4];
        if (at)
        {
            [e setFragmentBuffer:cs->ub offset:cs->uoff atIndex:4];
            for (int t = 0; t < 8; ++t)
                if (cs->tex[t])
                {
                    GfxSampler sk = cs->samp[t];
                    [e setFragmentTexture:cs->tex[t] atIndex:(NSUInteger)t];
                    [e setFragmentSamplerState:sampler(&sk) atIndex:(NSUInteger)t];
                }
        }
        if (cs->itype)
            [e drawIndexedPrimitives:cs->prim indexCount:cs->n
                           indexType:cs->itype == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                         indexBuffer:cs->ib indexBufferOffset:cs->ioff];
        else
            [e drawPrimitives:cs->prim vertexStart:cs->vstart vertexCount:cs->n];
        g_sm.drawn++;
    }
    [e endEncoding];
}

static void rt_cam_log(const char* what);

/* this scene's maps (vpinv: its camera, clip to the world); the draws of the next look themselves up */
static void rt_sun_maps(const double* vpinv, id<MTLBuffer> dyn_buf, NSUInteger dyn_off, uint32_t dyn_verts)
{
    g_sm.ok = 0;
    if (!rt_inpass() || g_rtx.sun_now[3] <= 0.0f || g_rtx.k_now[0] <= 0.0f || !sm_init() || fabs(vpinv[11]) < 1e-12)
        return;
    double cam[3] = { vpinv[8] / vpinv[11], vpinv[9] / vpinv[11], vpinv[10] / vpinv[11] };
    memcpy(g_sm.cam, cam, sizeof cam);
    {
        char w[96];
        snprintf(w, sizeof w, "maps at %.3f %.3f %.3f", cam[0], cam[1], cam[2]);
        rt_cam_log(w);
    }
    double L[3] = { g_rtx.sun_now[0], g_rtx.sun_now[1], g_rtx.sun_now[2] };
    v3_normalize(L);
    /* the cascades' reach: rt_near, rt_distance and one between (their geometric mean); each a sphere
     * round its slice of the view from half the one before's reach */
    double dfar = fmax(g_fxs.rt_distance, 8.0), dnear = fmin(fmax(g_fxs.rt_near, 2.0), dfar / 4.0);
    double reach[RT_CASCADES] = { dnear, sqrt(dnear * dfar), dfar };
    g_sm.nc = 0;
    for (int i = 0; i < RT_CASCADES; ++i)
    {
        if (!rt_cascade(vpinv, cam, L, i ? 0.5 * reach[i - 1] : 0.5, reach[i], &g_sm.c[i]))
            return;
        g_sm.c[i].reach = reach[i];
    }
    g_sm.nc = RT_CASCADES;
    /* what casts: the frame's own draws of the world, drawn again from the sun - each through what
     * placed it on the screen and back: a zone shader's through the frame's camera (the draws of the
     * next frame come back to the world through the same one, so where they fall on the map agrees
     * with where the casters went, whatever small error that camera has), a fixed-function draw's
     * through its own view. Nothing kept from frames before: no copies left where the game no longer
     * draws (a nearer or farther detail of a mesh), nothing placed wrong. */
    double rel[16];
    memcpy(rel, vpinv, sizeof rel);
    for (int r = 0; r < 4; ++r)
        for (int j = 0; j < 3; ++j)
            rel[r * 4 + j] -= cam[j] * rel[r * 4 + 3];
    g_sm.drawn = 0, g_sm.instances = g_ncasters;
    for (int c = 0; c < g_sm.nc; ++c)
        rt_map_casters((uint32_t)c, &g_sm.c[c], rel, cam);
    g_sm.ok = 1;
}

/* The structures for this frame's scene (camera: clip_world, its view to the world view_world); 0 when
 * there is nothing to trace. Each fixed caster places its mesh (object space to the world: its own
 * transform through the camera's inverse); a mesh new to it is captured in its own space (clip space
 * back through the inverse of that transform). The rest - what moves, or a shader that places its
 * vertices some other way - is captured in the world, every frame. */
static int rt_frame(const float* clip_world, const float* view_world, const float* view, const float* proj, int fresh)
{
    if (!rt_init())
        return 0;
    if (gfx_profiling)
        rt_projections(proj);
    /* the camera 50 units from the last: a new zone - when no anchor is drawn either (below): the scene's
     * camera alone is now and then another draw's view, hundreds of units off */
    const float* cam = view_world + 12;
    float jx = cam[0] - g_rtx.cam[0], jy = cam[1] - g_rtx.cam[1], jz = cam[2] - g_rtx.cam[2];
    int jumped = jx * jx + jy * jy + jz * jz > 50.0f * 50.0f;
    memcpy(g_rtx.cam, cam, 12);
    if (!(g_serial & 127))
        rt_trim(0);
    rt_hash();
    g_rtx.nnow = 0;
    float vp[16];
    mat_mul(vp, view, proj);
    double vpd[16], vpinv[16];
    for (int j = 0; j < 16; ++j)
        vpd[j] = vp[j];
    {
        double vd[16], pd[16];
        for (int j = 0; j < 16; ++j)
            vd[j] = view[j], pd[j] = proj[j];
        matd_mul(vpd, vd, pd);
    }
    int have_vpinv = matd_inverse(vpinv, vpd);
    /* each fixed mesh's draws this frame and the last one's transform (object to clip, for rt_camera) */
    for (uint32_t i = 0; i < g_ncasters; ++i)
    {
        const Caster* c = &g_casters[i];
        float wvp[16];
        if (!c->fixed || !rt_wvp(c, wvp))
            continue;
        uint32_t mi = rt_mesh(c); /* first: it may move the meshes */
        RtMesh* me = &g_rtx.mesh[mi];
        if (me->drawn != g_serial)
            me->drawn = g_serial, me->ndrawn = 0, me->order = i;
        me->ndrawn++;
        for (int j = 0; j < 16; ++j)
            me->wd[j] = wvp[j];
        if (me->clip_serial != g_serial)
        {
            memcpy(me->clip_prev, me->clip, sizeof me->clip);
            me->clip_prev_serial = me->clip_serial, me->nclip_prev = me->nclip;
            me->clip_serial = g_serial, me->nclip = 0;
        }
        if (me->nclip < 16)
            memcpy(me->clip[me->nclip], wvp, 64);
        me->nclip++;
    }
    double scene_inv[16];
    memcpy(scene_inv, vpinv, sizeof scene_inv);
    int anchored = have_vpinv && rt_camera(vpinv);
    /* no anchor drawn: the last frame's camera moved on as the meshes drawn in both moved (its own step,
     * the world's transforms cancelling), else the camera the draws found this frame (a mesh at its one
     * place) - in the same reference, before the scene's own, which can be far off and would start anew */
    if (!anchored && have_vpinv && rt_chain(vpinv))
        anchored = 1, g_rtx.cam_chain++;
    if (!anchored && g_rtx.fvp_vp_serial == g_serial && matd_inverse(vpinv, g_rtx.fvp_vp))
        anchored = 1, g_rtx.cam_from_draws++;
    /* none of those: the last camera moved on as it moved the frame before - the same reference, off by
     * little - unless the scene's camera is its own this frame (a fixed-function draw's: exact), or it
     * jumped (a new zone): an earlier scene's is far off, and taken it would start another reference,
     * every place kept then misplaced */
    if (!anchored && fresh)
        g_rtx.cam_fresh++; /* the scene's own camera, found this frame: as good as an anchor */
    else if (!anchored && !jumped && g_rtx.have_vp_last && g_rtx.vp_last_serial + 30 >= g_serial)
    {
        double guess[16], ib[16], step[16];
        memcpy(guess, g_rtx.vp_last, sizeof guess);
        if (g_rtx.have_vp_prev && g_rtx.vp_last_serial + 1 == g_serial && matd_inverse(ib, g_rtx.vp_prev))
            matd_mul(step, g_rtx.vp_last, ib), matd_mul(guess, step, g_rtx.vp_last);
        if (matd_inverse(vpinv, guess))
            anchored = 1, g_rtx.cam_guessed++;
    }
    if (anchored && fabs(scene_inv[11]) > 1e-12 && fabs(vpinv[11]) > 1e-12)
    {
        double d = 0.0;
        for (int k = 0; k < 3; ++k)
            d += pow(scene_inv[8 + k] / scene_inv[11] - vpinv[8 + k] / vpinv[11], 2.0);
        g_rtx.cam_scene = fmax(g_rtx.cam_scene, sqrt(d));
    }
    g_rtx.anchored += anchored, g_rtx.anchor_fallback += !anchored;
    if (!anchored) /* no anchor drawn: a new frame of reference (the scene's camera), anchors found anew */
    {
        g_rtx.nanchor = 0;
        if (jumped) /* and the camera far from the last: a new zone, all of the last one's gone */
        {
            rt_trim(1), rt_hash(), g_rtx.zones++;
            for (uint32_t i = 0; i < g_ncasters; ++i) /* this frame's draws counted again, of new meshes */
            {
                const Caster* c = &g_casters[i];
                float wvp[16];
                if (!c->fixed || !rt_wvp(c, wvp))
                    continue;
                uint32_t mi = rt_mesh(c);
                RtMesh* me = &g_rtx.mesh[mi];
                if (me->drawn != g_serial)
                    me->drawn = g_serial, me->ndrawn = 0;
                me->ndrawn++;
            }
        }
    }
    /* the frame's camera, from here on: the world to clip (the places' culling), clip to the world (what
     * moves), and for the rays */
    if (anchored)
    {
        double vpa[16];
        if (matd_inverse(vpa, vpinv))
            for (int j = 0; j < 16; ++j)
                vp[j] = (float)vpa[j];
    }
    float cwf[16];
    for (int j = 0; j < 16; ++j)
        cwf[j] = (float)vpinv[j];
    clip_world = cwf;
    /* the next frame's draws start from this camera, moved on as it moved from the one before
     * (rt_frame_camera) */
    {
        double vpn[16];
        if (have_vpinv && matd_inverse(vpn, vpinv))
        {
            g_rtx.have_vp_prev = g_rtx.have_vp_last && g_rtx.vp_last_serial + 1 == g_serial;
            memcpy(g_rtx.vp_prev, g_rtx.vp_last, sizeof vpn);
            memcpy(g_rtx.vp_last, vpn, sizeof vpn);
            g_rtx.have_vp_last = 1, g_rtx.vp_last_serial = g_serial;
        }
    }
    if (have_vpinv && fabs(vpinv[11]) > 1e-12)
    {
        double cam3[3] = { vpinv[8] / vpinv[11], vpinv[9] / vpinv[11], vpinv[10] / vpinv[11] }, rel[16], vpr[16];
        /* world to clip less the camera's place: row 3 gains the camera's place through the rest */
        double vpw[16];
        if (matd_inverse(vpw, vpinv))
        {
            memcpy(vpr, vpw, sizeof vpr);
            for (int j = 0; j < 4; ++j)
                vpr[12 + j] = cam3[0] * vpw[j] + cam3[1] * vpw[4 + j] + cam3[2] * vpw[8 + j] + vpw[12 + j];
            if (matd_inverse(rel, vpr))
                for (int j = 0; j < 16; ++j)
                    g_rtx.ray_ivp[j] = (float)rel[j];
            for (int k = 0; k < 3; ++k)
                g_rtx.ray_cam[k] = (float)cam3[k];
        }
    }
    typedef struct
    {
        const Caster* c;
        uint32_t mi; /* the mesh, or UINT32_MAX: in the world, with the frame's moving structure */
        id<MTLRenderPipelineState> p;
        id<MTLBuffer> ids;
        NSUInteger ids_off;
        uint32_t m;
        float sm[16];
    } Job;
    Job* jobs = (Job*)malloc(((size_t)g_ncasters + g_nlate + 1) * sizeof(Job));
    uint32_t njobs = 0, dyn_verts = 0, tris = 0;
    g_rtx.waiting = 0;
    for (uint32_t i = 0; i < g_ncasters; ++i)
    {
        const Caster* c = &g_casters[i];
        float wvp[16], w[16], inv[16];
        double wvpd[16], wd[16], invd[16];
        int placed = 0;
        if (c->fixed && have_vpinv && rt_wvp(c, wvp))
        {
            for (int j = 0; j < 16; ++j)
                wvpd[j] = wvp[j];
            if (matd_inverse(invd, wvpd))
            {
                matd_mul(wd, wvpd, vpinv);
                /* one matrix from the object to the world (no projection left in it) */
                if (fabs(wd[15]) > 1e-9)
                {
                    double k = 1.0 / wd[15];
                    for (int j = 0; j < 16; ++j)
                        w[j] = (float)(wd[j] * k), inv[j] = (float)invd[j];
                    placed = fabsf(w[3]) + fabsf(w[7]) + fabsf(w[11]) < 1e-3f;
                }
            }
            /* in the world: finite, and not far out (the sky's meshes, drawn around the camera at
             * its infinite distance, are neither) - left out of the tracing */
            int in = 1;
            for (int j = 0; placed && in && j < 16; ++j)
                in = isfinite(w[j]) && fabsf(w[j]) < (float)RT_WORLD;
            if (placed && !in)
            {
                g_rtx.pl_far++;
                continue;
            }
        }
        RtMesh* me = NULL;
        uint32_t mi = UINT32_MAX;
        if (placed)
        {
            mi = rt_mesh(c);
            {
                RtMesh* mm = &g_rtx.mesh[mi];
                double k = 1.0 / wd[15];
                for (int j = 0; j < 16; ++j)
                    mm->wd[j] = wd[j] * k; /* from here on: object to world */
            }
            if (g_rtx.mesh[mi].unstable)
            {
                /* placed for this frame alone */
                if (g_rtx.nnow == g_rtx.cap_now)
                    g_rtx.cap_now = g_rtx.cap_now ? g_rtx.cap_now * 2 : 256,
                    g_rtx.now = realloc(g_rtx.now, g_rtx.cap_now * sizeof *g_rtx.now);
                g_rtx.now[g_rtx.nnow].mesh = mi;
                memcpy(g_rtx.now[g_rtx.nnow++].w, w, 64);
            }
            else
            {
                rt_place(mi, w);
            }
            me = &g_rtx.mesh[mi];
            me->seen = g_serial;
            if (me->blas || me->none || me->tried == g_serial)
                continue;
            if (njobs >= RT_BUILDS || tris >= RT_BUILD_TRIS)
            {
                g_rtx.waiting++;
                continue;
            }
        }
        id<MTLRenderPipelineState> p = rt_pipeline(c);
        if (!p)
        {
            g_rtx.waiting += me != NULL;
            continue;
        }
        Job* j = &jobs[njobs];
        j->m = rt_ids(c, &j->ids, &j->ids_off);
        if (!j->m)
        {
            if (me)
                me->none = 1;
            continue;
        }
        j->c = c, j->mi = mi, j->p = p;
        if (me)
            me->tried = g_serial, memcpy(j->sm, inv, 64), tris += j->m / 3;
        else
            rt_matrix(c, clip_world, view_world, j->sm), dyn_verts += j->m;
        njobs++;
    }
    /* the world's draws out of its scene (characters, some frames): after the effects of the scene
     * traced before, through that frame's camera; into another target before this scene, through this
     * one's - in the world, with what moves */
    int late_now = g_late_serial == g_serial, late_before = g_rtx.prev && g_late_serial == g_rtx.prev;
    if (late_now || late_before)
        for (uint32_t i = 0; i < g_nlate; ++i)
        {
            const Caster* c = &g_late[i];
            id<MTLRenderPipelineState> p = rt_pipeline(c);
            if (!p)
                continue;
            Job* j = &jobs[njobs];
            j->m = rt_ids(c, &j->ids, &j->ids_off);
            if (!j->m)
                continue;
            j->c = c, j->mi = UINT32_MAX, j->p = p;
            rt_matrix(c, late_now ? clip_world : g_rtx.late_cw, late_now ? view_world : g_rtx.late_vw, j->sm);
            dyn_verts += j->m, g_rtx.late_used += j->m / 3;
            njobs++;
        }
    /* anchors for the frames after: meshes drawn once this frame that hold still, where they stand now
     * (in this frame's reference: the anchors' camera, or the scene's when no anchor was drawn) - the
     * ones drawn earliest first: the world's draws find the frame's camera by the first anchor drawn */
    while (g_rtx.nanchor < 8)
    {
        uint32_t best = UINT32_MAX;
        for (uint32_t i = 0; i < g_rtx.nmesh; ++i)
        {
            RtMesh* me = &g_rtx.mesh[i];
            if (me->drawn != g_serial || me->ndrawn != 1 || me->unstable || !me->blas ||
                (best != UINT32_MAX && me->order >= g_rtx.mesh[best].order))
                continue;
            uint32_t a = 0;
            for (; a < g_rtx.nanchor; ++a)
                if (g_rtx.anchor[a].mesh == i)
                    break;
            if (a == g_rtx.nanchor)
                best = i;
        }
        if (best == UINT32_MAX)
            break;
        RtMesh* me = &g_rtx.mesh[best];
        uint32_t a = g_rtx.nanchor++;
        g_rtx.anchor[a].mesh = best, g_rtx.anchor[a].seen = g_serial;
        memcpy(g_rtx.anchor[a].w, me->wd, sizeof me->wd);
        cache_key(&g_rtx.anchor[a].key, &me->c);
    }
    /* a full set drifts toward the meshes drawn first: the latest-drawn anchor gives way to an earlier
     * candidate, one a frame (the fewer draws before the first anchor, the fewer on a camera guessed) */
    if (anchored && g_rtx.nanchor == 8)
    {
        uint32_t late = UINT32_MAX, cand = UINT32_MAX;
        for (uint32_t a = 0; a < g_rtx.nanchor; ++a)
        {
            RtMesh* me = &g_rtx.mesh[g_rtx.anchor[a].mesh];
            if (me->drawn == g_serial && (late == UINT32_MAX || me->order > g_rtx.mesh[g_rtx.anchor[late].mesh].order))
                late = a;
        }
        for (uint32_t i = 0; late != UINT32_MAX && i < g_rtx.nmesh; ++i)
        {
            RtMesh* me = &g_rtx.mesh[i];
            if (me->drawn != g_serial || me->ndrawn != 1 || me->unstable || !me->blas ||
                me->order >= g_rtx.mesh[g_rtx.anchor[late].mesh].order ||
                (cand != UINT32_MAX && me->order >= g_rtx.mesh[cand].order))
                continue;
            uint32_t a = 0;
            for (; a < g_rtx.nanchor; ++a)
                if (g_rtx.anchor[a].mesh == i)
                    break;
            if (a == g_rtx.nanchor)
                cand = i;
        }
        if (cand != UINT32_MAX)
        {
            RtMesh* me = &g_rtx.mesh[cand];
            g_rtx.anchor[late].mesh = cand, g_rtx.anchor[late].seen = g_serial;
            memcpy(g_rtx.anchor[late].w, me->wd, sizeof me->wd);
            cache_key(&g_rtx.anchor[late].key, &me->c);
        }
    }
    /* the triangles out */
    id<MTLBuffer> dyn_buf = nil;
    NSUInteger dyn_off = 0;
    uint32_t nstatic = 0;
    if (njobs)
    {
        if (dyn_verts)
            ring((size_t)dyn_verts * 48, 16, &dyn_buf, &dyn_off);
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.renderTargetWidth = 1, rp.renderTargetHeight = 1, rp.defaultRasterSampleCount = 1;
        id<MTLRenderCommandEncoder> e = [cmd() renderCommandEncoderWithDescriptor:rp];
        NSUInteger at = dyn_off;
        for (uint32_t i = 0; i < njobs; ++i)
        {
            Job* j = &jobs[i];
            if (j->mi != UINT32_MAX)
            {
                RtMesh* me = &g_rtx.mesh[j->mi];
                /* shared: its box is read back once the GPU is done (rt_bounds) */
                me->rtv = [g_dev newBufferWithLength:(NSUInteger)j->m * 48 options:MTLResourceStorageModeShared];
                me->normals = j->c->lib.vs.el[GFX_R_NORMAL].used;
                /* alpha-tested: its first texture and the test's reference, for the shadow maps */
                [me->tex release], me->tex = nil, me->alpha = 0;
                if (alpha_tested(&j->c->lib.fs) && j->c->tex[0])
                {
                    const GfxU* cu = (const GfxU*)((const uint8_t*)[j->c->ub contents] + j->c->uoff);
                    me->tex = [j->c->tex[0] retain], me->samp = j->c->samp[0], me->alpha = 1;
                    me->aref = fminf(fmaxf(cu->params[1] / 255.0f, 0.02f), 0.98f);
                }
                me->captured = g_serial, me->bounds = 0;
                rt_capture(e, j->p, j->c, j->sm, j->ids, j->ids_off, j->m, me->rtv, 0);
                jobs[nstatic++] = *j; /* the meshes first, in order (the moving ones are all in dyn_buf) */
            }
            else
            {
                rt_capture(e, j->p, j->c, j->sm, j->ids, j->ids_off, j->m, dyn_buf, at);
                at += (NSUInteger)j->m * 48;
            }
        }
        [e endEncoding];
    }
    /* their structures: one for each mesh captured, one for what moves */
    uint32_t nb = nstatic + (dyn_verts ? 1 : 0);
    rt_release_later(g_rtx.dyn), g_rtx.dyn = nil;
    if (nb)
    {
        MTLPrimitiveAccelerationStructureDescriptor** desc = malloc(nb * sizeof *desc);
        id<MTLAccelerationStructure>* as = malloc(nb * sizeof *as);
        NSUInteger* soff = malloc(nb * sizeof *soff), scratch = 0;
        for (uint32_t i = 0; i < nb; ++i)
        {
            desc[i] = i < nstatic ? rt_blas_desc(g_rtx.mesh[jobs[i].mi].rtv, 0, jobs[i].m / 3) : rt_blas_desc(dyn_buf, dyn_off, dyn_verts / 3);
            MTLAccelerationStructureSizes sz = [g_dev accelerationStructureSizesWithDescriptor:desc[i]];
            as[i] = [g_dev newAccelerationStructureWithSize:sz.accelerationStructureSize];
            soff[i] = scratch, scratch += (sz.buildScratchBufferSize + 255) & ~(NSUInteger)255;
        }
        id<MTLBuffer> sb = [g_dev newBufferWithLength:scratch ? scratch : 256 options:MTLResourceStorageModePrivate];
        id<MTLAccelerationStructureCommandEncoder> e = [cmd() accelerationStructureCommandEncoder];
        for (uint32_t i = 0; i < nb; ++i)
            if (as[i])
                [e buildAccelerationStructure:as[i] descriptor:desc[i] scratchBuffer:sb scratchBufferOffset:soff[i]];
        [e endEncoding];
        [sb release]; /* the command buffer holds it until it is done */
        for (uint32_t i = 0; i < nstatic; ++i)
        {
            RtMesh* me = &g_rtx.mesh[jobs[i].mi];
            me->blas = as[i], me->tris = jobs[i].m / 3;
            if (!as[i]) /* no room: captured again when drawn again */
                rt_release_later(me->rtv), me->rtv = nil;
        }
        if (dyn_verts)
            g_rtx.dyn = as[nstatic];
        free(desc), free(as), free(soff);
    }
    g_rtx.built = nstatic;
    g_rtx.prev = g_serial;
    /* this frame's camera, for what the world draws after its effects (traced with the next scene) */
    memcpy(g_rtx.late_cw, clip_world, 64), memcpy(g_rtx.late_vw, view_world, 64);
    late_clear(); /* encoded: the command buffer holds what it reads */
    free(jobs);
    /* a place kept that the camera looks at (its mesh's box well inside the view) and the game has
     * not drawn for three frames traced is not there any more: a nearer or farther detail of the mesh
     * is drawn in its stead, or it is out of the draw distance */
    for (uint32_t i = 0; i < g_rtx.nplace; ++i)
    {
        RtPlace* p = &g_rtx.place[i];
        if (p->gone || p->seen == g_serial)
            continue;
        RtMesh* me = &g_rtx.mesh[p->mesh];
        int in;
        if (rt_bounds(me))
            in = rt_in_view(me, p->w, vp);
        else
        {
            /* no box yet: its origin */
            float c[4];
            for (int j = 0; j < 4; ++j)
                c[j] = p->w[12] * vp[j] + p->w[13] * vp[4 + j] + p->w[14] * vp[8 + j] + vp[12 + j];
            in = c[3] > 1.0f && fabsf(c[0]) < 0.8f * c[3] && fabsf(c[1]) < 0.8f * c[3];
        }
        if (in)
        {
            if (++p->missed >= 3)
                p->gone = 1, g_rtx.pl_gone++;
        }
        else
            p->missed = 0;
    }
    /* the sun's shadow maps from all of it (the world's draws of the next frame look themselves up) */
    rt_sun_maps(vpinv, dyn_buf, dyn_off, dyn_verts);
    /* the instances: every place kept whose mesh has a structure, and what moves */
    NSMutableArray* list = [NSMutableArray arrayWithCapacity:g_rtx.nmesh + 1];
    g_rtx.nkept = g_rtx.kept_tris = 0;
    for (uint32_t i = 0; i < g_rtx.nmesh; ++i)
    {
        RtMesh* me = &g_rtx.mesh[i];
        me->at = UINT32_MAX;
        if (!me->dead && me->blas && (me->nplace || me->unstable))
            me->at = (uint32_t)list.count, [list addObject:me->blas], g_rtx.nkept++;
    }
    uint32_t ninst = 0;
    for (uint32_t i = 0; i < g_rtx.nplace; ++i)
        ninst += !g_rtx.place[i].gone && g_rtx.place[i].seen + RT_KEEP >= g_serial && g_rtx.mesh[g_rtx.place[i].mesh].at != UINT32_MAX;
    for (uint32_t i = 0; i < g_rtx.nnow; ++i)
        ninst += g_rtx.mesh[g_rtx.now[i].mesh].at != UINT32_MAX;
    g_rtx.moving_tris = dyn_verts / 3;
    uint32_t dyn_at = (uint32_t)list.count;
    if (g_rtx.dyn)
        [list addObject:g_rtx.dyn], ninst++;
    g_rtx.ninst = ninst;
    rt_release_later(g_rtx.tlas), g_rtx.tlas = nil;
    [g_rtx.blas release], g_rtx.blas = nil;
    if (!ninst)
        return 0;
    id<MTLBuffer> ib;
    NSUInteger ioff;
    MTLAccelerationStructureInstanceDescriptor* inst =
        (MTLAccelerationStructureInstanceDescriptor*)ring(ninst * sizeof *inst, 16, &ib, &ioff);
    uint64_t* vtx = (uint64_t*)ring(ninst * 8, 16, &g_rtx.vtx, &g_rtx.vtx_off);
    [g_rtx.res release];
    g_rtx.res = [[NSMutableArray alloc] initWithCapacity:g_rtx.nkept + 1];
    for (uint32_t i = 0; i < g_rtx.nmesh; ++i)
        if (g_rtx.mesh[i].at != UINT32_MAX)
            [g_rtx.res addObject:g_rtx.mesh[i].rtv];
    uint32_t n = 0;
    for (uint32_t i = 0; i < g_rtx.nplace + g_rtx.nnow; ++i)
    {
        int kept = i < g_rtx.nplace;
        const float* pw = kept ? g_rtx.place[i].w : g_rtx.now[i - g_rtx.nplace].w;
        RtMesh* me = &g_rtx.mesh[kept ? g_rtx.place[i].mesh : g_rtx.now[i - g_rtx.nplace].mesh];
        if ((kept && (g_rtx.place[i].gone || g_rtx.place[i].seen + RT_KEEP < g_serial)) || me->at == UINT32_MAX)
            continue;
        memset(&inst[n], 0, sizeof inst[n]);
        /* column j of the (column-vector) transform is row j of the row-vector one */
        for (int j = 0; j < 4; ++j)
        {
            inst[n].transformationMatrix.columns[j].x = pw[4 * j];
            inst[n].transformationMatrix.columns[j].y = pw[4 * j + 1];
            inst[n].transformationMatrix.columns[j].z = pw[4 * j + 2];
        }
        inst[n].options = MTLAccelerationStructureInstanceOptionOpaque;
        /* 1 the zone's (its shaders' draws); 2 the fixed function's (characters, and what they carry:
         * they shade what stands around them, not themselves); 4 what is alpha-tested (leaves: traced
         * solid, they would shade as squares) */
        inst[n].mask = !me->c.lib.vs.prog ? 2 : alpha_tested(&me->c.lib.fs) ? 4 : 1;
        inst[n].accelerationStructureIndex = me->at;
        vtx[n] = me->rtv.gpuAddress | (uint64_t)(me->normals != 0); /* the low bit: with normals */
        g_rtx.kept_tris += me->tris;
        n++;
    }
    if (g_rtx.dyn)
    {
        memset(&inst[n], 0, sizeof inst[n]);
        inst[n].transformationMatrix.columns[0].x = 1.0f;
        inst[n].transformationMatrix.columns[1].y = 1.0f;
        inst[n].transformationMatrix.columns[2].z = 1.0f;
        inst[n].options = MTLAccelerationStructureInstanceOptionOpaque;
        inst[n].mask = 2; /* what moves */
        inst[n].accelerationStructureIndex = dyn_at;
        vtx[n] = dyn_buf.gpuAddress + dyn_off;
        [g_rtx.res addObject:dyn_buf];
        n++;
    }
    MTLInstanceAccelerationStructureDescriptor* td = [MTLInstanceAccelerationStructureDescriptor descriptor];
    td.instancedAccelerationStructures = list;
    td.instanceCount = n;
    td.instanceDescriptorBuffer = ib;
    td.instanceDescriptorBufferOffset = ioff;
    MTLAccelerationStructureSizes sz = [g_dev accelerationStructureSizesWithDescriptor:td];
    g_rtx.tlas = [g_dev newAccelerationStructureWithSize:sz.accelerationStructureSize];
    id<MTLBuffer> sb = [g_dev newBufferWithLength:sz.buildScratchBufferSize ? sz.buildScratchBufferSize : 256
                                          options:MTLResourceStorageModePrivate];
    id<MTLAccelerationStructureCommandEncoder> e = [cmd() accelerationStructureCommandEncoder];
    [e buildAccelerationStructure:g_rtx.tlas descriptor:td scratchBuffer:sb scratchBufferOffset:0];
    [e endEncoding];
    [sb release];
    g_rtx.blas = [list retain];
    return g_rtx.tlas != nil;
}

/* --- the world's draws shaded by the sun's shadow maps, as they are drawn ---------------------------------
 * The world's opaque draws (GfxDraw.caster) get a traced pipeline (the rt keys, gfx_msl.c): the vertex
 * function passes on each vertex's place in the world, less the camera's; the fragment function looks
 * that place up in the sun's shadow maps of the last scene (rt_sun_maps) and takes a share of its colour
 * where something nearer the sun is there - before the game's fog, so the fog lies over the shadow as it
 * would over the lit ground. What the draws need to find their place in the world is the camera they
 * were drawn with: a fixed-function draw's own view and projection; for the zone's shaders (their
 * transforms carry no camera of their own), the frame's camera - the last frame's, moved on as it moved,
 * until the first anchor is drawn, then that anchor's (its transform through the inverse of where it
 * stands). */
typedef struct RtDraw
{
    float m[16];  /* clip space to the world less the camera's place */
    float cam[4]; /* the camera's place */
    float sun[4]; /* toward the sun; w = 1 when there is one */
    float k[4];   /* the shadow's strength, the debug view, how far out, - */
    float s[RT_CASCADES][16]; /* the world less the camera's place to each cascade's map */
    float cs[RT_CASCADES][4]; /* each one's normal offset (world units), filter step (uv), depth bias, reach */
} RtDraw;

static int rt_inpass(void)
{
    return g_rtx.ok && g_fxs.fx != 0.0f && g_fxs.rt > 0.0f && g_fxs.rt_shadow > 0.0f;
}

static int rt_draw_wanted(const GfxDraw* d)
{
    return (d->caster == 1 || d->caster == 2 || d->receive) && !d->vs.rhw && !g_rt_face && !g_rt_level && g_sm.ok && rt_inpass();
}

/* clip space back to the world less the camera's place (m) and that place (cam), from world to clip */
static int rt_ray_camera(const double* vp, float* m, float* cam)
{
    double inv[16], vpr[16], rel[16];
    if (!matd_inverse(inv, vp) || fabs(inv[11]) < 1e-12)
        return 0;
    double c[3] = { inv[8] / inv[11], inv[9] / inv[11], inv[10] / inv[11] };
    memcpy(vpr, vp, sizeof vpr);
    for (int j = 0; j < 4; ++j)
        vpr[12 + j] = c[0] * vp[j] + c[1] * vp[4 + j] + c[2] * vp[8 + j] + vp[12 + j];
    if (!matd_inverse(rel, vpr))
        return 0;
    for (int j = 0; j < 16; ++j)
        m[j] = (float)rel[j];
    for (int k = 0; k < 3; ++k)
        cam[k] = (float)c[k];
    return 1;
}

/* the mesh a draw is (its CacheKey), as its recorded caster's would be; 0 when it is not one the game
 * keeps */
static int rt_draw_key(const GfxDraw* d, CacheKey* k)
{
    if (d->prim == GFX_TRIANGLEFAN || (d->indices && !d->ibuf))
        return 0;
    memset(k, 0, sizeof *k);
    for (int s = 0; s < GFX_NSTREAMS; ++s)
    {
        if (d->buf[s])
            k->vb[s] = (void*)d->buf[s]->b, k->voff[s] = d->buf_off[s];
        else if (d->data[s] && d->size[s])
            return 0;
        else
            k->vb[s] = (void*)g_dummy;
    }
    if (d->ibuf)
        k->ib = (void*)d->ibuf->b, k->ioff = d->ibuf_off;
    k->n = vertex_count(d->prim, d->count), k->vstart = d->vertex_start;
    k->lib.vs = d->vs, k->lib.fs = d->fs;
    return 1;
}

/* FFXI_RTCAM=<file>: each change of the draws' camera, for tracing */
static void rt_cam_log(const char* what)
{
    static FILE* f;
    static int tried, lines;
    if (!tried)
        tried = 1, f = getenv("FFXI_RTCAM") ? fopen(getenv("FFXI_RTCAM"), "w") : NULL;
    if (!f || ++lines > 200000)
        return;
    fprintf(f, "%llu t%p %s cam %.3f %.3f %.3f ok %d exact %d draw %u\n", (unsigned long long)g_serial, (void*)g_rt, what,
        g_rtx.fvp_cam[0], g_rtx.fvp_cam[1], g_rtx.fvp_cam[2], g_rtx.fvp_ok, g_rtx.fvp_exact, g_rtx.cam_draws);
}

/* the frame's camera for a zone's draw, as far as the draws so far give it (rd.m, rd.cam) */
static int rt_frame_camera(const GfxDraw* d, RtDraw* rd)
{
    /* each target drawn this frame has a camera of its own: the one left kept, the one come to
     * restored (the game draws the world into more than one - a reflection, the characters) */
    if (g_rtx.fvp_serial == g_serial && g_rtx.fvp_target != g_rt)
    {
        int put = 0, got = -1;
        for (int i = 0; i < 4; ++i)
        {
            if (g_rtx.fvp_slot[i].serial != g_serial || g_rtx.fvp_slot[i].t == g_rtx.fvp_target)
                put = i;
            if (g_rtx.fvp_slot[i].serial == g_serial && g_rtx.fvp_slot[i].t == g_rt)
                got = i;
        }
        if (got == put)
            got = -1;
        __typeof__(g_rtx.fvp_slot[0])* o = &g_rtx.fvp_slot[put];
        o->t = g_rtx.fvp_target, o->serial = g_serial, o->ok = g_rtx.fvp_ok, o->exact = g_rtx.fvp_exact;
        o->have_pend = g_rtx.have_pend;
        memcpy(o->m, g_rtx.fvp_m, 64), memcpy(o->cam, g_rtx.fvp_cam, 16);
        memcpy(o->pm, g_rtx.pend_m, 64), memcpy(o->pcam, g_rtx.pend_cam, 16);
        g_rtx.fvp_target = g_rt;
        if (got >= 0)
        {
            const __typeof__(g_rtx.fvp_slot[0])* q = &g_rtx.fvp_slot[got];
            g_rtx.fvp_ok = q->ok, g_rtx.fvp_exact = q->exact, g_rtx.have_pend = q->have_pend;
            memcpy(g_rtx.fvp_m, q->m, 64), memcpy(g_rtx.fvp_cam, q->cam, 16);
            memcpy(g_rtx.pend_m, q->pm, 64), memcpy(g_rtx.pend_cam, q->pcam, 16);
            g_rtx.fvp_slot[got].serial = 0;
            rt_cam_log("restore");
        }
        else
            g_rtx.fvp_serial = 0; /* new to this frame: from the guess */
    }
    if (g_rtx.fvp_serial != g_serial)
    {
        g_rtx.fvp_serial = g_serial, g_rtx.fvp_exact = 0, g_rtx.have_pend = 0, g_rtx.fvp_target = g_rt;
        /* the last camera, moved on by as much as it moved the frame before: last * (before^-1 * last)
         * in D3D's order (world to clip: the step is a move of the world, then the camera) */
        double guess[16], ib[16], step[16];
        memcpy(guess, g_rtx.vp_last, sizeof guess);
        if (g_rtx.have_vp_prev && g_rtx.vp_last_serial + 1 == g_serial && matd_inverse(ib, g_rtx.vp_prev))
            matd_mul(step, g_rtx.vp_last, ib), matd_mul(guess, step, g_rtx.vp_last);
        g_rtx.fvp_ok = g_rtx.have_vp_last && rt_ray_camera(guess, g_rtx.fvp_m, g_rtx.fvp_cam);
        g_rtx.cam_draws = 0;
        rt_cam_log("guess");
    }
    g_rtx.cam_draws++;
    CacheKey k;
    float wvp[16];
    int keyed = rt_draw_key(d, &k);
    /* A mesh the world keeps at just one place is an anchor too, from each of its draws: the camera
     * is the draw's transform less the place's (most of the zone's meshes are drawn at one place, the
     * first draws of a frame among them - before any of the anchors). Taken when it is within half a unit
     * of the guess;
     * one elsewhere is held until a second agrees - a cut, or a second view of the world drawn in the
     * same frame (the title's flythrough cross-fades two), which then takes over. */
    if (keyed)
    {
        if (g_rtx.cap_solo < g_rtx.nmesh || g_rtx.solo_serial != g_serial)
        {
            if (g_rtx.cap_solo < g_rtx.nmesh)
                g_rtx.cap_solo = g_rtx.nmesh + 1024, g_rtx.solo = (int32_t*)realloc(g_rtx.solo, g_rtx.cap_solo * sizeof *g_rtx.solo);
            for (uint32_t i = 0; i < g_rtx.nmesh; ++i)
                g_rtx.solo[i] = -1;
            for (uint32_t i = 0; i < g_rtx.nplace; ++i)
            {
                const RtPlace* p = &g_rtx.place[i];
                if (p->gone || p->seen + 60 < g_serial || g_rtx.mesh[p->mesh].unstable)
                    continue;
                int32_t* o = &g_rtx.solo[p->mesh];
                *o = *o == -1 ? (int32_t)i : -2;
            }
            g_rtx.solo_serial = g_serial;
        }
        uintptr_t at = (uintptr_t)map_get(&g_rtx.map, &k, sizeof k);
        RtMesh* me = at && at - 1 < g_rtx.nmesh ? &g_rtx.mesh[at - 1] : NULL;
        int32_t pi = me ? g_rtx.solo[at - 1] : -1;
        uint32_t nth = 0;
        if (me)
        {
            if (me->dt_serial != g_serial)
                me->dt_serial = g_serial, me->dt_n = 0;
            nth = me->dt_n++;
        }
        double iw[16], wd[16], vp[16], w[16];
        float m[16], cam[4];
        /* the camera from the mesh's same draw last frame: VP = VP_last clip_last^-1 clip (its transform
         * to the world cancels) - any mesh drawn as often then; else from its one place */
        int chain = me && g_rtx.have_vp_last && me->clip_serial == g_rtx.vp_last_serial && me->clip_serial + 1 == g_serial &&
            nth < me->nclip && nth < 16;
        if ((chain || pi >= 0) && rt_wvp_u(&d->u, &d->vs, d->vs_tokens, wvp))
        {
            int got;
            for (int j = 0; j < 16; ++j)
                wd[j] = wvp[j];
            if (chain)
            {
                double a[16], ia[16], step[16];
                for (int j = 0; j < 16; ++j)
                    a[j] = me->clip[nth][j];
                got = matd_inverse(ia, a) && (matd_mul(step, ia, wd), matd_mul(vp, g_rtx.vp_last, step), 1);
            }
            else
            {
                for (int j = 0; j < 16; ++j)
                    w[j] = g_rtx.place[pi].w[j];
                got = matd_inverse(iw, w) && (matd_mul(vp, iw, wd), 1);
            }
            if (got && rt_ray_camera(vp, m, cam))
            {
                float dg = 0.0f, dp = 0.0f;
                for (int j = 0; j < 3; ++j)
                    dg += (cam[j] - g_rtx.fvp_cam[j]) * (cam[j] - g_rtx.fvp_cam[j]),
                    dp += (cam[j] - g_rtx.pend_cam[j]) * (cam[j] - g_rtx.pend_cam[j]);
                int take;
                if (g_rtx.fvp_exact)
                    take = dg < 0.25f ? 2 : g_rtx.have_pend && dp < 0.01f; /* 2: the same camera, as it is */
                else
                    take = (g_rtx.fvp_ok && dg < 0.25f) || (g_rtx.have_pend && dp < 0.01f);
                if (take == 1)
                {
                    const char* why = g_rtx.fvp_exact ? "solo-switch" : "solo";
                    memcpy(g_rtx.fvp_m, m, 64), memcpy(g_rtx.fvp_cam, cam, 16);
                    memcpy(g_rtx.fvp_vp, vp, sizeof vp), g_rtx.fvp_vp_serial = g_serial;
                    g_rtx.fvp_ok = g_rtx.fvp_exact = 1, g_rtx.have_pend = 0, g_rtx.dr_solo++;
                    rt_cam_log(why);
                }
                else if (!take)
                {
                    memcpy(g_rtx.pend_m, m, 64), memcpy(g_rtx.pend_cam, cam, 16), memcpy(g_rtx.pend_vp, vp, sizeof vp);
                    g_rtx.have_pend = 1, g_rtx.dr_solo_far++;
                    rt_cam_log("solo-held");
                }
                keyed = !g_rtx.fvp_exact;
            }
        }
    }
    keyed = keyed && !g_rtx.fvp_exact;
    if (keyed && g_rtx.nanchor)
        for (uint32_t a = 0; a < g_rtx.nanchor; ++a)
        {
            if (memcmp(&k, &g_rtx.anchor[a].key, sizeof k))
                continue;
            double iw[16], wd[16], vp[16];
            if (rt_wvp_u(&d->u, &d->vs, d->vs_tokens, wvp) && matd_inverse(iw, g_rtx.anchor[a].w))
            {
                for (int j = 0; j < 16; ++j)
                    wd[j] = wvp[j];
                matd_mul(vp, iw, wd);
                if (rt_ray_camera(vp, g_rtx.fvp_m, g_rtx.fvp_cam))
                    g_rtx.fvp_ok = 1, g_rtx.fvp_exact = 1, memcpy(g_rtx.fvp_vp, vp, sizeof vp),
                    g_rtx.fvp_vp_serial = g_serial, rt_cam_log("anchor");
            }
            break;
        }
    if (!g_rtx.fvp_ok)
        return 0;
    memcpy(rd->m, g_rtx.fvp_m, 64);
    memcpy(rd->cam, g_rtx.fvp_cam, 12);
    return 1;
}

/* a traced draw's camera, the sun and the maps, into the encoder */
static void rt_draw_bind(const GfxDraw* d, id<MTLRenderCommandEncoder> e)
{
    RtDraw rd;
    memset(&rd, 0, sizeof rd);
    int ok;
    if (!d->vs.prog)
    {
        double vp[16];
        for (int j = 0; j < 16; ++j)
            vp[j] = d->view_proj[j];
        ok = rt_ray_camera(vp, rd.m, rd.cam);
        g_rtx.dr_ffp += ok;
    }
    else
    {
        ok = rt_frame_camera(d, &rd);
        if (ok)
            g_rtx.fvp_exact ? g_rtx.dr_exact++ : g_rtx.dr_pred++;
    }
    if (ok)
    {
        memcpy(rd.sun, g_rtx.sun_now, 16);
        rd.k[0] = g_rtx.k_now[0], rd.k[1] = g_fxs.rt_debug, rd.k[2] = g_rtx.k_now[2];
        /* the maps are relative to their scene's camera, the draw's place to its own: moved by the
         * difference (else every shadow slides by the camera's step between them) */
        float dc[3];
        for (int i = 0; i < 3; ++i)
            dc[i] = (float)((double)rd.cam[i] - g_sm.cam[i]);
        for (int c = 0; c < RT_CASCADES; ++c)
        {
            const SmCascade* k = &g_sm.c[c < g_sm.nc ? c : g_sm.nc - 1];
            float* m = rd.s[c];
            memcpy(m, k->m, 64);
            for (int j = 0; j < 3; ++j)
                m[12 + j] += dc[0] * m[j] + dc[1] * m[4 + j] + dc[2] * m[8 + j];
            /* two texels off the face along its normal, a texel of depth, the filter's step a texel
             * and a half */
            rd.cs[c][0] = (float)(2.0 * k->texel), rd.cs[c][1] = 1.5f / RT_MAP;
            rd.cs[c][2] = (float)(k->texel / k->range), rd.cs[c][3] = (float)k->reach;
        }
    }
    else
    {
        g_rtx.dr_none++;
        rd.m[0] = rd.m[5] = rd.m[10] = rd.m[15] = 1.0f; /* no camera: no shadow (sun.w 0) */
    }
    [e setVertexBytes:&rd length:sizeof rd atIndex:5];
    [e setFragmentBytes:&rd length:sizeof rd atIndex:5];
    [e setFragmentTexture:g_sm.map atIndex:8];
}

/* the traced scene against the drawn one (debug = 6), at w x h */
static id<MTLTexture> rt_view(const FxU* u, id<MTLTexture> depth, NSUInteger w, NSUInteger h)
{
    if (!g_rtx.out || g_rtx.out.width != w || g_rtx.out.height != h)
    {
        [g_rtx.out release];
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                     width:w height:h mipmapped:NO];
        d.storageMode = MTLStorageModePrivate;
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        g_rtx.out = [g_dev newTextureWithDescriptor:d];
    }
    NSUInteger n = g_rtx.blas.count;
    id<MTLResource>* r = (id<MTLResource>*)malloc((n ? n : 1) * sizeof *r);
    [g_rtx.blas getObjects:r range:NSMakeRange(0, n)];
    id<MTLComputeCommandEncoder> e = [cmd() computeCommandEncoder];
    [e setComputePipelineState:g_rtx.view_pipe];
    [e setBytes:u length:sizeof *u atIndex:0];
    [e setAccelerationStructure:g_rtx.tlas atBufferIndex:1];
    [e setBuffer:g_rtx.vtx offset:g_rtx.vtx_off atIndex:2];
    [e useResources:r count:n usage:MTLResourceUsageRead];
    NSUInteger nv = g_rtx.res.count;
    id<MTLResource>* rv = (id<MTLResource>*)malloc((nv ? nv : 1) * sizeof *rv);
    [g_rtx.res getObjects:rv range:NSMakeRange(0, nv)];
    [e useResources:rv count:nv usage:MTLResourceUsageRead];
    free(rv);
    [e setTexture:depth atIndex:0];
    [e setTexture:g_rtx.out atIndex:1];
    [e dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [e endEncoding];
    free(r);
    return g_rtx.out;
}

void gfx_trace_dump(const char* path)
{
    FILE* f = fopen(path, "w");
    if (!f)
        return;
    fprintf(f, "frame own-sun strength day live cached no-pipeline map-across sun(x y z) camera(x y z) depth(1 as drawn, 2 another since) sun-light ambient fog(r g b)\n");
    uint32_t n = g_fx.ntrace < 1200 ? g_fx.ntrace : 1200;
    for (uint32_t k = g_fx.ntrace - n; k < g_fx.ntrace; ++k)
    {
        __typeof__(g_fx.trace[0])* t = &g_fx.trace[k % 1200];
        fprintf(f, "%llu %u %.2f %.2f %u %u %u %.0f %.3f %.3f %.3f %.1f %.1f %.1f %u %.3f %.3f %.3f %.3f %.3f\n", (unsigned long long)t->serial, t->own,
            t->strength, t->day, t->live, t->cached, t->skipped, t->across, t->sun[0], t->sun[1], t->sun[2], t->cam[0], t->cam[1],
            t->cam[2], t->depth, t->dl, t->al, t->fog[0], t->fog[1], t->fog[2]);
    }
    fclose(f);
}

void gfx_scene_done(GfxTex* color, const GfxScene* s)
{
    if (!g_dev || g_fxs.fx == 0.0f || !color || color->type != GFX_TEX_2D)
        return;
    @autoreleasepool
    {
        flush_pass();
        id<MTLTexture> ct = color->tex, depth = color->depth_world ? color->depth_world : color->depth_seen;
        g_fx.tr_depth = color->depth_world && color->depth_world != color->depth_seen ? 2 : depth ? 1 : 0;
        uint64_t t0 = gfx_profiling ? gfx_now_ns() : 0;
        if (depth && depth.width == ct.width && depth.height == ct.height && s->proj[11] != 0.0f && fx_init())
        {
            /* the scene's viewport, within the target */
            float vx = (float)s->vp[0], vy = (float)s->vp[1], vw = (float)s->vp[2], vh = (float)s->vp[3];
            if (vw < 16 || vh < 16 || vx + vw > ct.width || vy + vh > ct.height)
                vx = vy = 0, vw = (float)ct.width, vh = (float)ct.height;
            float minz, maxz;
            memcpy(&minz, &s->vp[4], 4);
            memcpy(&maxz, &s->vp[5], 4);
            if (maxz <= minz)
                minz = 0, maxz = 1;
            /* the occlusion at about 2000 pixels across (half a 4096 background, all of 1920): fine
             * enough that its edges hold still; bloom and rays, soft anyway, at about 1000 */
            uint32_t div = vw > 2048 ? 2 : 1, bdiv = vw > 2048 ? 4 : vw > 1024 ? 2 : 1;
            NSUInteger aw = (NSUInteger)((vw + div - 1) / div), ah = (NSUInteger)((vh + div - 1) / div);
            NSUInteger bw = (NSUInteger)((vw + bdiv - 1) / bdiv), bh = (NSUInteger)((vh + bdiv - 1) / bdiv);
            float hand = s->proj[11] < 0.0f ? -1.0f : 1.0f;
            FxU u = {
                { s->proj[0], s->proj[5], s->proj[8], s->proj[9] },
                { s->proj[10], s->proj[14], minz, maxz },
                { vx, vy, vw, vh },
                { (float)ct.width, (float)ct.height, (float)aw, (float)ah },
                { g_fxs.radius, g_fxs.ao, 0.15f, vh * 0.1f },
                { g_fxs.grade, g_fxs.sat, g_fxs.contrast, g_fxs.debug },
                { hand, 0, 0, 0 },
            };
            /* world up in view space: the world's y axis through the inverse view matrix, pointing
             * the way the camera's own up does (FFXI's world y points down) */
            g_fx.eased = g_fx.eased_serial && g_fx.eased_serial + 1 == g_serial;
            g_fx.eased_serial = g_serial;
            float inv[16], up[3];
            if (mat_inverse(inv, s->view))
            {
                float sign = inv[5] < 0.0f ? -1.0f : 1.0f;
                up[0] = inv[1] * sign, up[1] = inv[5] * sign, up[2] = inv[9] * sign;
                normalize3(up);
                /* fog where the game fogs its world (its fog color is the zone's), fading in and out */
                float on = s->fog[2] != 0.0f ? 1.0f : 0.0f;
                fx_ease(&g_fx.fog_on, &on, 1, 0.1f);
                /* the fog's colour eased over about a second, whether or not the last scene was the frame
                 * before: FFXI sets it per object, and the first fogged object differs from frame to
                 * frame near its lights; a new zone or a camera jump (the cache's) takes it at once */
                if (g_fxs.fog <= 0.0f)
                    g_fx.fogc_set = 0; /* no fog: the next one takes its colour at once */
                else if (!g_fx.fogc_set)
                    memcpy(g_fx.fogc, s->fogcolor, 12), g_fx.fogc_set = 1;
                else
                    for (int j = 0; j < 3; ++j)
                        g_fx.fogc[j] += (s->fogcolor[j] - g_fx.fogc[j]) * 0.04f;
                memcpy(g_fx.tr_fog, s->fogcolor, 12);
                fx_ease(g_fx.up, up, 3, 0.2f);
                normalize3(g_fx.up);
                memcpy(u.up, g_fx.up, 12);
                memcpy(u.fogc, g_fx.fogc, 12);
                u.fogc[3] = g_fxs.fog * g_fx.fog_on * expf(-g_fxs.fog_falloff * g_fxs.fog_height);
                u.fogp[0] = g_fxs.fog_falloff, u.fogp[1] = g_fxs.fog_max, u.fogp[2] = g_fxs.fog_sun;
                u.fogp[3] = fminf(fmaxf(g_fxs.fog_g, 0.0f), 0.95f);
            }
            /* the sun, kept in the world: the zone's shader draws carry no light, so a frame gives one
             * only when a lit draw (a character) is in view - the shadows hold the last one rather
             * than come and go with what the camera sees */
            float vinv[16];
            int have_v = mat_inverse(vinv, s->view), own = s->sun_dir[3] != 0.0f && have_v;
            if (own)
            {
                float w[3];
                for (int j = 0; j < 3; ++j)
                    w[j] = s->sun_dir[0] * vinv[j] + s->sun_dir[1] * vinv[4 + j] + s->sun_dir[2] * vinv[8 + j];
                normalize3(w);
                /* the sun moves on in steps of a quarter degree: the shadows' edges hold still
                 * between them, rather than crawl a little every frame */
                float dot = w[0] * g_fx.sunw[0] + w[1] * g_fx.sunw[1] + w[2] * g_fx.sunw[2];
                if (!g_fx.sunw_seen || dot < 0.99999f)
                    memcpy(g_fx.sunw, w, 12);
                g_fx.sunw_seen = g_serial;
                fx_ease(g_fx.suncol, s->sun_color, 3, 0.1f);
            }
            int shadows = g_fxs.sun > 0.0f || g_fxs.shadow > 0.0f || rt_inpass();
            g_rtx.day = 0.0f;
            if (have_v && (own || (shadows && g_fx.sunw_seen && g_fx.sunw_seen + 600 > g_serial)))
            {
                /* back into this frame's view */
                for (int j = 0; j < 3; ++j)
                    g_fx.sun[j] = g_fx.sunw[0] * s->view[j] + g_fx.sunw[1] * s->view[4 + j] + g_fx.sunw[2] * s->view[8 + j];
                normalize3(g_fx.sun);
                memcpy(u.sun, g_fx.sun, 12);
                u.sun[3] = 1.0f;
                memcpy(u.suncol, g_fx.suncol, 12);
                /* shadows while the sun (or moon) is up: fading as it nears the horizon */
                float e = g_fx.sun[0] * g_fx.up[0] + g_fx.sun[1] * g_fx.up[1] + g_fx.sun[2] * g_fx.up[2];
                float day = fminf(fmaxf((e - 0.05f) / 0.15f, 0.0f), 1.0f);
                if (own)
                {
                    float dl = 0.2126f * s->sun_color[0] + 0.7152f * s->sun_color[1] + 0.0722f * s->sun_color[2];
                    float al = 0.2126f * s->ambient[0] + 0.7152f * s->ambient[1] + 0.0722f * s->ambient[2];
                    float r = dl / fmaxf(dl + al, 1e-3f);
                    float want = fminf(fmaxf((r - g_fxs.sun_direct) / 0.3f, 0.0f), 1.0f);
                    g_fx.direct = g_fx.eased && g_fx.direct >= 0.0f ? g_fx.direct + (want - g_fx.direct) * 0.05f : want;
                    g_fx.tr_dl = dl, g_fx.tr_al = al;
                }
                day *= g_fxs.sun_direct > 0.0f ? g_fx.direct : 1.0f;
                g_rtx.day = day;
                /* the world's draws traced (rt_draw_bind) shade themselves: not again from the sun's
                 * maps and the contact rays */
                u.shadow[0] = rt_inpass() ? 0.0f : g_fxs.shadow * day;
                u.shadow[1] = g_fxs.shadow_length, u.shadow[2] = 0.3f, u.shadow[3] = 40.0f;
                if (g_fxs.sun > 0.0f && day > 0.0f && !rt_inpass() && sun_map(s, g_fx.sunw, &u))
                    u.smap[0] = g_fxs.sun * day, g_fx.st_drawn_this = 1;
                g_fx.tr_strength = u.smap[0], g_fx.tr_day = day;
                u.smap2[1] = fminf(fmaxf(g_fxs.sun_face, 0.0f), 1.0f), u.smap2[2] = fmaxf(g_fxs.sun_min, 0.0f);
                /* the sun's place on screen: far along its direction, through the projection */
                const float* sd = g_fx.sun;
                float cw = sd[2] * s->proj[11];
                if (cw > 0.05f)
                {
                    float nx = (sd[0] * s->proj[0] + sd[2] * s->proj[8]) / cw;
                    float ny = (sd[1] * s->proj[5] + sd[2] * s->proj[9]) / cw;
                    u.sunuv[0] = nx * 0.5f + 0.5f, u.sunuv[1] = 0.5f - ny * 0.5f;
                    float out = fmaxf(fabsf(u.sunuv[0] - 0.5f), fabsf(u.sunuv[1] - 0.5f)) - 0.5f; /* past the edge */
                    u.sunuv[2] = fminf(fmaxf(1.0f - out / 0.5f, 0.0f), 1.0f) * fminf(cw / 0.3f, 1.0f);
                }
            }
            /* the frame before, for the temporal pass: its camera, if it was the frame just before, at this
             * size, and not a jump away */
            {
                float vinv2[16], m[16];
                int ok = g_fx.hist_serial && g_fx.hist_serial + 1 == g_serial && g_fx.hist[0] &&
                    g_fx.hist[0].width == aw && g_fx.hist[0].height == ah && mat_inverse(vinv2, s->view);
                if (ok)
                {
                    float dx = vinv2[12] - g_fx.prev_cam[0], dy = vinv2[13] - g_fx.prev_cam[1], dz = vinv2[14] - g_fx.prev_cam[2];
                    ok = dx * dx + dy * dy + dz * dz < 25.0f;
                }
                if (ok)
                {
                    mat_mul(m, vinv2, g_fx.prev_view);
                    mat_mul(u.reproj, m, g_fx.prev_proj);
                    u.hist[0] = 1.0f;
                }
                /* the pattern's turn: a golden-ratio step each frame */
                u.hist[1] = (float)fmod((double)g_serial * 0.6180339887, 1.0);
                u.hist[2] = fminf(fmaxf(g_fxs.temporal, 0.0f), 0.95f);
                memcpy(g_fx.prev_view, s->view, 64), memcpy(g_fx.prev_proj, s->proj, 64);
                if (mat_inverse(vinv2, s->view))
                    memcpy(g_fx.prev_cam, vinv2 + 12, 12);
            }
            /* the ray tracing's structures, for this frame's camera */
            int traced = 0;
            if (g_fxs.rt > 0.0f && have_v)
            {
                float invP[16], cw[16];
                memcpy(u.vw, vinv, 64);
                /* the sun as the effects keep it: in the world, the last a lit draw gave */
                memcpy(u.sunw, g_fx.sunw, 12), u.sunw[3] = u.sun[3];
                if (g_fxs.rt_sun_elev != 0.0f)
                {
                    /* the tester's sun: FFXI's world y points down */
                    float el = g_fxs.rt_sun_elev * (float)M_PI / 180.0f, az = g_fxs.rt_sun_azim * (float)M_PI / 180.0f;
                    u.sunw[0] = cosf(el) * cosf(az), u.sunw[1] = -sinf(el), u.sunw[2] = cosf(el) * sinf(az), u.sunw[3] = 1.0f;
                }
                /* for the maps and the world's draws until the next scene: the sun, and the shadow's
                 * strength by day (the tester's sun at full) */
                memcpy(g_rtx.sun_now, u.sunw, 16);
                float strength = g_fxs.rt_shadow * (g_fxs.rt_sun_elev != 0.0f ? 1.0f : g_rtx.day);
                g_rtx.sun_now[3] = strength > 0.0f ? g_rtx.sun_now[3] : 0.0f;
                g_rtx.k_now[0] = strength, g_rtx.k_now[2] = fmaxf(g_fxs.rt_distance, 1.0f);
                if (mat_inverse(invP, s->proj))
                    mat_mul(cw, invP, vinv), traced = rt_frame(cw, vinv, s->view, s->proj, s->cam != 0);
                memcpy(u.ivp, g_rtx.ray_ivp, 64), memcpy(u.cam, g_rtx.ray_cam, 12);
            }
            u.bloom[0] = g_fxs.threshold, u.bloom[1] = g_fxs.bloom, u.bloom[2] = 0.25f;
            u.rays[0] = u.sunuv[2] > 0.0f ? g_fxs.rays : 0.0f, u.rays[1] = g_fxs.rays_decay, u.rays[2] = g_fxs.rays_length;
            if (fx_tex(&g_fx.src, ct.pixelFormat, ct.width, ct.height) && fx_tex(&g_fx.ao0, MTLPixelFormatRGBA16Float, aw, ah) &&
                fx_tex(&g_fx.ao1, MTLPixelFormatRGBA16Float, aw, ah) && fx_tex(&g_fx.b1a, MTLPixelFormatRGBA16Float, bw, bh) &&
                fx_tex(&g_fx.b1b, MTLPixelFormatRGBA16Float, bw, bh) &&
                fx_tex(&g_fx.b2a, MTLPixelFormatRGBA16Float, (bw + 1) / 2, (bh + 1) / 2) &&
                fx_tex(&g_fx.b2b, MTLPixelFormatRGBA16Float, (bw + 1) / 2, (bh + 1) / 2) &&
                fx_tex(&g_fx.ra, MTLPixelFormatRGBA16Float, bw, bh) && fx_tex(&g_fx.rb, MTLPixelFormatRGBA16Float, bw, bh))
            {
                if (!g_fx.comp_pipe || g_fx.comp_fmt != ct.pixelFormat)
                {
                    [g_fx.comp_pipe release];
                    g_fx.comp_pipe = fx_pipeline(@"fx_comp", ct.pixelFormat);
                    g_fx.comp_fmt = ct.pixelFormat;
                }
                if (g_fx.comp_pipe)
                {
                    id<MTLBlitCommandEncoder> b = [cmd() blitCommandEncoder];
                    [b copyFromTexture:ct sourceSlice:0 sourceLevel:0 toTexture:g_fx.src destinationSlice:0 destinationLevel:0
                            sliceCount:1 levelCount:1];
                    [b endEncoding];
                    static const int32_t across[2] = { 1, 0 }, down[2] = { 0, 1 }, across2[2] = { 2, 0 }, down2[2] = { 0, 2 };
                    MTLViewport q = fx_full(g_fx.ao0), qb = fx_full(g_fx.b1a), e = fx_full(g_fx.b2a);
                    id<MTLTexture> ao_out = g_fx.ao0;
                    if (u.ao[1] > 0.0f || u.shadow[0] > 0.0f || u.smap[0] > 0.0f)
                    {
                        id<MTLTexture> ao_in[3] = { depth, u.smap[0] > 0.0f ? g_fx.smap : g_fx.sdummy,
                            u.smap[0] > 0.0f && u.smapn2[1] > 0.0f ? g_fx.smapn : g_fx.sdummy };
                        fx_pass(g_fx.ao0, MTLLoadActionDontCare, g_fx.ao_pipe, q, &u, ao_in, 3, NULL);
                        fx_pass(g_fx.ao1, MTLLoadActionDontCare, g_fx.blur_pipe, q, &u, &g_fx.ao0, 1, across);
                        fx_pass(g_fx.ao0, MTLLoadActionDontCare, g_fx.blur_pipe, q, &u, &g_fx.ao1, 1, down);
                        if (g_fxs.temporal > 0.0f && fx_tex(&g_fx.hist[0], MTLPixelFormatRGBA16Float, aw, ah) &&
                            fx_tex(&g_fx.hist[1], MTLPixelFormatRGBA16Float, aw, ah))
                        {
                            int to = g_fx.hist_at ^ 1;
                            id<MTLTexture> t_in[2] = { g_fx.ao0, g_fx.hist[g_fx.hist_at] };
                            fx_pass(g_fx.hist[to], MTLLoadActionDontCare, g_fx.temporal_pipe, q, &u, t_in, 2, NULL);
                            ao_out = g_fx.hist[to];
                            g_fx.hist_at = to, g_fx.hist_serial = g_serial;
                        }
                    }
                    if (u.bloom[1] > 0.0f)
                    {
                        fx_pass(g_fx.b1a, MTLLoadActionDontCare, g_fx.bright_pipe, qb, &u, &g_fx.src, 1, NULL);
                        fx_pass(g_fx.b1b, MTLLoadActionDontCare, g_fx.gauss_pipe, qb, &u, &g_fx.b1a, 1, across2);
                        fx_pass(g_fx.b1a, MTLLoadActionDontCare, g_fx.gauss_pipe, qb, &u, &g_fx.b1b, 1, down2);
                        fx_pass(g_fx.b2a, MTLLoadActionDontCare, g_fx.down_pipe, e, &u, &g_fx.b1a, 1, NULL);
                        fx_pass(g_fx.b2b, MTLLoadActionDontCare, g_fx.gauss_pipe, e, &u, &g_fx.b2a, 1, across2);
                        fx_pass(g_fx.b2a, MTLLoadActionDontCare, g_fx.gauss_pipe, e, &u, &g_fx.b2b, 1, down2);
                    }
                    if (u.rays[0] > 0.0f)
                    {
                        id<MTLTexture> mask_in[2] = { g_fx.src, depth };
                        fx_pass(g_fx.ra, MTLLoadActionDontCare, g_fx.raymask_pipe, qb, &u, mask_in, 2, NULL);
                        fx_pass(g_fx.rb, MTLLoadActionDontCare, g_fx.rays_pipe, qb, &u, &g_fx.ra, 1, NULL);
                        fx_pass(g_fx.ra, MTLLoadActionDontCare, g_fx.gauss_pipe, qb, &u, &g_fx.rb, 1, across);
                        fx_pass(g_fx.rb, MTLLoadActionDontCare, g_fx.gauss_pipe, qb, &u, &g_fx.ra, 1, down);
                    }
                    id<MTLTexture> rt_out = traced && g_fxs.debug == 6.0f ? rt_view(&u, depth, aw, ah) : nil;
                    id<MTLTexture> comp_in[7] = { g_fx.src, ao_out, depth, g_fx.b1a, g_fx.b2a, g_fx.rb, rt_out ? rt_out : g_fx.src };
                    fx_pass(ct, MTLLoadActionLoad, g_fx.comp_pipe, (MTLViewport){ vx, vy, vw, vh, 0, 1 }, &u, comp_in, 7, NULL);
                }
            }
        }
        color->scene = 0; /* the effects changed it: its mips are behind (scene_mips) */
        color->used = g_serial;
        if (gfx_profiling)
        {
            static double last;
            g_fx.st_frames++;
            g_fx.st_own += s->sun_dir[3] != 0.0f;
            g_fx.st_map += g_fx.st_drawn_this;
            g_fx.st_cmin = g_fx.st_frames == 1 || g_ncasters < g_fx.st_cmin ? g_ncasters : g_fx.st_cmin;
            g_fx.st_cmax = g_ncasters > g_fx.st_cmax ? g_ncasters : g_fx.st_cmax;
            double now = CACurrentMediaTime();
            if (now - last > 2.0)
            {
                fprintf(stderr, "[recomp] gfx: shadows: %u frames, %u with the sun's own light, %u with a map; casters %u..%u; "
                    "%u cached; map %.0f units across; sun %.2f %.2f %.2f\n", g_fx.st_frames, g_fx.st_own, g_fx.st_map, g_fx.st_cmin,
                    g_fx.st_cmax, g_fx.st_cached, g_fx.st_across, g_fx.sunw[0], g_fx.sunw[1], g_fx.sunw[2]);
                if (g_fxs.rt > 0.0f && g_rtx.ok)
                    fprintf(stderr, "[recomp] gfx: rt: %u meshes, %u instances (%u triangles), %u moving triangles (%u drawn "
                        "after the effects, 2 s), %u built this frame, %u waiting\n", g_rtx.nkept, g_rtx.ninst, g_rtx.kept_tris,
                        g_rtx.moving_tris, g_rtx.late_used, g_rtx.built, g_rtx.waiting),
                    g_rtx.late_used = 0;
                if (rt_inpass())
                    fprintf(stderr, "[recomp] gfx: rt: shaded draws: %u with the frame's camera from an anchor, %u with the "
                        "last frame's, %u fixed-function, %u with none; frames from a solo place %u (%u held); sun %.2f %.2f %.2f %s, strength %.2f; maps %s: %u "
                        "instances, %u draws, texels %.3f / %.3f / %.3f units\n", g_rtx.dr_exact,
                        g_rtx.dr_pred, g_rtx.dr_ffp, g_rtx.dr_none, g_rtx.dr_solo, g_rtx.dr_solo_far, g_rtx.sun_now[0], g_rtx.sun_now[1], g_rtx.sun_now[2],
                        g_rtx.sun_now[3] > 0.0f ? "on" : "off", g_rtx.k_now[0], g_sm.ok ? "on" : "off", g_sm.instances,
                        g_sm.drawn, g_sm.c[0].texel, g_sm.c[1].texel, g_sm.c[2].texel),
                    g_rtx.dr_exact = g_rtx.dr_pred = g_rtx.dr_ffp = g_rtx.dr_none = g_rtx.dr_solo = g_rtx.dr_solo_far = 0;
                if (g_fxs.rt > 0.0f && g_rtx.ok)
                    fprintf(stderr, "[recomp] gfx: rt: camera from anchors %u frames (%.1f agreeing, spread at most %.5f units), the "
                        "scene's %u; %u chained, %u from the draws, %u the scene's own, %u guessed (chain: %u tries, %u meshes drawn, %u as often as before, %u drawn the frame before, %u pairs, %u agreeing); %u anchors; the scene's camera off by at most %.4f units; %u new zones\n", g_rtx.anchored,
                        g_rtx.anchored ? (double)g_rtx.cam_agree / g_rtx.anchored : 0.0, g_rtx.cam_spread, g_rtx.anchor_fallback,
                        g_rtx.cam_chain, g_rtx.cam_from_draws, g_rtx.cam_fresh, g_rtx.cam_guessed, g_rtx.ch_calls, g_rtx.ch_seen, g_rtx.ch_samecount, g_rtx.ch_prev,
                        g_rtx.ch_cand, g_rtx.ch_most, g_rtx.nanchor, g_rtx.cam_scene, g_rtx.zones),
                    g_rtx.zones = 0, g_rtx.cam_from_draws = g_rtx.cam_chain = g_rtx.cam_guessed = g_rtx.cam_fresh = 0,
                    g_rtx.ch_calls = g_rtx.ch_cand = g_rtx.ch_most = g_rtx.ch_seen = g_rtx.ch_samecount = g_rtx.ch_prev = 0,
                    g_rtx.anchored = g_rtx.anchor_fallback = g_rtx.cam_agree = 0, g_rtx.cam_spread = g_rtx.cam_scene = 0.0;
                if (g_fxs.rt > 0.0f && g_rtx.ok)
                    fprintf(stderr, "[recomp] gfx: rt: places: %u as kept, %u moved, %u new, %u drawn again; "
                        "a new one's nearest: %.5f rotation, %.4f units; %u not placed (out of the world), %u over %u a mesh; "
                        "most a mesh has %u; %u dropped (in view, not drawn)\n", g_rtx.pl_same, g_rtx.pl_moved, g_rtx.pl_new, g_rtx.pl_samefr, g_rtx.pl_nd,
                        g_rtx.pl_nt, g_rtx.pl_far, g_rtx.pl_full, (unsigned)RT_PLACES, rt_most_places(), g_rtx.pl_gone),
                    g_rtx.pl_same = g_rtx.pl_moved = g_rtx.pl_new = g_rtx.pl_samefr = g_rtx.pl_far = g_rtx.pl_full = g_rtx.pl_gone = 0;
                last = now, g_fx.st_frames = g_fx.st_own = g_fx.st_map = g_fx.st_cmax = 0;
            }
        }
        {
            __typeof__(g_fx.trace[0])* t = &g_fx.trace[g_fx.ntrace++ % 1200];
            float inv[16];
            t->serial = g_serial, t->live = g_fx.tr_live, t->cached = g_fx.tr_cached, t->skipped = g_fx.tr_skipped;
            memcpy(t->fog, g_fx.tr_fog, 12);
            t->own = s->sun_dir[3] != 0.0f, t->depth = g_fx.tr_depth, t->dl = g_fx.tr_dl, t->al = g_fx.tr_al, t->strength = g_fx.tr_strength, t->day = g_fx.tr_day;
            memcpy(t->sun, g_fx.sunw, 12);
            if (mat_inverse(inv, s->view))
                memcpy(t->cam, inv + 12, 12);
            t->across = g_fx.st_across;
            g_fx.tr_live = g_fx.tr_cached = g_fx.tr_skipped = 0, g_fx.tr_strength = g_fx.tr_day = 0;
        }
        g_fx.st_drawn_this = 0;
        casters_clear();
        if (gfx_profiling)
            g_prof.draw_ns += gfx_now_ns() - t0;
    }
}

/* --- frames ---------------------------------------------------------------------------------------------------- */
static void frame_end(void)
{
    casters_clear();
    id<MTLCommandBuffer> c = cmd();
    uint64_t serial = g_serial;
    dispatch_semaphore_t sem = g_frames_sem;
    [c addCompletedHandler:^(id<MTLCommandBuffer> done) {
        (void)done;
        atomic_store(&g_completed, serial);
        dispatch_semaphore_signal(sem);
    }];
    submit(0);
    g_frame_open = 0;
    g_frame = (g_frame + 1) % FRAMES;
    g_serial++;
}

/* "60 FPS 16.7ms": the text of the overlay, from the presents of the last half second */
static void fps_tick(void)
{
    double now = CACurrentMediaTime();
    if (!g_fps_since)
        g_fps_since = now;
    g_fps_frames++;
    double dt = now - g_fps_since;
    if (dt >= 0.5)
    {
        double fps = g_fps_frames / dt;
        snprintf(g_fps_text, sizeof g_fps_text, "%.0f FPS %.1fms", fps, dt * 1000.0 / g_fps_frames);
        g_fps_since = now, g_fps_frames = 0;
    }
}

static void draw_overlay(id<MTLRenderCommandEncoder> e, double w, double h)
{
    struct
    {
        float rect[4], scale;
        uint32_t n, pad[2], text[32];
    } u;
    memset(&u, 0, sizeof u);
    for (const char* c = g_fps_text; *c && u.n < 32; ++c)
    {
        uint32_t k = *c >= '0' && *c <= '9' ? (uint32_t)(*c - '0') : *c == 'F' ? 10 : *c == 'P' ? 11 : *c == 'S' ? 12
            : *c == 'm' ? 13 : *c == 's' ? 14 : *c == '.' ? 15 : 16;
        u.text[u.n++] = k;
    }
    u.scale = (float)(h >= 1400 ? 3 : 2);
    u.rect[0] = u.rect[1] = 4 * u.scale;
    u.rect[2] = (float)(u.n * 6 + 3) * u.scale, u.rect[3] = 11 * u.scale;
    float size[2] = { (float)w, (float)h };
    [e setRenderPipelineState:g_overlay_pipe];
    [e setVertexBytes:&u length:sizeof u atIndex:0];
    [e setVertexBytes:size length:sizeof size atIndex:1];
    [e setFragmentBytes:&u length:sizeof u atIndex:0];
    [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
}

void gfx_present(GfxTex* bb)
{
    if (!g_dev)
        return;
    uint64_t present_start = gfx_profiling ? gfx_now_ns() : 0;
    g_present_thread = pthread_self();
    @autoreleasepool
    {
        flush_pass();
        if (g_layer && bb)
        {
            int pw = 0, ph = 0;
            if (g_window)
                SDL_GetWindowSizeInPixels(g_window, &pw, &ph);
            if (pw > 0 && ph > 0 && (g_layer.drawableSize.width != pw || g_layer.drawableSize.height != ph))
                g_layer.drawableSize = CGSizeMake(pw, ph);
            uint64_t t0 = gfx_profiling ? gfx_now_ns() : 0;
            id<CAMetalDrawable> drawable = [g_layer nextDrawable];
            if (gfx_profiling)
                g_prof.drawable_ns += gfx_now_ns() - t0;
            if (drawable)
            {
                MTLRenderPassDescriptor* p = [MTLRenderPassDescriptor renderPassDescriptor];
                p.colorAttachments[0].texture = drawable.texture;
                p.colorAttachments[0].loadAction = MTLLoadActionDontCare;
                p.colorAttachments[0].storeAction = MTLStoreActionStore;
                id<MTLRenderCommandEncoder> e = [cmd() renderCommandEncoderWithDescriptor:p];
                float sharpen = g_fxs.fx != 0.0f ? g_fxs.sharpen : 0.0f;
                [e setRenderPipelineState:sharpen > 0.0f && g_present_cas_pipe ? g_present_cas_pipe : g_present_pipe];
                [e setFragmentBytes:&sharpen length:sizeof sharpen atIndex:0];
                [e setFragmentTexture:bb->view atIndex:0];
                [e setFragmentSamplerState:g_present_samp atIndex:0];
                [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                if (g_fxs.fps != 0.0f && g_overlay_pipe)
                    draw_overlay(e, drawable.texture.width, drawable.texture.height);
                [e endEncoding];
                [cmd() presentDrawable:drawable];
                bb->used = g_serial;
            }
        }
        fps_tick();
        fx_reload();
        frame_end();
    }
    if (gfx_profiling)
        prof_frame(present_start);
}

void gfx_finish(void)
{
    if (!g_dev)
        return;
    @autoreleasepool
    {
        submit(1);
    }
}

void gfx_resize(uint32_t w, uint32_t h)
{
    if (g_layer)
        g_layer.drawableSize = CGSizeMake(w, h);
}

int gfx_init(void* window, int vsync)
{
    if (g_dev)
    {
        /* up already (host64's sign-in screen, on this same window): the game's present interval */
        if (g_layer)
            g_layer.displaySyncEnabled = vsync ? YES : NO;
        return 1;
    }
    @autoreleasepool
    {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev)
        {
            fprintf(stderr, "[recomp] gfx: no Metal device\n");
            return 0;
        }
        g_queue = [g_dev newCommandQueue];
        g_frames_sem = dispatch_semaphore_create(FRAMES);
        g_dummy = [g_dev newBufferWithLength:256 options:MTLResourceStorageModeShared];
        g_util = compile(CLEAR_MSL);
        g_pipe_queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
        g_pipe_file_queue = dispatch_queue_create("ffxi.pipeline-cache", DISPATCH_QUEUE_SERIAL);
        if (!g_sync_pipelines)
            prewarm_pipelines();
        MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
        id<MTLFunction> vf = [g_util newFunctionWithName:@"present_vs"], ff = [g_util newFunctionWithName:@"present_fs"];
        pd.vertexFunction = vf;
        pd.fragmentFunction = ff;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        g_present_pipe = [g_dev newRenderPipelineStateWithDescriptor:pd error:NULL];
        [ff release];
        ff = [g_util newFunctionWithName:@"present_cas_fs"];
        pd.fragmentFunction = ff;
        g_present_cas_pipe = [g_dev newRenderPipelineStateWithDescriptor:pd error:NULL];
        [vf release];
        [ff release];
        [pd release];
        /* the overlay: blended over the frame; FFXI_FPS=0 hides it (the settings file's fps, from
         * Config > Modern, has the last word) */
        const char* prof = getenv("FFXI_PROFILE");
        gfx_profiling = prof && prof[0] && prof[0] != '0';
        fx_config();
        const char* show = getenv("FFXI_FPS");
        if (show && show[0] == '0')
            g_fxs.fps = 0.0f;
        pd = [[MTLRenderPipelineDescriptor alloc] init];
        vf = [g_util newFunctionWithName:@"overlay_vs"], ff = [g_util newFunctionWithName:@"overlay_fs"];
        pd.vertexFunction = vf;
        pd.fragmentFunction = ff;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        pd.colorAttachments[0].blendingEnabled = YES;
        pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        g_overlay_pipe = [g_dev newRenderPipelineStateWithDescriptor:pd error:NULL];
        [vf release];
        [ff release];
        [pd release];
        MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        g_present_samp = [g_dev newSamplerStateWithDescriptor:sd];
        [sd release];
        if (window)
        {
            g_window = (SDL_Window*)window;
            g_view = SDL_Metal_CreateView(g_window);
            g_layer = g_view ? (CAMetalLayer*)SDL_Metal_GetLayer(g_view) : nil;
            if (g_layer)
            {
                g_layer.device = g_dev;
                g_layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
                g_layer.framebufferOnly = YES;
                g_layer.displaySyncEnabled = vsync ? YES : NO;
                int pw = 0, ph = 0;
                SDL_GetWindowSizeInPixels(g_window, &pw, &ph);
                if (pw > 0 && ph > 0)
                    g_layer.drawableSize = CGSizeMake(pw, ph);
            }
            else
                fprintf(stderr, "[recomp] gfx: no Metal layer for the window: %s\n", SDL_GetError());
        }
        fprintf(stderr, "[recomp] gfx: Metal on %s\n", [[g_dev name] UTF8String]);
        return 1;
    }
}
