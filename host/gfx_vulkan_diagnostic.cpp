/* Native Vulkan backend for the portable D3D8 frontend.
 *
 * This backend uses no Wine, DXVK or translated graphics code. D3D8 object/state
 * semantics remain in d3d8.c; the existing HLSL generator supplies the fixed
 * function and shader-model-1 programs. No descriptor indexing is required.
 * Upload rings and descriptor sets live until their frame fence completes.
 * BC1/2/3 are decoded without recompression when the native GPU lacks BCn.
 * Optional Metal/D3D12 modern scene effects are not implemented here.
 */
#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
extern "C" {
#include "gfx.h"
#include "gfx_hlsl.h"
#include "gfx_spirv.h"
}

namespace {
constexpr unsigned FRAMES = 3, PROBES = 8, READBACKS = (FRAMES + 1) * PROBES;
constexpr VkDeviceSize RING_BYTES = 8u << 20, SLAB_BYTES = 2u << 20;
constexpr uint32_t FOURCC(char a, char b, char c, char d) {
    return uint32_t(a) | uint32_t(b) << 8 | uint32_t(c) << 16 | uint32_t(d) << 24;
}
constexpr uint32_t DXT1 = FOURCC('D','X','T','1'), DXT2 = FOURCC('D','X','T','2'),
                   DXT3 = FOURCC('D','X','T','3'), DXT4 = FOURCC('D','X','T','4'), DXT5 = FOURCC('D','X','T','5');
struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };
uint32_t failures = 0;
std::recursive_mutex mutex;
[[noreturn]] void fail(const std::string& why) {
    ++failures; std::fprintf(stderr, "[gfx-vulkan] failure: %s\n", why.c_str()); throw Failure(why);
}
void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) fail(std::string(what) + ": VkResult " + std::to_string(r));
}
#define VK_CHECK(x) check((x), #x)
VkDeviceSize aligned(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) & ~(a - 1); }
uint32_t mip_size(uint32_t n, uint32_t level) { return std::max(1u, n >> level); }
uint32_t block_bytes(uint32_t fmt) { return fmt == DXT1 ? 8 : (fmt == DXT2 || fmt == DXT3 || fmt == DXT4 || fmt == DXT5) ? 16 : 0; }
uint32_t d3d_bpp(uint32_t fmt) {
    if (fmt == 28 || fmt == 50) return 1;
    if ((fmt >= 23 && fmt <= 26) || fmt == 51 || fmt == 60 || fmt == 80) return 2;
    return 4;
}
std::string bytes_key(const void* p, size_t n) { return std::string(static_cast<const char*>(p), n); }
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* cpu = nullptr;
    VkDeviceSize size = 0;
};
struct Ring { Buffer b; VkDeviceSize used = 0; };
struct Slab { Buffer b; uint32_t block = 0; std::vector<uint32_t> free; };
struct Slot { Slab* slab = nullptr; uint32_t offset = 0, size = 0; };
struct Readback { Buffer b; uint64_t serial = 0; uint32_t face = 0, level = 0, probe = 0; };
#if defined(FFXI_ANDROID_VULKAN)
constexpr unsigned KEYED_HISTORIES = 128, KEYED_SLOTS = FRAMES + 1;
struct KeyedReadback { Readback transfer; uint64_t issuedFrame = UINT64_MAX; };
struct KeyedHistory {
    uint64_t key = 0, lastReadFrame = UINT64_MAX;
    uint32_t face = 0, level = 0, maxAge = 0;
    std::array<KeyedReadback,KEYED_SLOTS> slots;
};
#endif
struct Frame {
    VkCommandPool commands = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> lists;
    unsigned cursor = 0;
    VkFence fence = VK_NULL_HANDLE;
    uint64_t lastSerial = 0;
    VkSemaphore acquired = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> pools;
    unsigned poolCursor = 0;
    std::vector<Ring> rings;
    unsigned ringCursor = 0;
    std::unordered_map<std::string, VkDescriptorSet> sets;
};
struct Dead { uint64_t serial; std::function<void()> destroy; };
struct RenderKey { VkFormat color, depth; };
struct Framebuffer {
    VkFramebuffer fb;
    VkImage color, depth;
};
struct ShaderPair { VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE; };
struct LibKey { GfxVsKey vs; GfxFsKey fs; };
struct PipeKey {
    LibKey lib;
    GfxPipeKey pipe;
    GfxDepthKey depth;
    VkFormat color, z;
    int32_t zbias;
    uint8_t cull, fill, prim, x8;
};
struct DescriptorKey {
    VkBuffer stream[4], uniform;
    VkImageView two[8], cube[8];
    VkSampler sampler[8];
};
struct Profile {
    uint64_t frames = 0, draws = 0, bytes = 0, front = 0, encode = 0, present = 0, since = 0;
    uint64_t pipelines = 0, descriptorSets = 0, passes = 0, submissions = 0;
    uint64_t skips[GFX_NSKIPS]{};
#if defined(FFXI_ANDROID_VULKAN)
    uint64_t probeKnown = 0, probeUnknown = 0, probeSync = 0;
    uint64_t keyedReads = 0, keyedLate = 0, keyedExact = 0, keyedDuplicate = 0,
        keyedOverflow = 0, keyedBusy = 0, keyedStale = 0, keyedQueued = 0;
#endif
};
/* Default-off Android diagnostic; no new GPU commands or waits. */
enum class DiagReason : uint32_t {
    Other, Target, Sampling, Upload, Initialize, Copy, Readback, Submit, Destroy,
    Present, Attachment, Count
};
constexpr unsigned DIAG_MAX_PASSES = 256, DIAG_MAX_BINDS = 512;
struct DiagPass {
    uint64_t colorId=0, depthId=0, colorImage=0, depthImage=0, drawStart=0, draws=0;
    uint64_t causeImage=0, colorPixels=0, depthPixels=0, nominalBytes=0;
    uint32_t width=0,height=0,depthWidth=0,depthHeight=0,face=0,level=0;
    uint32_t colorFmt=0,depthFmt=0,clearCount=0,firstClearFlags=0,fullFirstClearFlags=0;
    uint32_t firstClearColor=0,firstClearZBits=0,firstClearStencil=0;
    uint32_t reason=0,beforeLayout=0,afterLayout=0;
    bool firstWasClear=false,causeAttachment=false;
};
struct DiagBind {
    uint64_t fromColor=0,toColor=0,fromDepth=0,toDepth=0,drawsAt=0;
    bool passWasOpen=false,previousBindWithoutWork=false;
};
struct DiagFrame {
    uint64_t frame=0,startNs=0,endNs=0,wallNs=0,residualNs=0;
    uint64_t frameFenceWaitNs=0,syncFenceWaitNs=0,deviceIdleWaitNs=0;
    uint64_t commandEndNs=0,queueSubmitNs=0,acquireNs=0,queuePresentNs=0,presentHostNs=0;
    uint64_t presentBoundaryBefore=0,submitCalls=0,waitSubmitCalls=0;
    uint64_t drawStart=0,byteStart=0,draws=0,uploadedBytes=0;
    uint64_t knownStart=0,unknownStart=0,lateStart=0,exactStart=0;
    uint64_t known=0,unknown=0,late=0,exact=0;
    uint64_t allocationCount=0,allocatedBytes=0,readbackAllocations=0,bufferDestroys=0;
    uint64_t colorPixels=0,depthPixels=0,nominalBytes=0;
    uint64_t fullFirstColorClearPixels=0,fullFirstDepthClearPixels=0;
    uint64_t passCount=0,bindCount=0,emptyBindCount=0;
    uint32_t passStored=0,bindStored=0;
    std::array<uint64_t,size_t(DiagReason::Count)> reasonCounts{};
    std::array<uint64_t,size_t(DiagReason::Count)> reasonBytes{};
    std::array<DiagPass,DIAG_MAX_PASSES> passes;
    std::array<DiagBind,DIAG_MAX_BINDS> binds;
};
struct Diagnostic {
    FILE* file=nullptr;
    bool pending=true,active=false,passTracked=false,pendingBind=false,bindHadWork=false;
    uint32_t completed=0;
    DiagPass pass;
    std::vector<DiagFrame> frames;
    ~Diagnostic(){if(file)std::fclose(file);}
};

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceFeatures features{};
    uint32_t family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swap = VK_NULL_HANDLE;
    VkFormat swapFormat = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR swapColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkCompositeAlphaFlagBitsKHR swapAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    VkExtent2D extent{};
    std::vector<VkImage> swapImages;
    std::vector<VkImageLayout> swapLayouts;
    std::vector<VkSemaphore> rendered;
    SDL_Window* window = nullptr;
    bool vsync = true, swapDirty = false, swapReview = false, notedCompositorSuboptimal = false;
    uint32_t wantWidth = 0, wantHeight = 0;
    VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    VkFence syncFence = VK_NULL_HANDLE;
    Frame frames[FRAMES];
    unsigned frame = 0;
    uint64_t frameNumber = 0, nextSerial = 1, completedSerial = 0;
    bool frameOpen = false;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    bool pass = false;
    VkPipeline boundPipeline = VK_NULL_HANDLE;
    VkRenderPass activePass = VK_NULL_HANDLE;
    struct GfxTex* rt = nullptr;
    struct GfxTex* ds = nullptr;
    uint32_t rtFace = 0, rtLevel = 0;
    struct GfxTex *dummy2 = nullptr, *dummyCube = nullptr;
    Buffer dummyBuffer;
    std::vector<std::unique_ptr<Slab>> slabs;
    std::vector<Dead> dead;
    std::unordered_map<std::string,VkRenderPass> renderPasses;
    std::unordered_map<std::string,Framebuffer> framebuffers;
    std::unordered_map<std::string,ShaderPair> shaders;
    std::unordered_map<std::string,VkPipeline> pipelines;
    std::unordered_map<std::string,VkSampler> samplers;
    Profile prof;
    std::unique_ptr<Diagnostic> diagnostic;
    uint64_t diagnosticNextTexture=1;
    std::thread::id presentThread;
} g;
}

/* Opaque frontend objects own their D3D representation and native Vulkan data. */
struct GfxBuf { Slot allocation; uint32_t size = 0; uint64_t used = 0; };
struct GfxTex {
    uint64_t diagnosticId=0;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView sample = VK_NULL_HANDLE;
    VkFormat native = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t fmt = 0, w = 0, h = 0, levels = 0, faces = 0;
    int type = 0, use = 0;
    uint32_t gpuBpp = 4;
    bool x8 = false, compressed = false;
    std::vector<VkImageView> attachments;
    std::vector<VkImageLayout> layouts;
    std::vector<std::vector<uint8_t>> shadow;
    std::vector<bool> shadowValid;
    Readback readbacks[READBACKS];
    uint64_t readFrame = UINT64_MAX;
    uint32_t readCount = 0;
#if defined(FFXI_ANDROID_VULKAN)
    // Lazy storage belongs to this live texture, so destroyed/recycled texture
    // pointers cannot inherit history. At most128 logical lifetimes are kept.
    std::vector<std::unique_ptr<KeyedHistory>> keyed;
#endif
};

namespace {
DiagFrame* diag_frame() {
    return g.diagnostic && g.diagnostic->active ? &g.diagnostic->frames[g.diagnostic->completed] : nullptr;
}
uint64_t diag_image(VkImage image) {return uint64_t(reinterpret_cast<uintptr_t>(image));}
uint64_t diag_tex_id(GfxTex* t) {
    if(!t)return 0;
    if(!t->diagnosticId)t->diagnosticId=g.diagnosticNextTexture++;
    return t->diagnosticId;
}
const char* diag_reason_name(uint32_t reason) {
    static const char* names[]={"other","target-change","sampling-transition","upload",
        "initialization","copy","readback","submit","destruction","presentation","attachment-transition"};
    return reason<size_t(DiagReason::Count)?names[reason]:"invalid";
}
uint32_t diag_native_bpp(GfxTex* t) {
    if(t->native==VK_FORMAT_D32_SFLOAT_S8_UINT)return 8;
    if(t->native==VK_FORMAT_D24_UNORM_S8_UINT)return 4;
    return t->gpuBpp;
}
uint64_t diag_partitioned(const DiagFrame& f) {
    return f.frameFenceWaitNs+f.syncFenceWaitNs+f.deviceIdleWaitNs+f.commandEndNs+
        f.queueSubmitNs+f.acquireNs+f.queuePresentNs;
}
template<class F> VkResult diag_timed(uint64_t DiagFrame::* bucket,F&& body) {
    DiagFrame* f=diag_frame();if(!f)return body();
    uint64_t start=gfx_now_ns();VkResult result=body();f->*bucket+=gfx_now_ns()-start;return result;
}
void diag_allocation(VkDeviceSize size,VkBufferUsageFlags usage) {
    if(auto* f=diag_frame()){++f->allocationCount;f->allocatedBytes+=size;if(usage==VK_BUFFER_USAGE_TRANSFER_DST_BIT)++f->readbackAllocations;}
}
void diag_destroy_buffer(){if(auto* f=diag_frame())++f->bufferDestroys;}
void diag_submit(bool wait){if(auto* f=diag_frame()){++f->submitCalls;if(wait)++f->waitSubmitCalls;}}
void diag_begin_pass() {
    if(!diag_frame())return;
    auto& d=*g.diagnostic;d.pass=DiagPass{};d.passTracked=true;d.bindHadWork=true;
    auto& p=d.pass;p.colorId=diag_tex_id(g.rt);p.depthId=diag_tex_id(g.ds);
    p.colorImage=diag_image(g.rt->image);p.depthImage=g.ds?diag_image(g.ds->image):0;
    p.width=mip_size(g.rt->w,g.rtLevel);p.height=mip_size(g.rt->h,g.rtLevel);
    p.depthWidth=g.ds?g.ds->w:0;p.depthHeight=g.ds?g.ds->h:0;
    p.face=g.rtFace;p.level=g.rtLevel;p.colorFmt=g.rt->fmt;p.depthFmt=g.ds?g.ds->fmt:0;
    p.drawStart=g.prof.draws;p.colorPixels=uint64_t(p.width)*p.height;
    p.depthPixels=g.ds?p.colorPixels:0;
    // This is nominal render-area payload, not actual compressed DRAM traffic.
    p.nominalBytes=2*(p.colorPixels*diag_native_bpp(g.rt)+(g.ds?p.depthPixels*diag_native_bpp(g.ds):0));
}
void diag_end_pass(DiagReason reason,uint64_t image,uint32_t before,uint32_t after) {
    DiagFrame* f=diag_frame();if(!f || !g.diagnostic->passTracked)return;
    auto& d=*g.diagnostic;auto& p=d.pass;d.passTracked=false;
    p.draws=g.prof.draws-p.drawStart;p.reason=uint32_t(reason);p.causeImage=image;
    p.beforeLayout=before;p.afterLayout=after;p.causeAttachment=image && (image==p.colorImage || image==p.depthImage);
    ++f->passCount;f->colorPixels+=p.colorPixels;f->depthPixels+=p.depthPixels;f->nominalBytes+=p.nominalBytes;
    if(p.fullFirstClearFlags&1)f->fullFirstColorClearPixels+=p.colorPixels;
    if(p.fullFirstClearFlags&2)f->fullFirstDepthClearPixels+=p.depthPixels;
    ++f->reasonCounts[p.reason];f->reasonBytes[p.reason]+=p.nominalBytes;
    if(f->passStored<DIAG_MAX_PASSES)f->passes[f->passStored++]=p;
}
void diag_clear(const std::vector<VkClearRect>& rects,const VkClearAttachment* attachments,
    uint32_t count,uint32_t color,float z,uint32_t stencil) {
    if(!diag_frame() || !g.diagnostic->passTracked || rects.empty())return;
    auto& p=g.diagnostic->pass;uint32_t actual=0;
    for(uint32_t i=0;i<count;++i){auto a=attachments[i].aspectMask;if(a&VK_IMAGE_ASPECT_COLOR_BIT)actual|=1;if(a&VK_IMAGE_ASPECT_DEPTH_BIT)actual|=2;if(a&VK_IMAGE_ASPECT_STENCIL_BIT)actual|=4;}
    if(!p.clearCount && g.prof.draws==p.drawStart) {
        p.firstWasClear=true;p.firstClearFlags=actual;p.firstClearColor=color;
        std::memcpy(&p.firstClearZBits,&z,4);p.firstClearStencil=stencil;
        if(rects.size()==1 && rects[0].rect.offset.x==0 && rects[0].rect.offset.y==0 &&
            rects[0].rect.extent.width==p.width && rects[0].rect.extent.height==p.height)p.fullFirstClearFlags=actual;
    }
    ++p.clearCount;
}
void diag_target_bind(GfxTex* color,GfxTex* depth) {
    DiagFrame* f=diag_frame();if(!f)return;
    auto& d=*g.diagnostic;DiagBind b;
    b.fromColor=diag_tex_id(g.rt);b.toColor=diag_tex_id(color);b.fromDepth=diag_tex_id(g.ds);b.toDepth=diag_tex_id(depth);
    b.drawsAt=g.prof.draws-f->drawStart;b.passWasOpen=g.pass;
    b.previousBindWithoutWork=d.pendingBind && !d.bindHadWork;
    ++f->bindCount;if(b.previousBindWithoutWork)++f->emptyBindCount;
    if(f->bindStored<DIAG_MAX_BINDS)f->binds[f->bindStored++]=b;
    d.pendingBind=true;d.bindHadWork=false;
}
void diag_present_begin(){if(auto* f=diag_frame())f->presentBoundaryBefore=diag_partitioned(*f);}
void diag_write_capture(Diagnostic& d) {
    FILE* out=d.file;
    std::fprintf(out,"{\"event\":\"vk-diagnostic-metadata\",\"schema\":1,\"frames\":%zu,\"GPU_timestamps\":false,\"new_GPU_waits\":false,\"default_off\":true,\"nominal_bytes_are_not_DRAM_measurements\":true,\"instrumented_FPS_excluded\":true}\n",d.frames.size());
    for(const auto& f:d.frames) {
        std::fprintf(out,"{\"event\":\"vk-diagnostic-frame\",\"frame\":%llu,\"start_ns\":%llu,\"end_ns\":%llu,\"wall_ns\":%llu,\"remaining_CPU_and_unclassified_ns\":%llu,\"frame_fence_wait_ns\":%llu,\"sync_fence_wait_ns\":%llu,\"device_idle_wait_ns\":%llu,\"command_end_ns\":%llu,\"queue_submit_ns\":%llu,\"acquire_ns\":%llu,\"queue_present_ns\":%llu,\"present_host_exclusive_ns\":%llu,\"bucket_overflow\":%s,\"draws\":%llu,\"uploaded_bytes\":%llu,\"submits\":%llu,\"wait_submits\":%llu,\"known\":%llu,\"unknown\":%llu,\"late\":%llu,\"exact\":%llu,\"buffer_allocations\":%llu,\"buffer_allocated_bytes\":%llu,\"readback_allocations\":%llu,\"buffer_destroys\":%llu,\"passes\":%llu,\"pass_records\":%u,\"binds\":%llu,\"bind_records\":%u,\"bind_without_work\":%llu,\"color_pixel_passes\":%llu,\"depth_pixel_passes\":%llu,\"nominal_load_store_bytes\":%llu,\"full_first_color_clear_pixels\":%llu,\"full_first_depth_clear_pixels\":%llu,\"pass_causes\":[",
            (unsigned long long)f.frame,(unsigned long long)f.startNs,(unsigned long long)f.endNs,(unsigned long long)f.wallNs,(unsigned long long)f.residualNs,
            (unsigned long long)f.frameFenceWaitNs,(unsigned long long)f.syncFenceWaitNs,(unsigned long long)f.deviceIdleWaitNs,(unsigned long long)f.commandEndNs,(unsigned long long)f.queueSubmitNs,(unsigned long long)f.acquireNs,(unsigned long long)f.queuePresentNs,(unsigned long long)f.presentHostNs,
            diag_partitioned(f)+f.presentHostNs>f.wallNs?"true":"false",(unsigned long long)f.draws,(unsigned long long)f.uploadedBytes,(unsigned long long)f.submitCalls,(unsigned long long)f.waitSubmitCalls,
            (unsigned long long)f.known,(unsigned long long)f.unknown,(unsigned long long)f.late,(unsigned long long)f.exact,(unsigned long long)f.allocationCount,(unsigned long long)f.allocatedBytes,(unsigned long long)f.readbackAllocations,(unsigned long long)f.bufferDestroys,
            (unsigned long long)f.passCount,f.passStored,(unsigned long long)f.bindCount,f.bindStored,(unsigned long long)f.emptyBindCount,(unsigned long long)f.colorPixels,(unsigned long long)f.depthPixels,(unsigned long long)f.nominalBytes,(unsigned long long)f.fullFirstColorClearPixels,(unsigned long long)f.fullFirstDepthClearPixels);
        for(size_t r=0;r<size_t(DiagReason::Count);++r)std::fprintf(out,"%s{\"reason\":\"%s\",\"count\":%llu,\"nominal_load_store_bytes\":%llu}",r?",":"",diag_reason_name(uint32_t(r)),(unsigned long long)f.reasonCounts[r],(unsigned long long)f.reasonBytes[r]);
        std::fputs("]}\n",out);
        for(uint32_t i=0;i<f.passStored;++i) {
            const auto& p=f.passes[i];
            std::fprintf(out,"{\"event\":\"vk-diagnostic-pass\",\"frame\":%llu,\"ordinal\":%u,\"color_id\":%llu,\"depth_id\":%llu,\"width\":%u,\"height\":%u,\"depth_width\":%u,\"depth_height\":%u,\"face\":%u,\"level\":%u,\"color_format\":%u,\"depth_format\":%u,\"draws\":%llu,\"clear_count\":%u,\"first_was_clear\":%s,\"first_clear_flags\":%u,\"full_first_clear_flags\":%u,\"first_clear_color\":%u,\"first_clear_z_bits\":%u,\"first_clear_stencil\":%u,\"end_reason\":\"%s\",\"cause_is_attachment\":%s,\"cause_image\":\"%016llx\",\"before_layout\":%u,\"after_layout\":%u,\"color_pixels\":%llu,\"depth_pixels\":%llu,\"nominal_load_store_bytes\":%llu}\n",
                (unsigned long long)f.frame,i,(unsigned long long)p.colorId,(unsigned long long)p.depthId,p.width,p.height,p.depthWidth,p.depthHeight,p.face,p.level,p.colorFmt,p.depthFmt,(unsigned long long)p.draws,p.clearCount,p.firstWasClear?"true":"false",p.firstClearFlags,p.fullFirstClearFlags,p.firstClearColor,p.firstClearZBits,p.firstClearStencil,diag_reason_name(p.reason),p.causeAttachment?"true":"false",(unsigned long long)p.causeImage,p.beforeLayout,p.afterLayout,(unsigned long long)p.colorPixels,(unsigned long long)p.depthPixels,(unsigned long long)p.nominalBytes);
        }
        for(uint32_t i=0;i<f.bindStored;++i) {
            const auto& b=f.binds[i];
            std::fprintf(out,"{\"event\":\"vk-diagnostic-bind\",\"frame\":%llu,\"ordinal\":%u,\"from_color\":%llu,\"to_color\":%llu,\"from_depth\":%llu,\"to_depth\":%llu,\"draws_at\":%llu,\"pass_was_open\":%s,\"previous_bind_without_work\":%s}\n",
                (unsigned long long)f.frame,i,(unsigned long long)b.fromColor,(unsigned long long)b.toColor,(unsigned long long)b.fromDepth,(unsigned long long)b.toDepth,(unsigned long long)b.drawsAt,b.passWasOpen?"true":"false",b.previousBindWithoutWork?"true":"false");
        }
    }
    std::fputs("{\"event\":\"vk-diagnostic-complete\",\"instrumented_FPS_excluded\":true}\n",out);
    std::fflush(out);
}
void diag_present_end(uint64_t started,uint64_t now) {
    DiagFrame* f=diag_frame();if(!f)return;
    auto& d=*g.diagnostic;f->endNs=now;f->wallNs=now-f->startNs;
    uint64_t nested=diag_partitioned(*f)-f->presentBoundaryBefore;
    f->presentHostNs=now-started>nested?now-started-nested:0;
    uint64_t recorded=diag_partitioned(*f)+f->presentHostNs;
    f->residualNs=f->wallNs>recorded?f->wallNs-recorded:0;
    f->draws=g.prof.draws-f->drawStart;f->uploadedBytes=g.prof.bytes-f->byteStart;
    f->known=g.prof.probeKnown-f->knownStart;f->unknown=g.prof.probeUnknown-f->unknownStart;
    f->late=g.prof.keyedLate-f->lateStart;f->exact=g.prof.keyedExact-f->exactStart;
    d.active=false;++d.completed;
    if(d.completed==d.frames.size()){diag_write_capture(d);g.diagnostic.reset();}
}
void diag_start_next_frame(uint64_t now) {
    if(!g.diagnostic)return;
    auto& d=*g.diagnostic;d.pending=false;d.active=true;d.passTracked=false;
    auto& f=d.frames[d.completed];f.frame=g.frameNumber;f.startNs=now;
    f.drawStart=g.prof.draws;f.byteStart=g.prof.bytes;
    f.knownStart=g.prof.probeKnown;f.unknownStart=g.prof.probeUnknown;
    f.lateStart=g.prof.keyedLate;f.exactStart=g.prof.keyedExact;
}

template<typename F> void guarded(F&& f) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    try { f(); } catch (const Failure&) {} catch (const std::exception& e) { ++failures; std::fprintf(stderr,"[gfx-vulkan] exception: %s\n", e.what()); }
}
template<typename T, typename F> T guarded_value(T fallback, F&& f) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    try { return f(); } catch (const Failure&) { return fallback; }
    catch (const std::exception& e) { ++failures; std::fprintf(stderr,"[gfx-vulkan] exception: %s\n",e.what()); return fallback; }
}
uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags required) {
    for (uint32_t i = 0; i < g.memory.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (g.memory.memoryTypes[i].propertyFlags & required) == required) return i;
    fail("no Vulkan memory type for flags " + std::to_string(required));
}
Buffer make_buffer(VkDeviceSize size, VkBufferUsageFlags use) {
    Buffer b; b.size = std::max<VkDeviceSize>(size,16);
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=b.size;ci.usage=use;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(g.device,&ci,nullptr,&b.buffer));
    VkMemoryRequirements req;vkGetBufferMemoryRequirements(g.device,b.buffer,&req);
    diag_allocation(req.size,use);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;
    ai.memoryTypeIndex=memory_type(req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkResult r=vkAllocateMemory(g.device,&ai,nullptr,&b.memory);
    if(r!=VK_SUCCESS){vkDestroyBuffer(g.device,b.buffer,nullptr);check(r,"vkAllocateMemory(buffer)");}
    r=vkBindBufferMemory(g.device,b.buffer,b.memory,0);
    if(r==VK_SUCCESS)r=vkMapMemory(g.device,b.memory,0,VK_WHOLE_SIZE,0,&b.cpu);
    if(r!=VK_SUCCESS){vkDestroyBuffer(g.device,b.buffer,nullptr);vkFreeMemory(g.device,b.memory,nullptr);check(r,"map buffer");}
    return b;
}
void destroy_buffer(Buffer b) {
    if(b.buffer)diag_destroy_buffer();
    if(b.cpu)vkUnmapMemory(g.device,b.memory);
    if(b.buffer)vkDestroyBuffer(g.device,b.buffer,nullptr);
    if(b.memory)vkFreeMemory(g.device,b.memory,nullptr);
}
void retire(std::function<void()> f) { g.dead.push_back({g.nextSerial,std::move(f)}); }
void collect() {
    for(auto& frame:g.frames)
        if(frame.lastSerial && vkGetFenceStatus(g.device,frame.fence)==VK_SUCCESS)
            g.completedSerial=std::max(g.completedSerial,frame.lastSerial);
    size_t keep=0;
    for(auto& d:g.dead) { if(d.serial<=g.completedSerial)d.destroy();else {if(&g.dead[keep]!=&d)g.dead[keep]=std::move(d);++keep;} }
    g.dead.resize(keep);
}
void begin_frame() {
    if(g.frameOpen)return;
    auto& f=g.frames[g.frame];
    VK_CHECK(diag_timed(&DiagFrame::frameFenceWaitNs,[&]{return vkWaitForFences(g.device,1,&f.fence,VK_TRUE,UINT64_MAX);}));
    g.completedSerial=std::max(g.completedSerial,f.lastSerial);collect();
    VK_CHECK(vkResetCommandPool(g.device,f.commands,0));f.cursor=0;f.ringCursor=0;f.poolCursor=0;f.sets.clear();
    for(auto& r:f.rings)r.used=0;
    for(auto p:f.pools)VK_CHECK(vkResetDescriptorPool(g.device,p,0));
    g.frameOpen=true;
}
VkCommandBuffer command() {
    begin_frame();if(g.cmd)return g.cmd;
    auto& f=g.frames[g.frame];
    if(f.cursor==f.lists.size()) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ai.commandPool=f.commands;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=1;
        VkCommandBuffer c;VK_CHECK(vkAllocateCommandBuffers(g.device,&ai,&c));f.lists.push_back(c);
    }
    g.cmd=f.lists[f.cursor++];
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(g.cmd,&bi));g.boundPipeline=VK_NULL_HANDLE;return g.cmd;
}
void end_pass(DiagReason reason=DiagReason::Other,uint64_t image=0,uint32_t before=0,uint32_t after=0) { if(g.pass){diag_end_pass(reason,image,before,after);vkCmdEndRenderPass(g.cmd);g.pass=false;g.activePass=VK_NULL_HANDLE;g.boundPipeline=VK_NULL_HANDLE;} }
void submit(bool wait,bool final=false,VkSemaphore acquired=VK_NULL_HANDLE,VkSemaphore rendered=VK_NULL_HANDLE) {
    if(!g.cmd && !final)return;
    if(!g.cmd)command();
    end_pass(final?DiagReason::Present:DiagReason::Submit);
    VK_CHECK(diag_timed(&DiagFrame::commandEndNs,[&]{return vkEndCommandBuffer(g.cmd);}));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&g.cmd;
    VkPipelineStageFlags stage=VK_PIPELINE_STAGE_TRANSFER_BIT;
    if(acquired){si.waitSemaphoreCount=1;si.pWaitSemaphores=&acquired;si.pWaitDstStageMask=&stage;}
    if(rendered){si.signalSemaphoreCount=1;si.pSignalSemaphores=&rendered;}
    VkFence fence=VK_NULL_HANDLE;
    if(final) { fence=g.frames[g.frame].fence;VK_CHECK(vkResetFences(g.device,1,&fence));g.frames[g.frame].lastSerial=g.nextSerial; }
    else if(wait) { fence=g.syncFence;VK_CHECK(vkResetFences(g.device,1,&fence)); }
    diag_submit(wait);VK_CHECK(diag_timed(&DiagFrame::queueSubmitNs,[&]{return vkQueueSubmit(g.queue,1,&si,fence);}));uint64_t serial=g.nextSerial++;g.cmd=VK_NULL_HANDLE;++g.prof.submissions;
    if(wait){VK_CHECK(diag_timed(&DiagFrame::syncFenceWaitNs,[&]{return vkWaitForFences(g.device,1,&fence,VK_TRUE,UINT64_MAX);}));g.completedSerial=serial;collect();}
    if(final){g.frameOpen=false;g.frame=(g.frame+1)%FRAMES;++g.frameNumber;}
}
struct Allocation { Buffer* b; VkDeviceSize offset; void* cpu; };
Allocation ring(VkDeviceSize size,VkDeviceSize alignment=16) {
    begin_frame();auto& f=g.frames[g.frame];
    while(true) {
        if(f.ringCursor==f.rings.size()) {
            Ring r;r.b=make_buffer(std::max(RING_BYTES,aligned(size,alignment)),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            f.rings.push_back(r);
        }
        Ring& r=f.rings[f.ringCursor];VkDeviceSize offset=aligned(r.used,alignment);
        if(offset+size<=r.b.size){r.used=offset+size;return {&r.b,offset,static_cast<uint8_t*>(r.b.cpu)+offset};}
        ++f.ringCursor;
    }
}
Slot allocate_slot(uint32_t size) {
    uint32_t block=256;while(block<size && block<(1u<<28))block*=2;
    if(block<size)fail("oversized static buffer");
    for(auto& s:g.slabs)if(s->block==block && !s->free.empty()) {uint32_t off=s->free.back();s->free.pop_back();return {s.get(),off,block};}
    auto s=std::make_unique<Slab>();s->block=block;
    uint32_t bytes=std::max<uint32_t>(uint32_t(SLAB_BYTES),block);
    s->b=make_buffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    for(uint32_t off=bytes-block;;off-=block){s->free.push_back(off);if(!off)break;}
    Slab* p=s.get();g.slabs.push_back(std::move(s));uint32_t off=p->free.back();p->free.pop_back();return {p,off,block};
}
void release_slot(Slot s) { if(s.slab)retire([s]{s.slab->free.push_back(s.offset);}); }
VkAccessFlags access(VkImageLayout l) {
    switch(l) {
    case VK_IMAGE_LAYOUT_UNDEFINED:return 0;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:return VK_ACCESS_TRANSFER_READ_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:return VK_ACCESS_TRANSFER_WRITE_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:return VK_ACCESS_SHADER_READ_BIT;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:return VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:return 0;
    default:return VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
    }
}
VkPipelineStageFlags stages(VkImageLayout l) {
    switch(l) {
    case VK_IMAGE_LAYOUT_UNDEFINED:return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:return VK_PIPELINE_STAGE_TRANSFER_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:return VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    default:return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
}
void image_barrier(VkImage image,VkImageAspectFlags aspect,uint32_t face,uint32_t level,VkImageLayout before,VkImageLayout after,DiagReason reason=DiagReason::Other) {
    if(before==after && after!=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)return;
    end_pass(reason,diag_image(image),uint32_t(before),uint32_t(after));command();
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};b.srcAccessMask=access(before);b.dstAccessMask=access(after);b.oldLayout=before;b.newLayout=after;
    b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;b.image=image;b.subresourceRange={aspect,level,1,face,1};
    vkCmdPipelineBarrier(g.cmd,stages(before),stages(after),0,0,nullptr,0,nullptr,1,&b);
}
uint32_t subresource(const GfxTex* t,uint32_t face,uint32_t level) {
    if(!t || face>=t->faces || level>=t->levels)fail("texture subresource outside bounds");return face*t->levels+level;
}
void transition(GfxTex* t,uint32_t face,uint32_t level,VkImageLayout l,DiagReason reason=DiagReason::Other) {
    uint32_t i=subresource(t,face,level);image_barrier(t->image,t->aspect,face,level,t->layouts[i],l,reason);t->layouts[i]=l;
}
void sampled(GfxTex* t) {
    if(t==g.rt)fail("render-target sampling feedback is not admitted");
    for(uint32_t face=0;face<t->faces;++face)for(uint32_t level=0;level<t->levels;++level)transition(t,face,level,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,DiagReason::Sampling);
}
VkFormat depth_format(uint32_t fmt) {
    VkFormat candidates[3]={fmt==80?VK_FORMAT_D16_UNORM:VK_FORMAT_D24_UNORM_S8_UINT,VK_FORMAT_D32_SFLOAT_S8_UINT,VK_FORMAT_D16_UNORM};
    for(auto f:candidates) {
        if(fmt!=80 && f==VK_FORMAT_D16_UNORM)continue;
        VkFormatProperties p;vkGetPhysicalDeviceFormatProperties(g.physical,f,&p);
        if(p.optimalTilingFeatures&VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)return f;
    }
    fail("native driver has no compatible depth format");
}
VkImageView image_view(GfxTex* t,VkImageViewType type,uint32_t face,uint32_t nfaces,uint32_t level,uint32_t levels,bool sample) {
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};ci.image=t->image;ci.viewType=type;ci.format=t->native;
    ci.subresourceRange={t->aspect,level,levels,face,nfaces};
    if(sample && t->use==GFX_USE_DEPTH)ci.subresourceRange.aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT;
    if(sample && t->x8)ci.components.a=VK_COMPONENT_SWIZZLE_ONE;
    VkImageView v;VK_CHECK(vkCreateImageView(g.device,&ci,nullptr,&v));return v;
}
VkImageView attachment(GfxTex* t,uint32_t face,uint32_t level) {
    uint32_t i=subresource(t,face,level);
    if(!t->attachments[i])t->attachments[i]=image_view(t,VK_IMAGE_VIEW_TYPE_2D,face,1,level,1,false);
    return t->attachments[i];
}
GfxTex* texture_create(int type,uint32_t fmt,uint32_t w,uint32_t h,uint32_t levels,int use) {
    if(!g.device)return nullptr;
    if(!w || !h || !levels || levels>32 || (type==GFX_TEX_CUBE && w!=h))fail("invalid texture dimensions");
    uint32_t maxLevels=1;for(uint32_t n=std::max(w,h);n>1;n>>=1)++maxLevels;
    if(levels>maxLevels)fail("texture has more mip levels than dimensions allow");
    auto t=std::make_unique<GfxTex>();t->fmt=fmt;t->w=w;t->h=h;t->levels=levels;t->faces=type==GFX_TEX_CUBE?6:1;t->type=type;t->use=use;
    t->x8=fmt==22 || fmt==24;t->compressed=block_bytes(fmt)!=0;
    if(use==GFX_USE_DEPTH) {
        if(fmt!=75 && fmt!=77 && fmt!=80)fail("unknown D3D depth format");
        t->native=depth_format(fmt);t->aspect=VK_IMAGE_ASPECT_DEPTH_BIT;
        if(t->native==VK_FORMAT_D24_UNORM_S8_UINT || t->native==VK_FORMAT_D32_SFLOAT_S8_UINT)t->aspect|=VK_IMAGE_ASPECT_STENCIL_BIT;
        t->gpuBpp=t->native==VK_FORMAT_D16_UNORM?2:4;
    } else if(fmt==60) { t->native=VK_FORMAT_R8G8_SNORM;t->gpuBpp=2; }
    else {
        if(fmt!=21 && fmt!=22 && !(fmt>=23 && fmt<=26) && fmt!=28 && fmt!=50 && fmt!=51 && !t->compressed)fail("unknown D3D color format " + std::to_string(fmt));
        if(t->compressed && use!=GFX_USE_SAMPLE)fail("compressed render targets are unsupported in D3D8");
        t->native=VK_FORMAT_B8G8R8A8_UNORM;t->gpuBpp=4;
    }
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};ci.flags=type==GFX_TEX_CUBE?VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT:0;ci.imageType=VK_IMAGE_TYPE_2D;ci.format=t->native;
    ci.extent={w,h,1};ci.mipLevels=levels;ci.arrayLayers=t->faces;ci.samples=VK_SAMPLE_COUNT_1_BIT;ci.tiling=VK_IMAGE_TILING_OPTIMAL;
    ci.usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.usage|=use==GFX_USE_DEPTH?VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT:VK_IMAGE_USAGE_SAMPLED_BIT;
    if(use==GFX_USE_RT)ci.usage|=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;ci.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(g.device,&ci,nullptr,&t->image));
    VkMemoryRequirements req;vkGetImageMemoryRequirements(g.device,t->image,&req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=memory_type(req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkResult r=vkAllocateMemory(g.device,&ai,nullptr,&t->memory);
    if(r!=VK_SUCCESS){vkDestroyImage(g.device,t->image,nullptr);check(r,"vkAllocateMemory(image)");}
    r=vkBindImageMemory(g.device,t->image,t->memory,0);
    if(r!=VK_SUCCESS){vkDestroyImage(g.device,t->image,nullptr);vkFreeMemory(g.device,t->memory,nullptr);check(r,"vkBindImageMemory");}
    size_t count=size_t(t->faces)*levels;t->layouts.assign(count,VK_IMAGE_LAYOUT_UNDEFINED);t->attachments.assign(count,VK_NULL_HANDLE);t->shadow.resize(count);t->shadowValid.assign(count,false);
    for(uint32_t face=0;face<t->faces;++face)for(uint32_t level=0;level<levels;++level) {
        uint32_t mw=mip_size(w,level),mh=mip_size(h,level),block=block_bytes(fmt);
        size_t bytes=block?size_t((mw+3)/4)*((mh+3)/4)*block:size_t(mw)*mh*d3d_bpp(fmt);
        t->shadow[face*levels+level].resize(bytes);
    }
    if(use!=GFX_USE_DEPTH)t->sample=image_view(t.get(),type==GFX_TEX_CUBE?VK_IMAGE_VIEW_TYPE_CUBE:VK_IMAGE_VIEW_TYPE_2D,0,t->faces,0,levels,true);
    // D3D leaves newly allocated contents undefined; deterministic initialization
    // prevents accidental sampling of stale driver memory before the first upload.
    for(uint32_t face=0;face<t->faces;++face)for(uint32_t level=0;level<levels;++level)transition(t.get(),face,level,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,DiagReason::Initialize);
    VkImageSubresourceRange range{t->aspect,0,levels,0,t->faces};
    if(use==GFX_USE_DEPTH){VkClearDepthStencilValue v{1.0f,0};vkCmdClearDepthStencilImage(g.cmd,t->image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&v,1,&range);}
    else {VkClearColorValue v{};v.float32[3]=t->x8?1.0f:0.0f;vkCmdClearColorImage(g.cmd,t->image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&v,1,&range);}
    return t.release();
}
void texture_destroy(GfxTex* t) {
    if(!t)return;end_pass(DiagReason::Destroy,diag_image(t->image));
    // A synchronous mid-frame submission may release these views before the
    // descriptor pools reset. Do not let recycled numeric handles match stale
    // cached descriptor contents; existing in-flight sets remain untouched.
    for(auto& frame:g.frames)frame.sets.clear();
    if(g.rt==t)g.rt=nullptr;if(g.ds==t)g.ds=nullptr;
    for(auto i=g.framebuffers.begin();i!=g.framebuffers.end();) {
        if(i->second.color==t->image || i->second.depth==t->image) {VkFramebuffer fb=i->second.fb;retire([fb]{vkDestroyFramebuffer(g.device,fb,nullptr);});i=g.framebuffers.erase(i);}else ++i;
    }
    auto attachments=std::move(t->attachments);VkImageView sample=t->sample;VkImage image=t->image;VkDeviceMemory memory=t->memory;
    retire([attachments=std::move(attachments),sample,image,memory]{for(auto v:attachments)if(v)vkDestroyImageView(g.device,v,nullptr);if(sample)vkDestroyImageView(g.device,sample,nullptr);vkDestroyImage(g.device,image,nullptr);vkFreeMemory(g.device,memory,nullptr);});
    for(auto& r:t->readbacks)if(r.b.buffer){Buffer b=r.b;retire([b]{destroy_buffer(b);});}
#if defined(FFXI_ANDROID_VULKAN)
    for(auto& history:t->keyed)for(auto& slot:history->slots)
        if(slot.transfer.b.buffer){Buffer b=slot.transfer.b;retire([b]{destroy_buffer(b);});}
#endif
    delete t;
}
uint32_t expand5(uint32_t v){return (v<<3)|(v>>2);} uint32_t expand6(uint32_t v){return (v<<2)|(v>>4);}
uint32_t color565(uint16_t p){return 0xff000000u|expand5(p>>11)<<16|expand6((p>>5)&63)<<8|expand5(p&31);}
void decode_block(const uint8_t* data,uint32_t fmt,uint32_t pixels[16]) {
    const uint8_t* c=data+(fmt==DXT1?0:8);uint16_t c0=uint16_t(c[0])|uint16_t(c[1])<<8,c1=uint16_t(c[2])|uint16_t(c[3])<<8;
    uint32_t palette[4]={color565(c0),color565(c1),0,0};
    bool four=fmt!=DXT1 || c0>c1;
    for(unsigned shift=0;shift<24;shift+=8) {
        unsigned a=(palette[0]>>shift)&255,b=(palette[1]>>shift)&255;
        palette[2]|=(four?(2*a+b)/3:(a+b)/2)<<shift;
        if(four)palette[3]|=((a+2*b)/3)<<shift;
    }
    palette[2]|=0xff000000u;if(four)palette[3]|=0xff000000u;
    uint32_t indices=uint32_t(c[4])|uint32_t(c[5])<<8|uint32_t(c[6])<<16|uint32_t(c[7])<<24;
    uint8_t alpha[8]{};uint64_t abits=0;
    if(fmt==DXT4 || fmt==DXT5) {
        alpha[0]=data[0];alpha[1]=data[1];
        if(alpha[0]>alpha[1])for(unsigned i=2;i<8;++i)alpha[i]=uint8_t(((8-i)*alpha[0]+(i-1)*alpha[1])/7);
        else {for(unsigned i=2;i<6;++i)alpha[i]=uint8_t(((6-i)*alpha[0]+(i-1)*alpha[1])/5);alpha[6]=0;alpha[7]=255;}
        for(unsigned i=0;i<6;++i)abits|=uint64_t(data[2+i])<<(8*i);
    }
    for(unsigned i=0;i<16;++i) {
        pixels[i]=palette[(indices>>(2*i))&3];
        if(fmt==DXT2 || fmt==DXT3)pixels[i]=(pixels[i]&0xffffffu)|uint32_t(((data[i/2]>>((i%2)*4))&15)*17)<<24;
        else if(fmt==DXT4 || fmt==DXT5)pixels[i]=(pixels[i]&0xffffffu)|uint32_t(alpha[(abits>>(3*i))&7])<<24;
    }
}
uint32_t unpack_color(uint32_t fmt,const uint8_t* p) {
    uint32_t v=p[0];if(d3d_bpp(fmt)>=2)v|=uint32_t(p[1])<<8;
    switch(fmt) {
    case 21:case 22:std::memcpy(&v,p,4);return fmt==22?v|0xff000000u:v;
    case 23:return color565(uint16_t(v));
    case 24:case 25:return (fmt==24 || (v&0x8000)?0xff000000u:0)|expand5((v>>10)&31)<<16|expand5((v>>5)&31)<<8|expand5(v&31);
    case 26:return ((v>>12)&15)*17u<<24|((v>>8)&15)*17u<<16|((v>>4)&15)*17u<<8|(v&15)*17u;
    case 28:return v<<24; // D3DFMT_A8 samples zero RGB, as native D3D12 A8.
    case 50:return 0xff000000u|v<<16|v<<8|v;
    case 51:return (v&0xff00)<<16|(v&255)<<16|(v&255)<<8|(v&255);
    default:fail("unpack format not admitted");
    }
}
void encode_row(uint32_t fmt,const uint8_t* src,uint8_t* dst,uint32_t w) {
    for(uint32_t x=0;x<w;++x) {
        uint32_t c;std::memcpy(&c,src+x*4,4);uint32_t b=c&255,green=(c>>8)&255,r=(c>>16)&255,a=c>>24,v=0;
        switch(fmt) {
        case 21:std::memcpy(dst+x*4,&c,4);continue;
        case 22:c|=0xff000000u;std::memcpy(dst+x*4,&c,4);continue;
        case 23:v=(r>>3)<<11|(green>>2)<<5|(b>>3);break;
        case 24:v=0x8000|(r>>3)<<10|(green>>3)<<5|(b>>3);break;
        case 25:v=(a>>7)<<15|(r>>3)<<10|(green>>3)<<5|(b>>3);break;
        case 26:v=(a>>4)<<12|(r>>4)<<8|(green>>4)<<4|(b>>4);break;
        case 28:dst[x]=uint8_t(a);continue;
        case 50:dst[x]=uint8_t(r);continue;
        case 51:v=a<<8|r;break;
        default:fail("readback format not admitted");
        }
        dst[x*2]=uint8_t(v);dst[x*2+1]=uint8_t(v>>8);
    }
}
void texture_upload(GfxTex* t,uint32_t face,uint32_t level,uint32_t x,uint32_t y,uint32_t w,uint32_t h,const void* source,uint32_t pitch) {
    if(!t || !source)return;uint32_t i=subresource(t,face,level),mw=mip_size(t->w,level),mh=mip_size(t->h,level);
    if(!w || !h)return;
    if(x>mw || y>mh || w>mw-x || h>mh-y || t->use==GFX_USE_DEPTH)fail("invalid color texture upload rectangle");
    uint32_t block=block_bytes(t->fmt),bytes=block?((w+3)/4)*block:w*d3d_bpp(t->fmt),rows=block?(h+3)/4:h;
    if(pitch<bytes || (block && ((x|y)&3)))fail("texture upload pitch/block alignment");
    uint32_t targetPitch=block?((mw+3)/4)*block:mw*d3d_bpp(t->fmt);
    uint32_t start=block?(y/4)*targetPitch+(x/4)*block:y*targetPitch+x*d3d_bpp(t->fmt);
    const auto* src=static_cast<const uint8_t*>(source);
    for(uint32_t row=0;row<rows;++row)std::memcpy(t->shadow[i].data()+start+size_t(row)*targetPitch,src+size_t(row)*pitch,bytes);
    if(x==0 && y==0 && w==mw && h==mh)t->shadowValid[i]=true;
    Allocation upload=ring(size_t(w)*h*t->gpuBpp,16);
    if(block) {
        for(uint32_t by=0;by<(h+3)/4;++by)for(uint32_t bx=0;bx<(w+3)/4;++bx) {
            uint32_t colors[16];decode_block(src+size_t(by)*pitch+bx*block,t->fmt,colors);
            for(uint32_t yy=0;yy<4 && by*4+yy<h;++yy)for(uint32_t xx=0;xx<4 && bx*4+xx<w;++xx)
                std::memcpy(static_cast<uint8_t*>(upload.cpu)+(size_t(by*4+yy)*w+bx*4+xx)*4,&colors[yy*4+xx],4);
        }
    } else if(t->fmt==60)for(uint32_t row=0;row<h;++row)std::memcpy(static_cast<uint8_t*>(upload.cpu)+size_t(row)*w*2,src+size_t(row)*pitch,w*2);
    else for(uint32_t row=0;row<h;++row)for(uint32_t col=0;col<w;++col) {
        uint32_t c=unpack_color(t->fmt,src+size_t(row)*pitch+col*d3d_bpp(t->fmt));std::memcpy(static_cast<uint8_t*>(upload.cpu)+(size_t(row)*w+col)*4,&c,4);
    }
    transition(t,face,level,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,DiagReason::Upload);
    VkBufferImageCopy copy{};copy.bufferOffset=upload.offset;copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,level,face,1};copy.imageOffset={int32_t(x),int32_t(y),0};copy.imageExtent={w,h,1};
    vkCmdCopyBufferToImage(g.cmd,upload.b->buffer,t->image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);g.prof.bytes+=size_t(w)*h*t->gpuBpp;
}
VkRenderPass render_pass(GfxTex* color,GfxTex* depth) {
    RenderKey k{color->native,depth?depth->native:VK_FORMAT_UNDEFINED};std::string key=bytes_key(&k,sizeof k);
    auto found=g.renderPasses.find(key);if(found!=g.renderPasses.end())return found->second;
    VkAttachmentDescription a[2]{};a[0].format=k.color;a[0].samples=VK_SAMPLE_COUNT_1_BIT;a[0].loadOp=VK_ATTACHMENT_LOAD_OP_LOAD;a[0].storeOp=VK_ATTACHMENT_STORE_OP_STORE;
    a[0].stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;a[0].stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;a[0].initialLayout=a[0].finalLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if(depth){a[1].format=k.depth;a[1].samples=VK_SAMPLE_COUNT_1_BIT;a[1].loadOp=VK_ATTACHMENT_LOAD_OP_LOAD;a[1].storeOp=VK_ATTACHMENT_STORE_OP_STORE;
        a[1].stencilLoadOp=VK_ATTACHMENT_LOAD_OP_LOAD;a[1].stencilStoreOp=VK_ATTACHMENT_STORE_OP_STORE;a[1].initialLayout=a[1].finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;}
    VkAttachmentReference colorRef{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},depthRef{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sub.colorAttachmentCount=1;sub.pColorAttachments=&colorRef;if(depth)sub.pDepthStencilAttachment=&depthRef;
    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass=VK_SUBPASS_EXTERNAL;deps[0].dstSubpass=0;deps[0].srcStageMask=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    deps[0].dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;deps[0].dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass=0;deps[1].dstSubpass=VK_SUBPASS_EXTERNAL;deps[1].srcStageMask=deps[0].dstStageMask;deps[1].dstStageMask=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    deps[1].srcAccessMask=deps[0].dstAccessMask;deps[1].dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};ci.attachmentCount=depth?2:1;ci.pAttachments=a;ci.subpassCount=1;ci.pSubpasses=&sub;ci.dependencyCount=2;ci.pDependencies=deps;
    VkRenderPass rp;VK_CHECK(vkCreateRenderPass(g.device,&ci,nullptr,&rp));g.renderPasses.emplace(key,rp);return rp;
}
bool begin_pass() {
    if(!g.rt)return false;if(g.pass)return true;
    uint32_t w=mip_size(g.rt->w,g.rtLevel),h=mip_size(g.rt->h,g.rtLevel);
    if(g.ds && (g.ds->w<w || g.ds->h<h))fail("depth attachment smaller than color target");
    // Vulkan allows larger attachments than framebuffer dimensions; retain the
    // real bound depth surface, including contents, without a scratch substitute.
    transition(g.rt,g.rtFace,g.rtLevel,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,DiagReason::Attachment);
    if(g.ds)transition(g.ds,0,0,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,DiagReason::Attachment);
    VkRenderPass rp=render_pass(g.rt,g.ds);
    struct FbKey { VkImageView color,depth;VkRenderPass pass;uint32_t w,h; } k{};
    k.color=attachment(g.rt,g.rtFace,g.rtLevel);k.depth=g.ds?attachment(g.ds,0,0):VK_NULL_HANDLE;k.pass=rp;k.w=w;k.h=h;
    std::string key=bytes_key(&k,sizeof k);auto f=g.framebuffers.find(key);
    if(f==g.framebuffers.end()) {
        VkImageView views[2]={k.color,k.depth};VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};ci.renderPass=rp;ci.attachmentCount=g.ds?2:1;ci.pAttachments=views;ci.width=w;ci.height=h;ci.layers=1;
        VkFramebuffer fb;VK_CHECK(vkCreateFramebuffer(g.device,&ci,nullptr,&fb));f=g.framebuffers.emplace(key,Framebuffer{fb,g.rt->image,g.ds?g.ds->image:VK_NULL_HANDLE}).first;
    }
    VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};bi.renderPass=rp;bi.framebuffer=f->second.fb;bi.renderArea={{0,0},{w,h}};
    vkCmdBeginRenderPass(command(),&bi,VK_SUBPASS_CONTENTS_INLINE);g.pass=true;g.activePass=rp;++g.prof.passes;diag_begin_pass();return true;
}
VkCompareOp compare(uint32_t v) { return v>=1 && v<=8?VkCompareOp(v-1):VK_COMPARE_OP_ALWAYS; }
VkStencilOp stencil_op(uint32_t v) {
    switch(v){case 2:return VK_STENCIL_OP_ZERO;case 3:return VK_STENCIL_OP_REPLACE;case 4:return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case 5:return VK_STENCIL_OP_DECREMENT_AND_CLAMP;case 6:return VK_STENCIL_OP_INVERT;case 7:return VK_STENCIL_OP_INCREMENT_AND_WRAP;case 8:return VK_STENCIL_OP_DECREMENT_AND_WRAP;default:return VK_STENCIL_OP_KEEP;}
}
VkBlendFactor blend_factor(uint32_t v,bool x8,bool alpha) {
    switch(v) {
    case 1:return VK_BLEND_FACTOR_ZERO;case 2:return VK_BLEND_FACTOR_ONE;
    case 3:return alpha?VK_BLEND_FACTOR_SRC_ALPHA:VK_BLEND_FACTOR_SRC_COLOR;
    case 4:return alpha?VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 5:return VK_BLEND_FACTOR_SRC_ALPHA;case 6:return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 7:return x8?VK_BLEND_FACTOR_ONE:VK_BLEND_FACTOR_DST_ALPHA;
    case 8:return x8?VK_BLEND_FACTOR_ZERO:VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 9:return alpha?VK_BLEND_FACTOR_DST_ALPHA:VK_BLEND_FACTOR_DST_COLOR;
    case 10:return alpha?VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 11:return alpha?VK_BLEND_FACTOR_ONE:VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    default:return VK_BLEND_FACTOR_ONE;
    }
}
VkBlendOp blend_op(uint32_t v) {return v>=1 && v<=5?VkBlendOp(v-1):VK_BLEND_OP_ADD;}
VkPrimitiveTopology topology(uint32_t p) {
    switch(p){case GFX_POINTLIST:return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;case GFX_LINELIST:return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;case GFX_LINESTRIP:return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case GFX_TRIANGLELIST:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;case GFX_TRIANGLESTRIP:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;case GFX_TRIANGLEFAN:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;default:fail("unknown primitive topology");}
}
uint32_t vertex_count(uint32_t p,uint32_t count) {
    uint64_t n=p==GFX_POINTLIST?count:p==GFX_LINELIST?uint64_t(count)*2:p==GFX_LINESTRIP?uint64_t(count)+1:p==GFX_TRIANGLELIST?uint64_t(count)*3:uint64_t(count)+2;
    if(n>UINT32_MAX)fail("primitive count overflow");return uint32_t(n);
}
VkShaderModule shader_module(const char* src,const char* entry,bool vertex) {
    uint32_t* words=nullptr;size_t count=0;
    if(!gfx_spirv_compile(src,entry,vertex,&words,&count))fail(std::string("shader compile ")+entry);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};ci.codeSize=count*4;ci.pCode=words;
    VkShaderModule module;VkResult r=vkCreateShaderModule(g.device,&ci,nullptr,&module);std::free(words);check(r,"vkCreateShaderModule");return module;
}
ShaderPair shaders(const GfxDraw* d) {
    LibKey k{d->vs,d->fs};std::string key=bytes_key(&k,sizeof k);auto found=g.shaders.find(key);if(found!=g.shaders.end())return found->second;
    char* source=gfx_hlsl_generate(&k.vs,&k.fs,d->vs_tokens,d->ps_tokens);if(!source)fail("D3D shader translation rejected");
    ShaderPair pair;
    try {pair.vs=shader_module(source,"vs_main",true);pair.fs=shader_module(source,"fs_main",false);}
    catch(...){std::free(source);if(pair.vs)vkDestroyShaderModule(g.device,pair.vs,nullptr);throw;}
    std::free(source);g.shaders.emplace(key,pair);return pair;
}
VkPipeline pipeline(const GfxDraw* d) {
    PipeKey k{};k.lib={d->vs,d->fs};k.pipe=d->pipe;k.depth=d->depth;k.color=g.rt->native;k.z=g.ds?g.ds->native:VK_FORMAT_UNDEFINED;
    k.zbias=d->zbias;k.cull=d->cull;k.fill=d->fill;k.prim=uint8_t(d->prim);k.x8=g.rt->x8;
    std::string key=bytes_key(&k,sizeof k);auto found=g.pipelines.find(key);if(found!=g.pipelines.end())return found->second;
    ShaderPair pair=shaders(d);
    VkPipelineShaderStageCreateInfo stagesInfo[2]{};
    stagesInfo[0].sType=stagesInfo[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stagesInfo[0].stage=VK_SHADER_STAGE_VERTEX_BIT;stagesInfo[0].module=pair.vs;stagesInfo[0].pName="vs_main";
    stagesInfo[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT;stagesInfo[1].module=pair.fs;stagesInfo[1].pName="fs_main";
    VkPipelineVertexInputStateCreateInfo vertex{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};input.topology=topology(d->prim);
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};viewport.viewportCount=1;viewport.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};raster.lineWidth=1.0f;
    if(d->fill==1 || d->fill==2) {if(!g.features.fillModeNonSolid)fail("requested non-solid fill unsupported");raster.polygonMode=d->fill==1?VK_POLYGON_MODE_POINT:VK_POLYGON_MODE_LINE;}else raster.polygonMode=VK_POLYGON_MODE_FILL;
    raster.cullMode=d->cull==3?VK_CULL_MODE_BACK_BIT:d->cull==2?VK_CULL_MODE_FRONT_BIT:VK_CULL_MODE_NONE;
    raster.frontFace=VK_FRONT_FACE_CLOCKWISE;
    raster.depthBiasEnable=d->zbias!=0;raster.depthBiasConstantFactor=-float(d->zbias);raster.depthBiasSlopeFactor=-float(d->zbias)*0.5f;
    VkPipelineMultisampleStateCreateInfo multi{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};multi.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};depth.depthTestEnable=g.ds && d->depth.zenable;
    depth.depthWriteEnable=g.ds && d->depth.zenable && d->depth.zwrite;depth.depthCompareOp=d->depth.zenable?compare(d->depth.zfunc):VK_COMPARE_OP_ALWAYS;
    depth.stencilTestEnable=g.ds && g.ds->fmt==75 && d->depth.stencil;
    depth.front={stencil_op(d->depth.sfail),stencil_op(d->depth.spass),stencil_op(d->depth.szfail),compare(d->depth.sfunc),d->depth.sread,d->depth.swrite,0};depth.back=depth.front;
    VkPipelineColorBlendAttachmentState blend{};blend.colorWriteMask=VkColorComponentFlags(d->pipe.write_mask&15);blend.blendEnable=d->pipe.blend;
    uint32_t sf=d->pipe.src,df=d->pipe.dst;if(sf==12){sf=5;df=6;}else if(sf==13){sf=6;df=5;}
    blend.srcColorBlendFactor=blend_factor(sf,k.x8,false);blend.dstColorBlendFactor=blend_factor(df,k.x8,false);blend.colorBlendOp=blend_op(d->pipe.op);
    blend.srcAlphaBlendFactor=blend_factor(sf,k.x8,true);blend.dstAlphaBlendFactor=blend_factor(df,k.x8,true);blend.alphaBlendOp=blend_op(d->pipe.op);
    VkPipelineColorBlendStateCreateInfo colors{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};colors.attachmentCount=1;colors.pAttachments=&blend;
    VkDynamicState dynamicStates[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR,VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dynamic.dynamicStateCount=3;dynamic.pDynamicStates=dynamicStates;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};ci.stageCount=2;ci.pStages=stagesInfo;ci.pVertexInputState=&vertex;ci.pInputAssemblyState=&input;ci.pViewportState=&viewport;
    ci.pRasterizationState=&raster;ci.pMultisampleState=&multi;ci.pDepthStencilState=&depth;ci.pColorBlendState=&colors;ci.pDynamicState=&dynamic;ci.layout=g.pipelineLayout;ci.renderPass=render_pass(g.rt,g.ds);ci.subpass=0;
    VkPipeline p;VK_CHECK(vkCreateGraphicsPipelines(g.device,g.pipelineCache,1,&ci,nullptr,&p));g.pipelines.emplace(key,p);++g.prof.pipelines;return p;
}
VkSamplerAddressMode address(uint32_t a) {
    switch(a){case 2:return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;case 3:return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;case 4:return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case 5:fail("mirror-once sampling requires unadmitted native extension");default:return VK_SAMPLER_ADDRESS_MODE_REPEAT;}
}
VkSampler sampler(const GfxSampler& k) {
    std::string key=bytes_key(&k,sizeof k);auto found=g.samplers.find(key);if(found!=g.samplers.end())return found->second;
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};ci.magFilter=k.mag>=2?VK_FILTER_LINEAR:VK_FILTER_NEAREST;ci.minFilter=k.min>=2?VK_FILTER_LINEAR:VK_FILTER_NEAREST;
    ci.mipmapMode=k.mip>=2?VK_SAMPLER_MIPMAP_MODE_LINEAR:VK_SAMPLER_MIPMAP_MODE_NEAREST;ci.addressModeU=address(k.addr_u);ci.addressModeV=address(k.addr_v);ci.addressModeW=address(k.addr_w);
    ci.anisotropyEnable=g.features.samplerAnisotropy && (k.min==3 || k.mag==3) && k.max_aniso>1;ci.maxAnisotropy=ci.anisotropyEnable?std::min(float(k.max_aniso),g.props.limits.maxSamplerAnisotropy):1.0f;
    ci.minLod=k.mip?k.max_level:0.0f;ci.maxLod=k.mip?(k.lod_cap?float(k.lod_cap-1):VK_LOD_CLAMP_NONE):0.0f;
    if(ci.maxLod<ci.minLod)ci.maxLod=ci.minLod;
    if(k.border==0)ci.borderColor=VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    else if(k.border==0xff000000u)ci.borderColor=VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    else if(k.border==0xffffffffu)ci.borderColor=VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    else if(k.addr_u==4 || k.addr_v==4 || k.addr_w==4)fail("arbitrary sampler border color needs explicit native support");
    VkSampler s;VK_CHECK(vkCreateSampler(g.device,&ci,nullptr,&s));g.samplers.emplace(key,s);return s;
}
VkDescriptorPool descriptor_pool() {
    VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,512*4},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,512},{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,512*16},{VK_DESCRIPTOR_TYPE_SAMPLER,512*8}};
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};ci.maxSets=512;ci.poolSizeCount=4;ci.pPoolSizes=sizes;
    VkDescriptorPool pool;VK_CHECK(vkCreateDescriptorPool(g.device,&ci,nullptr,&pool));return pool;
}
VkDescriptorSet descriptors(const DescriptorKey& key) {
    auto& f=g.frames[g.frame];std::string bytes=bytes_key(&key,sizeof key);auto found=f.sets.find(bytes);if(found!=f.sets.end())return found->second;
    VkDescriptorSet set;
    for(;;) {
        if(f.poolCursor==f.pools.size())f.pools.push_back(descriptor_pool());
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=f.pools[f.poolCursor];ai.descriptorSetCount=1;ai.pSetLayouts=&g.descriptorLayout;
        VkResult r=vkAllocateDescriptorSets(g.device,&ai,&set);if(r==VK_ERROR_OUT_OF_POOL_MEMORY || r==VK_ERROR_FRAGMENTED_POOL){++f.poolCursor;continue;}check(r,"vkAllocateDescriptorSets");break;
    }
    VkDescriptorBufferInfo buffers[5]{};VkDescriptorImageInfo images[24]{};VkWriteDescriptorSet writes[8]{};
    for(unsigned i=0;i<4;++i){buffers[i]={key.stream[i],0,VK_WHOLE_SIZE};writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=set;writes[i].dstBinding=GFX_SPIRV_STREAM0+i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&buffers[i];}
    buffers[4]={key.uniform,0,sizeof(GfxU)};writes[4].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[4].dstSet=set;writes[4].dstBinding=GFX_SPIRV_U;writes[4].descriptorCount=1;writes[4].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;writes[4].pBufferInfo=&buffers[4];
    for(unsigned i=0;i<8;++i){images[i]={VK_NULL_HANDLE,key.two[i],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};images[8+i]={VK_NULL_HANDLE,key.cube[i],VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};images[16+i]={key.sampler[i],VK_NULL_HANDLE,VK_IMAGE_LAYOUT_UNDEFINED};}
    uint32_t bindings[3]={GFX_SPIRV_TEXTURE2D,GFX_SPIRV_TEXTURECUBE,GFX_SPIRV_SAMPLER};
    for(unsigned i=0;i<3;++i){writes[5+i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[5+i].dstSet=set;writes[5+i].dstBinding=bindings[i];writes[5+i].descriptorCount=8;writes[5+i].descriptorType=i==2?VK_DESCRIPTOR_TYPE_SAMPLER:VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;writes[5+i].pImageInfo=images+i*8;}
    vkUpdateDescriptorSets(g.device,8,writes,0,nullptr);f.sets.emplace(std::move(bytes),set);++g.prof.descriptorSets;return set;
}
void draw(const GfxDraw* d) {
    if(!g.device || !d)return;
    if(!g.rt){++g.prof.skips[GFX_SKIP_NO_TARGET];return;}
    uint64_t before=gfx_profiling?gfx_now_ns():0;
    command();GfxU u=d->u;DescriptorKey key{};
    for(unsigned s=0;s<4;++s) {
        uint64_t offset=0;key.stream[s]=g.dummyBuffer.buffer;
        if(d->buf[s]) {
            GfxBuf* b=d->buf[s];if(d->buf_off[s]>b->size)fail("static stream offset exceeds buffer");
            key.stream[s]=b->allocation.slab->b.buffer;offset=uint64_t(b->allocation.offset)+d->buf_off[s];b->used=g.nextSerial;
        } else if(d->data[s] && d->size[s]) {
            Allocation a=ring(uint64_t(d->size[s])+16,16);std::memcpy(a.cpu,d->data[s],d->size[s]);std::memset(static_cast<uint8_t*>(a.cpu)+d->size[s],0,16);
            key.stream[s]=a.b->buffer;offset=a.offset;g.prof.bytes+=d->size[s];
        }
        for(unsigned reg=0;reg<GFX_NREGS;++reg)if(d->vs.el[reg].used && d->vs.el[reg].stream==s) {
            int64_t corrected=int64_t(u.offset[reg])+int64_t(offset);if(corrected<0 || corrected>INT32_MAX)fail("vertex fetch offset overflow");u.offset[reg]=int32_t(corrected);
        }
    }
    sampled(g.dummy2);sampled(g.dummyCube);
    for(unsigned i=0;i<8;++i) {
        key.two[i]=g.dummy2->sample;key.cube[i]=g.dummyCube->sample;
        int wanted=(d->fs.prog || i<d->fs.nstages)?d->fs.st[i].tex:0;
        GfxTex* t=d->tex[i];
        if(wanted && t && t!=g.rt && ((wanted==2)==(t->type==GFX_TEX_CUBE))) {
            sampled(t);if(t->type==GFX_TEX_CUBE)key.cube[i]=t->sample;else key.two[i]=t->sample;
        }
        // Match the HLSL reference: unused stages, mismatched texture types,
        // missing resources and render-target feedback use the zero null view.
        GfxSampler fallback{};key.sampler[i]=sampler(wanted?d->samp[i]:fallback);
    }
    Allocation uniform=ring(sizeof u,std::max<VkDeviceSize>(16,g.props.limits.minUniformBufferOffsetAlignment));std::memcpy(uniform.cpu,&u,sizeof u);key.uniform=uniform.b->buffer;g.prof.bytes+=sizeof u;
    uint32_t dynamicOffset=uint32_t(uniform.offset);VkDescriptorSet set=descriptors(key);
    uint32_t n=vertex_count(d->prim,d->count);if(!n)return;
    VkBuffer index=VK_NULL_HANDLE;VkDeviceSize indexOffset=0;VkIndexType indexType=d->index_size==2?VK_INDEX_TYPE_UINT16:VK_INDEX_TYPE_UINT32;
    if(d->ibuf) {
        if(d->index_size!=2 && d->index_size!=4)fail("invalid static index width");
        auto* b=d->ibuf;if(uint64_t(d->ibuf_off)+uint64_t(n)*d->index_size>b->size)fail("static index range exceeds buffer");
        index=b->allocation.slab->b.buffer;indexOffset=uint64_t(b->allocation.offset)+d->ibuf_off;b->used=g.nextSerial;
    } else if(d->indices) {
        if(d->index_size!=2 && d->index_size!=4)fail("invalid index width");
        Allocation a=ring(uint64_t(n)*d->index_size,16);std::memcpy(a.cpu,d->indices,size_t(n)*d->index_size);index=a.b->buffer;indexOffset=a.offset;
    }
    VkPipeline p=pipeline(d);if(!begin_pass())return;
    if(g.boundPipeline!=p){vkCmdBindPipeline(g.cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,p);g.boundPipeline=p;}
    vkCmdBindDescriptorSets(g.cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,g.pipelineLayout,0,1,&set,1,&dynamicOffset);
    float zmin,zmax;std::memcpy(&zmin,&d->vp[4],4);std::memcpy(&zmax,&d->vp[5],4);
    uint32_t tw=mip_size(g.rt->w,g.rtLevel),th=mip_size(g.rt->h,g.rtLevel);
    if(!d->vp[2] || !d->vp[3])return;
    VkViewport viewport{float(d->vp[0]),float(d->vp[1])+float(d->vp[3]),float(d->vp[2]),-float(d->vp[3]),zmin,zmax};vkCmdSetViewport(g.cmd,0,1,&viewport);
    int64_t x0=0,y0=0,x1=tw,y1=th;
    if(d->scissor[2]>0){x0=std::max<int64_t>(0,d->scissor[0]);y0=std::max<int64_t>(0,d->scissor[1]);x1=std::min<int64_t>(tw,int64_t(d->scissor[0])+d->scissor[2]);y1=std::min<int64_t>(th,int64_t(d->scissor[1])+d->scissor[3]);}
    x0=std::min<int64_t>(tw,x0);y0=std::min<int64_t>(th,y0);x1=std::max(x1,x0);y1=std::max(y1,y0);
    VkRect2D scissor{{int32_t(x0),int32_t(y0)},{uint32_t(x1-x0),uint32_t(y1-y0)}};vkCmdSetScissor(g.cmd,0,1,&scissor);vkCmdSetStencilReference(g.cmd,VK_STENCIL_FACE_FRONT_AND_BACK,d->stencil_ref);
    if(index){vkCmdBindIndexBuffer(g.cmd,index,indexOffset,indexType);vkCmdDrawIndexed(g.cmd,n,1,0,0,0);}else vkCmdDraw(g.cmd,n,1,d->vertex_start,0);
    g.rt->shadowValid[subresource(g.rt,g.rtFace,g.rtLevel)]=false;
    if(g.ds && (d->depth.zwrite || d->depth.stencil))g.ds->shadowValid[0]=false;
    ++g.prof.draws;if(gfx_profiling)g.prof.encode+=gfx_now_ns()-before;
}
void clear(uint32_t nrects,const int32_t* rects,uint32_t flags,uint32_t color,float z,uint32_t stencil,const uint32_t vp[6]) {
    if(!g.device || !g.rt || !flags)return;
    if(!g.ds)flags&=1;if(!flags || !begin_pass())return;
    VkClearAttachment a[2]{};uint32_t count=0;
    if(flags&1){a[count].aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;a[count].colorAttachment=0;a[count].clearValue.color.float32[0]=float((color>>16)&255)/255;
        a[count].clearValue.color.float32[1]=float((color>>8)&255)/255;a[count].clearValue.color.float32[2]=float(color&255)/255;a[count].clearValue.color.float32[3]=g.rt->x8?1.0f:float(color>>24)/255;++count;}
    if(g.ds && (flags&6)){VkImageAspectFlags aspect=(flags&2)?VK_IMAGE_ASPECT_DEPTH_BIT:0;if((flags&4) && g.ds->fmt==75)aspect|=VK_IMAGE_ASPECT_STENCIL_BIT;
        if(aspect){a[count].aspectMask=aspect;a[count].clearValue.depthStencil={z,stencil};++count;}}
    if(!count)return;
    uint32_t w=mip_size(g.rt->w,g.rtLevel),h=mip_size(g.rt->h,g.rtLevel);
    int64_t vx0=vp[0],vy0=vp[1],vx1=vx0+vp[2],vy1=vy0+vp[3];
    std::vector<VkClearRect> r;r.reserve(nrects?nrects:1);
    for(uint32_t i=0;i<(nrects?nrects:1);++i) {
        int64_t x0=vx0,y0=vy0,x1=vx1,y1=vy1;
        if(nrects){if(!rects)fail("missing clear rectangles");x0=std::max(x0,int64_t(rects[4*i]));y0=std::max(y0,int64_t(rects[4*i+1]));x1=std::min(x1,int64_t(rects[4*i+2]));y1=std::min(y1,int64_t(rects[4*i+3]));}
        x0=std::max<int64_t>(0,x0);y0=std::max<int64_t>(0,y0);x1=std::min<int64_t>(w,x1);y1=std::min<int64_t>(h,y1);
        if(x1>x0 && y1>y0)r.push_back({{{int32_t(x0),int32_t(y0)},{uint32_t(x1-x0),uint32_t(y1-y0)}},0,1});
    }
    if(!r.empty()){diag_clear(r,a,count,color,z,stencil);vkCmdClearAttachments(g.cmd,count,a,uint32_t(r.size()),r.data());}
    if(flags&1)g.rt->shadowValid[subresource(g.rt,g.rtFace,g.rtLevel)]=false;if(g.ds && (flags&6))g.ds->shadowValid[0]=false;
}
void copy_shadow(GfxTex* t,uint32_t i,void* dst,uint32_t pitch) {
    uint32_t level=i%t->levels,w=mip_size(t->w,level),h=mip_size(t->h,level),block=block_bytes(t->fmt),rowBytes=block?((w+3)/4)*block:w*d3d_bpp(t->fmt),rows=block?(h+3)/4:h;
    if(pitch<rowBytes)fail("D3D readback pitch too small");for(uint32_t y=0;y<rows;++y)std::memcpy(static_cast<uint8_t*>(dst)+size_t(y)*pitch,t->shadow[i].data()+size_t(y)*rowBytes,rowBytes);
}
void queue_readback(GfxTex* t,uint32_t face,uint32_t level,Readback& r) {
    if(t->compressed)fail("GPU-modified compressed readback cannot be recompressed accurately");
    uint32_t w=mip_size(t->w,level),h=mip_size(t->h,level);VkDeviceSize need=size_t(w)*h*t->gpuBpp;
    if(!r.b.buffer || r.b.size<need){if(r.b.buffer){Buffer old=r.b;retire([old]{destroy_buffer(old);});}r.b=make_buffer(need,VK_BUFFER_USAGE_TRANSFER_DST_BIT);}
    transition(t,face,level,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,DiagReason::Readback);
    VkBufferImageCopy copy{};copy.imageSubresource={t->use==GFX_USE_DEPTH?VK_IMAGE_ASPECT_DEPTH_BIT:VK_IMAGE_ASPECT_COLOR_BIT,level,face,1};copy.imageExtent={w,h,1};
    vkCmdCopyImageToBuffer(g.cmd,t->image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,r.b.buffer,1,&copy);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;barrier.buffer=r.b.buffer;barrier.size=VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(g.cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&barrier,0,nullptr);r.serial=g.nextSerial;r.face=face;r.level=level;
}
void readback_bytes(GfxTex* t,uint32_t level,const Buffer& b,void* dst,uint32_t pitch) {
    uint32_t w=mip_size(t->w,level),h=mip_size(t->h,level);if(pitch<w*d3d_bpp(t->fmt))fail("readback output pitch too small");
    for(uint32_t y=0;y<h;++y) {
        const uint8_t* src=static_cast<const uint8_t*>(b.cpu)+size_t(y)*w*t->gpuBpp;uint8_t* target=static_cast<uint8_t*>(dst)+size_t(y)*pitch;
        if(t->fmt==60 || t->fmt==80)std::memcpy(target,src,w*2);
        else if(t->use==GFX_USE_DEPTH) {
            // D24X8 is the only lockable 24-bit depth form in this frontend.
            if(t->fmt==75)fail("stencil-bearing depth readback needs a separate stencil copy");
            for(uint32_t x=0;x<w;++x){uint32_t value;if(t->native==VK_FORMAT_D32_SFLOAT_S8_UINT){float f;std::memcpy(&f,src+x*4,4);value=uint32_t(std::round(std::clamp(f,0.0f,1.0f)*16777215.0f));}else std::memcpy(&value,src+x*4,4);value&=0xffffffu;std::memcpy(target+x*4,&value,4);}
        } else encode_row(t->fmt,src,target,w);
    }
}
void texture_read(GfxTex* t,uint32_t face,uint32_t level,void* dst,uint32_t pitch,bool async) {
    if(!t || !dst)return;uint32_t i=subresource(t,face,level);
    if(t->shadowValid[i]){copy_shadow(t,i,dst,pitch);return;}
    if(!async) {
        Readback r;queue_readback(t,face,level,r);submit(true);readback_bytes(t,level,r.b,dst,pitch);destroy_buffer(r.b);return;
    }
    collect();if(t->readFrame!=g.frameNumber){t->readFrame=g.frameNumber;t->readCount=0;}
    uint32_t probe=t->readCount++;if(probe>=PROBES){texture_read(t,face,level,dst,pitch,false);return;}
    Readback* newest=nullptr;
    for(auto& candidate:t->readbacks)if(candidate.b.buffer && candidate.serial && candidate.serial<=g.completedSerial && candidate.face==face && candidate.level==level && candidate.probe==probe && (!newest || candidate.serial>newest->serial))newest=&candidate;
    Readback& slot=t->readbacks[(g.frameNumber%(FRAMES+1))*PROBES+probe];
    if(slot.serial>g.completedSerial){submit(true);collect();}
    if(newest)readback_bytes(t,level,newest->b,dst,pitch);
    queue_readback(t,face,level,slot);slot.probe=probe;
    if(!newest){submit(true);readback_bytes(t,level,slot.b,dst,pitch);}
}
#if defined(FFXI_ANDROID_VULKAN)
void texture_read_keyed(GfxTex* t,uint32_t face,uint32_t level,void* dst,uint32_t pitch,uint64_t key,uint32_t maxAge) {
    if(!t || !dst)return;uint32_t i=subresource(t,face,level);++g.prof.keyedReads;
    if(!key || maxAge>64 || mip_size(t->w,level)>128 || mip_size(t->h,level)>128
        || t->use==GFX_USE_DEPTH || t->compressed) {
        ++g.prof.keyedExact;texture_read(t,face,level,dst,pitch,false);return;
    }
    collect();KeyedHistory* history=nullptr;
    for(auto& item:t->keyed)if(item->key==key && item->face==face && item->level==level){history=item.get();break;}
    if(!history) {
        // Reassign only an expired logical key whose every copy is complete.
        // Buffers are retained, but no unfinished GPU destination is reused.
        for(auto& item:t->keyed) {
            bool complete=true;for(auto& slot:item->slots)if(slot.transfer.serial>g.completedSerial)complete=false;
            if(complete && item->lastReadFrame!=UINT64_MAX && item->lastReadFrame<g.frameNumber
                && g.frameNumber-item->lastReadFrame>item->maxAge){history=item.get();break;}
        }
        if(!history && t->keyed.size()<KEYED_HISTORIES) {
            t->keyed.push_back(std::make_unique<KeyedHistory>());history=t->keyed.back().get();
        }
        if(!history){++g.prof.keyedOverflow;++g.prof.keyedExact;texture_read(t,face,level,dst,pitch,false);return;}
        history->key=key;history->face=face;history->level=level;history->lastReadFrame=UINT64_MAX;
        for(auto& slot:history->slots){slot.transfer.serial=0;slot.issuedFrame=UINT64_MAX;}
    }
    if(history->lastReadFrame==g.frameNumber){++g.prof.keyedDuplicate;++g.prof.keyedExact;texture_read(t,face,level,dst,pitch,false);return;}
    history->lastReadFrame=g.frameNumber;history->maxAge=maxAge?maxAge:16;
    // Exact age0/CPU-current reads also consume this key's request in the frame;
    // a later GPU write cannot turn a duplicate request into delayed pixels.
    if(!maxAge || t->shadowValid[i]){++g.prof.keyedExact;texture_read(t,face,level,dst,pitch,false);return;}
    KeyedReadback* newest=nullptr;KeyedReadback* freeSlot=nullptr;
    for(auto& slot:history->slots) {
        auto& r=slot.transfer;
        if(!r.serial || r.serial<=g.completedSerial) {
            if(!freeSlot || (!r.serial && freeSlot->transfer.serial)
                || (r.serial && freeSlot->transfer.serial && r.serial<freeSlot->transfer.serial))freeSlot=&slot;
        }
        if(r.b.buffer && r.serial && r.serial<=g.completedSerial && slot.issuedFrame<g.frameNumber
            && g.frameNumber-slot.issuedFrame>maxAge)++g.prof.keyedStale;
        if(r.b.buffer && r.serial && r.serial<=g.completedSerial && slot.issuedFrame<g.frameNumber
            && g.frameNumber-slot.issuedFrame<=maxAge
            && (!newest || slot.issuedFrame>newest->issuedFrame
                || (slot.issuedFrame==newest->issuedFrame && r.serial>newest->transfer.serial)))newest=&slot;
    }
    // Snapshot CPU bytes before reusing a completed buffer that may be newest.
    if(newest){++g.prof.keyedLate;readback_bytes(t,level,newest->transfer.b,dst,pitch);}
    if(!freeSlot) {
        // A busy ring never induces a wait when bounded, same-key history exists.
        // No usable history means current exact pixels, not an ordinal substitute.
        ++g.prof.keyedBusy;
        if(!newest){++g.prof.keyedExact;texture_read(t,face,level,dst,pitch,false);}
        return;
    }
    queue_readback(t,face,level,freeSlot->transfer);freeSlot->issuedFrame=g.frameNumber;++g.prof.keyedQueued;
    if(!newest){++g.prof.keyedExact;submit(true);readback_bytes(t,level,freeSlot->transfer.b,dst,pitch);}
}
#endif
void texture_copy(GfxTex* src,uint32_t sf,uint32_t sl,uint32_t sx,uint32_t sy,uint32_t w,uint32_t h,GfxTex* dst,uint32_t df,uint32_t dl,uint32_t dx,uint32_t dy) {
    if(!src || !dst || !w || !h)return;
    uint32_t si=subresource(src,sf,sl),di=subresource(dst,df,dl),sw=mip_size(src->w,sl),sh=mip_size(src->h,sl),dw=mip_size(dst->w,dl),dh=mip_size(dst->h,dl);
    if(src->fmt!=dst->fmt || src->native!=dst->native || sx>sw || sy>sh || dx>dw || dy>dh || w>sw-sx || h>sh-sy || w>dw-dx || h>dh-dy)fail("texture copy format or rectangle mismatch");
    if(src->image==dst->image && si==di) {
        // Self-copy regions may overlap; snapshot through a staging buffer rather
        // than issue Vulkan's forbidden overlapping same-image copy.
        uint32_t block=block_bytes(src->fmt),bpp=d3d_bpp(src->fmt);
        if(block && ((sx|sy|dx|dy)&3))fail("compressed self-copy block alignment");
        uint32_t pitch=block?((sw+3)/4)*block:sw*bpp,rows=block?(sh+3)/4:sh;
        std::vector<uint8_t> whole(size_t(pitch)*rows);texture_read(src,sf,sl,whole.data(),pitch,false);
        size_t offset=block?size_t(sy/4)*pitch+(sx/4)*block:size_t(sy)*pitch+sx*bpp;
        texture_upload(dst,df,dl,dx,dy,w,h,whole.data()+offset,pitch);return;
    }
    if(src->shadowValid[si]) {
        uint32_t block=block_bytes(src->fmt),bpp=d3d_bpp(src->fmt);if(block && ((sx|sy|dx|dy)&3))fail("compressed copy block alignment");
        uint32_t sp=block?((sw+3)/4)*block:sw*bpp,dp=block?((dw+3)/4)*block:dw*bpp;
        uint32_t rowBytes=block?((w+3)/4)*block:w*bpp,rows=block?(h+3)/4:h;
        size_t so=block?size_t(sy/4)*sp+(sx/4)*block:size_t(sy)*sp+sx*bpp,doff=block?size_t(dy/4)*dp+(dx/4)*block:size_t(dy)*dp+dx*bpp;
        for(uint32_t y=0;y<rows;++y)std::memcpy(dst->shadow[di].data()+doff+size_t(y)*dp,src->shadow[si].data()+so+size_t(y)*sp,rowBytes);
        if(dx==0 && dy==0 && w==dw && h==dh)dst->shadowValid[di]=true;
    }else dst->shadowValid[di]=false;
    transition(src,sf,sl,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,DiagReason::Copy);transition(dst,df,dl,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,DiagReason::Copy);
    VkImageCopy c{};c.srcSubresource={src->aspect,sl,sf,1};c.dstSubresource={dst->aspect,dl,df,1};c.srcOffset={int32_t(sx),int32_t(sy),0};c.dstOffset={int32_t(dx),int32_t(dy),0};c.extent={w,h,1};
    vkCmdCopyImage(g.cmd,src->image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,dst->image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&c);
}
void destroy_swap() {
    for(auto semaphore:g.rendered)vkDestroySemaphore(g.device,semaphore,nullptr);g.rendered.clear();g.swapImages.clear();g.swapLayouts.clear();
    if(g.swap)vkDestroySwapchainKHR(g.device,g.swap,nullptr);g.swap=VK_NULL_HANDLE;
}
bool compositor_suboptimal(const VkSurfaceCapabilitiesKHR& caps,int width,int height) {
    // Android reports SUBOPTIMAL whenever our intentionally unrotated IDENTITY
    // swapchain differs from its native transform hint. That stable compositor
    // path is usable; rebuilding it would reproduce the mismatch every frame.
    if(caps.currentTransform==VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR || !(caps.supportedTransforms&VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR))return false;
    if(!(caps.supportedUsageFlags&VK_IMAGE_USAGE_TRANSFER_DST_BIT) || !(caps.supportedCompositeAlpha&g.swapAlpha))return false;
    if(g.swapImages.size()<caps.minImageCount || (caps.maxImageCount && g.swapImages.size()>caps.maxImageCount))return false;
    VkExtent2D extent=caps.currentExtent;
    if(extent.width==UINT32_MAX){extent.width=std::clamp(uint32_t(width),caps.minImageExtent.width,caps.maxImageExtent.width);extent.height=std::clamp(uint32_t(height),caps.minImageExtent.height,caps.maxImageExtent.height);}
    if(extent.width!=g.extent.width || extent.height!=g.extent.height)return false;
    uint32_t count=0;VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(g.physical,g.surface,&count,nullptr));std::vector<VkSurfaceFormatKHR> formats(count);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(g.physical,g.surface,&count,formats.data()));
    for(auto f:formats)if((f.format==g.swapFormat || f.format==VK_FORMAT_UNDEFINED) && f.colorSpace==g.swapColorSpace)return true;
    return false;
}
void fit_swap() {
    if(!g.surface)return;
    int width=0,height=0;SDL_GetWindowSizeInPixels(g.window,&width,&height);
    if(width<=0 || height<=0)return;
    if(g.swap && g.swapReview) {
        VkSurfaceCapabilitiesKHR caps;VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g.physical,g.surface,&caps));g.swapReview=false;
        if(!compositor_suboptimal(caps,width,height))g.swapDirty=true;
        else if(!g.notedCompositorSuboptimal){std::fprintf(stderr,"[gfx-vulkan] stable compositor transform mismatch, tolerating SUBOPTIMAL: current=0x%x chosen=0x%x\n",unsigned(caps.currentTransform),unsigned(VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR));g.notedCompositorSuboptimal=true;}
    }
    if(g.swap && !g.swapDirty && g.extent.width==uint32_t(width) && g.extent.height==uint32_t(height))return;
    submit(true);VK_CHECK(diag_timed(&DiagFrame::deviceIdleWaitNs,[&]{return vkDeviceWaitIdle(g.device);}));g.completedSerial=g.nextSerial-1;collect();destroy_swap();
    VkSurfaceCapabilitiesKHR caps;VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g.physical,g.surface,&caps));
    if(!(caps.supportedUsageFlags&VK_IMAGE_USAGE_TRANSFER_DST_BIT))fail("surface cannot receive Vulkan backbuffer blits");
    uint32_t count=0;VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(g.physical,g.surface,&count,nullptr));std::vector<VkSurfaceFormatKHR> formats(count);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(g.physical,g.surface,&count,formats.data()));
    VkSurfaceFormatKHR format{VK_FORMAT_UNDEFINED,VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    for(auto f:formats)if(f.format==VK_FORMAT_B8G8R8A8_UNORM || f.format==VK_FORMAT_R8G8B8A8_UNORM || f.format==VK_FORMAT_UNDEFINED) {
        format=f;if(format.format==VK_FORMAT_UNDEFINED)format.format=VK_FORMAT_B8G8R8A8_UNORM;if(format.format==VK_FORMAT_B8G8R8A8_UNORM)break;
    }
    if(format.format==VK_FORMAT_UNDEFINED)fail("no accurate UNORM swapchain format");
    VkFormatProperties fp;vkGetPhysicalDeviceFormatProperties(g.physical,format.format,&fp);if(!(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_BLIT_DST_BIT))fail("swapchain format does not support blit destination");
    VkExtent2D extent=caps.currentExtent;
    if(extent.width==UINT32_MAX){extent.width=std::clamp(uint32_t(width),caps.minImageExtent.width,caps.maxImageExtent.width);extent.height=std::clamp(uint32_t(height),caps.minImageExtent.height,caps.maxImageExtent.height);}
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(g.physical,g.surface,&count,nullptr));std::vector<VkPresentModeKHR> modes(count);VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(g.physical,g.surface,&count,modes.data()));
    VkPresentModeKHR mode=VK_PRESENT_MODE_FIFO_KHR;
    if(!g.vsync){if(std::find(modes.begin(),modes.end(),VK_PRESENT_MODE_IMMEDIATE_KHR)!=modes.end())mode=VK_PRESENT_MODE_IMMEDIATE_KHR;else if(std::find(modes.begin(),modes.end(),VK_PRESENT_MODE_MAILBOX_KHR)!=modes.end())mode=VK_PRESENT_MODE_MAILBOX_KHR;}
    uint32_t images=std::max(3u,caps.minImageCount);if(caps.maxImageCount)images=std::min(images,caps.maxImageCount);
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};ci.surface=g.surface;ci.minImageCount=images;ci.imageFormat=format.format;ci.imageColorSpace=format.colorSpace;ci.imageExtent=extent;ci.imageArrayLayers=1;
    ci.imageUsage=VK_IMAGE_USAGE_TRANSFER_DST_BIT;ci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE;
    // The backbuffer and letterbox are in SDL window coordinates. Advertising
    // currentTransform would promise that we have already pre-rotated them.
    // Use IDENTITY so the presentation engine supplies the native-display
    // transform; reject surfaces that cannot preserve this unrotated path.
    if(!(caps.supportedTransforms&VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR))fail("surface lacks identity transform for unrotated presentation");
    ci.preTransform=VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    ci.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if(!(caps.supportedCompositeAlpha&ci.compositeAlpha)) {for(unsigned bit=0;bit<4;++bit)if(caps.supportedCompositeAlpha&(1u<<bit)){ci.compositeAlpha=VkCompositeAlphaFlagBitsKHR(1u<<bit);break;}}
    ci.presentMode=mode;ci.clipped=VK_TRUE;VK_CHECK(vkCreateSwapchainKHR(g.device,&ci,nullptr,&g.swap));g.extent=extent;g.swapFormat=format.format;g.swapColorSpace=format.colorSpace;g.swapAlpha=ci.compositeAlpha;g.swapDirty=false;g.swapReview=false;g.notedCompositorSuboptimal=false;
    VK_CHECK(vkGetSwapchainImagesKHR(g.device,g.swap,&count,nullptr));g.swapImages.resize(count);VK_CHECK(vkGetSwapchainImagesKHR(g.device,g.swap,&count,g.swapImages.data()));g.swapLayouts.assign(count,VK_IMAGE_LAYOUT_UNDEFINED);
    VkSemaphoreCreateInfo sem{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};g.rendered.resize(count);
    for(auto& s:g.rendered)VK_CHECK(vkCreateSemaphore(g.device,&sem,nullptr,&s));
    std::fprintf(stderr,"[gfx-vulkan] swapchain %ux%u, %u images, mode %u; transforms current=0x%x supported=0x%x chosen=0x%x\n",extent.width,extent.height,count,unsigned(mode),unsigned(caps.currentTransform),unsigned(caps.supportedTransforms),unsigned(ci.preTransform));
}
void init(void* window,int vsync) {
    g.window=static_cast<SDL_Window*>(window);g.vsync=vsync!=0;
    const char* prof=std::getenv("FFXI_PROFILE");gfx_profiling=prof && prof[0] && prof[0]!='0';
    std::vector<const char*> instanceExtensions;
    if(g.window) {
        if(!SDL_Vulkan_LoadLibrary(nullptr))fail(std::string("SDL Vulkan loader: ")+SDL_GetError());
        Uint32 count=0;const char*const* names=SDL_Vulkan_GetInstanceExtensions(&count);if(!names)fail(std::string("SDL Vulkan extensions: ")+SDL_GetError());instanceExtensions.assign(names,names+count);
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="XI Native Android";app.applicationVersion=1;app.pEngineName="D3D8 native Vulkan";app.engineVersion=1;app.apiVersion=VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ici.pApplicationInfo=&app;ici.enabledExtensionCount=uint32_t(instanceExtensions.size());ici.ppEnabledExtensionNames=instanceExtensions.data();
    VK_CHECK(vkCreateInstance(&ici,nullptr,&g.instance));
    if(g.window && !SDL_Vulkan_CreateSurface(g.window,g.instance,nullptr,&g.surface))fail(std::string("SDL Vulkan surface: ")+SDL_GetError());
    uint32_t n=0;VK_CHECK(vkEnumeratePhysicalDevices(g.instance,&n,nullptr));std::vector<VkPhysicalDevice> physicals(n);VK_CHECK(vkEnumeratePhysicalDevices(g.instance,&n,physicals.data()));
    for(auto physical:physicals) {
        VkPhysicalDeviceProperties props;vkGetPhysicalDeviceProperties(physical,&props);if(props.apiVersion<VK_API_VERSION_1_1)continue;
        uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,nullptr);std::vector<VkQueueFamilyProperties> queues(nq);vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,queues.data());
        for(uint32_t q=0;q<nq;++q)if(queues[q].queueFlags&VK_QUEUE_GRAPHICS_BIT) {
            VkBool32 present=VK_TRUE;if(g.surface)VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(physical,q,g.surface,&present));
            if(present){g.physical=physical;g.family=q;break;}
        }
        if(g.physical)break;
    }
    if(!g.physical)fail("no Vulkan 1.1 graphics/presentation queue");
    vkGetPhysicalDeviceProperties(g.physical,&g.props);vkGetPhysicalDeviceMemoryProperties(g.physical,&g.memory);VkPhysicalDeviceFeatures available;vkGetPhysicalDeviceFeatures(g.physical,&available);
    g.features.samplerAnisotropy=available.samplerAnisotropy;g.features.fillModeNonSolid=available.fillModeNonSolid;
    // Robust access is enabled only when genuinely exposed by the native device.
    g.features.robustBufferAccess=available.robustBufferAccess;
    float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=g.family;qi.queueCount=1;qi.pQueuePriorities=&priority;
    const char* extensions[]={VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dci.queueCreateInfoCount=1;dci.pQueueCreateInfos=&qi;dci.pEnabledFeatures=&g.features;dci.enabledExtensionCount=g.surface?1:0;dci.ppEnabledExtensionNames=extensions;
    VK_CHECK(vkCreateDevice(g.physical,&dci,nullptr,&g.device));vkGetDeviceQueue(g.device,g.family,0,&g.queue);
    VkDescriptorSetLayoutBinding bindings[8]{};
    for(unsigned i=0;i<4;++i){bindings[i].binding=GFX_SPIRV_STREAM0+i;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_VERTEX_BIT;}
    bindings[4]={GFX_SPIRV_U,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
    bindings[5]={GFX_SPIRV_TEXTURE2D,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,8,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};bindings[6]={GFX_SPIRV_TEXTURECUBE,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,8,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};bindings[7]={GFX_SPIRV_SAMPLER,VK_DESCRIPTOR_TYPE_SAMPLER,8,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};li.bindingCount=8;li.pBindings=bindings;VK_CHECK(vkCreateDescriptorSetLayout(g.device,&li,nullptr,&g.descriptorLayout));
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pi.setLayoutCount=1;pi.pSetLayouts=&g.descriptorLayout;VK_CHECK(vkCreatePipelineLayout(g.device,&pi,nullptr,&g.pipelineLayout));
    VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};VK_CHECK(vkCreatePipelineCache(g.device,&pci,nullptr,&g.pipelineCache));
    VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VK_CHECK(vkCreateFence(g.device,&fc,nullptr,&g.syncFence));fc.flags=VK_FENCE_CREATE_SIGNALED_BIT;
    VkSemaphoreCreateInfo sem{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for(auto& f:g.frames) {
        VK_CHECK(vkCreateFence(g.device,&fc,nullptr,&f.fence));VK_CHECK(vkCreateSemaphore(g.device,&sem,nullptr,&f.acquired));
        VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cp.flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;cp.queueFamilyIndex=g.family;VK_CHECK(vkCreateCommandPool(g.device,&cp,nullptr,&f.commands));
    }
    g.dummyBuffer=make_buffer(256,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);std::memset(g.dummyBuffer.cpu,0,256);
    g.dummy2=texture_create(GFX_TEX_2D,21,1,1,1,GFX_USE_SAMPLE);g.dummyCube=texture_create(GFX_TEX_CUBE,21,1,1,1,GFX_USE_SAMPLE);
    sampled(g.dummy2);sampled(g.dummyCube);submit(true);
    std::fprintf(stderr,"[gfx-vulkan] native %s, API %u.%u.%u, GfxU=%zu, no Wine/DXVK\n",g.props.deviceName,VK_VERSION_MAJOR(g.props.apiVersion),VK_VERSION_MINOR(g.props.apiVersion),VK_VERSION_PATCH(g.props.apiVersion),sizeof(GfxU));
    if(g.surface)fit_swap();
}
void profile_frame(uint64_t started) {
    uint64_t now=gfx_now_ns();diag_present_end(started,now);g.prof.present+=now-started;++g.prof.frames;if(!g.prof.since)g.prof.since=now;
    if(now-g.prof.since<2000000000ull){diag_start_next_frame(now);return;}
    double seconds=double(now-g.prof.since)*1e-9,ms=1e-6/double(g.prof.frames);
    std::fprintf(stderr,"[gfx-vulkan] %.2f fps: frame %.3f ms, front %.3f, encode %.3f, present %.3f; %.0f draws, %.0f KiB up, %.0f passes, %.0f sets, %.0f submits, %llu pipelines, failures %u\n",
        double(g.prof.frames)/seconds,seconds*1000/double(g.prof.frames),g.prof.front*ms,g.prof.encode*ms,g.prof.present*ms,
        double(g.prof.draws)/g.prof.frames,double(g.prof.bytes)/g.prof.frames/1024,double(g.prof.passes)/g.prof.frames,double(g.prof.descriptorSets)/g.prof.frames,double(g.prof.submissions)/g.prof.frames,(unsigned long long)g.prof.pipelines,failures);
#if defined(FFXI_ANDROID_VULKAN)
    std::fprintf(stderr,"[gfx-vulkan] keyed-probe: known %llu unknown %llu sync %llu; reads %llu late %llu exact %llu duplicate %llu overflow %llu busy %llu stale_slots %llu queued %llu; max_age 16, earlier-frame only\n",
        (unsigned long long)g.prof.probeKnown,(unsigned long long)g.prof.probeUnknown,(unsigned long long)g.prof.probeSync,
        (unsigned long long)g.prof.keyedReads,(unsigned long long)g.prof.keyedLate,(unsigned long long)g.prof.keyedExact,
        (unsigned long long)g.prof.keyedDuplicate,(unsigned long long)g.prof.keyedOverflow,(unsigned long long)g.prof.keyedBusy,
        (unsigned long long)g.prof.keyedStale,(unsigned long long)g.prof.keyedQueued);
#endif
    g.prof=Profile{};g.prof.since=now;diag_start_next_frame(now);
}
} // namespace

extern "C" {
#include "present_rect.h"
int gfx_profiling=0;

/* Native FFI diagnostic only; path must be an absent file in an owned writable
 * trial directory. The first following Present is only an arming boundary.
 * Subsequent1..32 full frames are retained; no GPU commands/waits are added. */
__attribute__((visibility("default"))) int FFXI_VulkanDiagnosticArm(const char* path,uint32_t frames) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!path || !path[0] || std::strlen(path)>1023 || !frames || frames>32 || g.diagnostic)return 0;
    try {
        auto d=std::make_unique<Diagnostic>();d->frames.resize(frames);
        d->file=std::fopen(path,"wx");if(!d->file)return 0;
        g.diagnostic=std::move(d);return 1;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"[gfx-vulkan] diagnostic arm rejected: %s\n",e.what());return 0;
    }
}

uint64_t gfx_now_ns(void) {return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
void gfx_prof_front(uint64_t ns) {g.prof.front+=ns;}
void gfx_prof_skip(int reason) {if(reason>=0 && reason<GFX_NSKIPS)++g.prof.skips[reason];}
void gfx_prof_shim(uint64_t ns) {(void)ns;}
int gfx_init(void* window,int vsync) {return guarded_value<int>(0,[&]{if(g.device)return 1;init(window,vsync);return 1;});}
void gfx_resize(uint32_t w,uint32_t h) {guarded([&]{g.wantWidth=w;g.wantHeight=h;g.swapDirty=true;});}
GfxBuf* gfx_buf_create(uint32_t size) {return guarded_value<GfxBuf*>(nullptr,[&]{if(!g.device || !size)return static_cast<GfxBuf*>(nullptr);auto b=std::make_unique<GfxBuf>();b->size=size;b->allocation=allocate_slot(size+16);return b.release();});}
void gfx_buf_destroy(GfxBuf* b) {guarded([&]{if(b){release_slot(b->allocation);delete b;}});}
void gfx_buf_upload(GfxBuf* b,const void* data,uint32_t size) {guarded([&]{if(!b || !data)return;if(size>b->size)fail("static upload exceeds buffer");if(b->used>g.completedSerial){release_slot(b->allocation);b->allocation=allocate_slot(b->size+16);b->used=0;}std::memcpy(static_cast<uint8_t*>(b->allocation.slab->b.cpu)+b->allocation.offset,data,size);g.prof.bytes+=size;});}
GfxTex* gfx_tex_create(int type,uint32_t fmt,uint32_t w,uint32_t h,uint32_t levels,int use) {return guarded_value<GfxTex*>(nullptr,[&]{return texture_create(type,fmt,w,h,levels,use);});}
void gfx_tex_destroy(GfxTex* t) {guarded([&]{texture_destroy(t);});}
void gfx_tex_upload(GfxTex* t,uint32_t face,uint32_t level,const void* src,uint32_t pitch) {guarded([&]{if(t)texture_upload(t,face,level,0,0,mip_size(t->w,level),mip_size(t->h,level),src,pitch);});}
void gfx_tex_upload_rect(GfxTex* t,uint32_t face,uint32_t level,uint32_t x,uint32_t y,uint32_t w,uint32_t h,const void* src,uint32_t pitch) {guarded([&]{texture_upload(t,face,level,x,y,w,h,src,pitch);});}
void gfx_tex_read(GfxTex* t,uint32_t face,uint32_t level,void* dst,uint32_t pitch) {guarded([&]{texture_read(t,face,level,dst,pitch,false);});}
void gfx_tex_read_async(GfxTex* t,uint32_t face,uint32_t level,void* dst,uint32_t pitch) {guarded([&]{texture_read(t,face,level,dst,pitch,true);});}
#if defined(FFXI_ANDROID_VULKAN)
void gfx_tex_read_async_keyed(GfxTex* t,uint32_t face,uint32_t level,void* dst,uint32_t pitch,uint64_t key,uint32_t maxAge) {guarded([&]{texture_read_keyed(t,face,level,dst,pitch,key,maxAge);});}
void gfx_android_probe_read(int known,int async) {if(known)++g.prof.probeKnown;else ++g.prof.probeUnknown;if(!known || !async)++g.prof.probeSync;}
#endif
void gfx_copy(GfxTex* src,uint32_t sf,uint32_t sl,uint32_t sx,uint32_t sy,uint32_t w,uint32_t h,GfxTex* dst,uint32_t df,uint32_t dl,uint32_t dx,uint32_t dy) {guarded([&]{texture_copy(src,sf,sl,sx,sy,w,h,dst,df,dl,dx,dy);});}
void gfx_set_targets(GfxTex* color,uint32_t face,uint32_t level,GfxTex* depth) {guarded([&]{if(color)subresource(color,face,level);if(color==g.rt && face==g.rtFace && level==g.rtLevel && depth==g.ds)return;diag_target_bind(color,depth);end_pass(DiagReason::Target,color?diag_image(color->image):0);g.rt=color;g.rtFace=face;g.rtLevel=level;g.ds=depth;});}
void gfx_clear(uint32_t nrects,const int32_t* rects,uint32_t flags,uint32_t color,float z,uint32_t stencil,const uint32_t vp[6]) {guarded([&]{clear(nrects,rects,flags,color,z,stencil,vp);});}
void gfx_draw(const GfxDraw* d) {guarded([&]{draw(d);});}
void gfx_scene_done(GfxTex* color,const GfxScene* scene) {(void)color;(void)scene;}
void gfx_fx_set(const char* key,float value) {(void)key;(void)value;}
float gfx_fx_get(const char* key) {(void)key;return 0;}
void gfx_trace_dump(const char* path) {(void)path;}
void gfx_set_focus(const float* pos) {(void)pos;}
void gfx_set_sync_pipelines(int on) {(void)on;}
uint32_t gfx_failures(void) {std::lock_guard<std::recursive_mutex> lock(mutex);return failures;}
void gfx_finish(void) {guarded([&]{if(!g.device)return;submit(true);VK_CHECK(diag_timed(&DiagFrame::deviceIdleWaitNs,[&]{return vkDeviceWaitIdle(g.device);}));g.completedSerial=g.nextSerial-1;collect();});}
void gfx_present(GfxTex* backbuffer) {
    guarded([&]{
        if(!g.device)return;uint64_t started=gfx_now_ns();diag_present_begin();g.presentThread=std::this_thread::get_id();
        if(!g.surface || !backbuffer){submit(false,true);profile_frame(started);return;}
        fit_swap();if(!g.swap){submit(false,true);profile_frame(started);return;}
        begin_frame();auto& f=g.frames[g.frame];uint32_t index=0;
        VkResult result=diag_timed(&DiagFrame::acquireNs,[&]{return vkAcquireNextImageKHR(g.device,g.swap,UINT64_MAX,f.acquired,VK_NULL_HANDLE,&index);});
        if(result==VK_ERROR_OUT_OF_DATE_KHR){g.swapDirty=true;fit_swap();result=diag_timed(&DiagFrame::acquireNs,[&]{return vkAcquireNextImageKHR(g.device,g.swap,UINT64_MAX,f.acquired,VK_NULL_HANDLE,&index);});}
        if(result!=VK_SUCCESS && result!=VK_SUBOPTIMAL_KHR)check(result,"vkAcquireNextImageKHR");if(result==VK_SUBOPTIMAL_KHR)g.swapReview=true;
        transition(backbuffer,0,0,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,DiagReason::Present);
        image_barrier(g.swapImages[index],VK_IMAGE_ASPECT_COLOR_BIT,0,0,g.swapLayouts[index],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,DiagReason::Present);g.swapLayouts[index]=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        VkClearColorValue black{};black.float32[3]=1;VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};vkCmdClearColorImage(g.cmd,g.swapImages[index],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&black,1,&range);
        image_barrier(g.swapImages[index],VK_IMAGE_ASPECT_COLOR_BIT,0,0,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,DiagReason::Present);
        GfxPresentRect rect=gfx_present_rect(g.extent.width,g.extent.height,backbuffer->w,backbuffer->h);
        VkImageBlit blit{};blit.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};blit.srcOffsets[1]={int32_t(backbuffer->w),int32_t(backbuffer->h),1};blit.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};blit.dstOffsets[0]={int32_t(rect.x),int32_t(rect.y),0};blit.dstOffsets[1]={int32_t(rect.x+rect.w),int32_t(rect.y+rect.h),1};
        VkFormatProperties props;vkGetPhysicalDeviceFormatProperties(g.physical,backbuffer->native,&props);if(!(props.optimalTilingFeatures&VK_FORMAT_FEATURE_BLIT_SRC_BIT))fail("backbuffer format cannot be presented by blit");
        VkFilter filter=(props.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)?VK_FILTER_LINEAR:VK_FILTER_NEAREST;
        vkCmdBlitImage(g.cmd,backbuffer->image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,g.swapImages[index],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&blit,filter);
        image_barrier(g.swapImages[index],VK_IMAGE_ASPECT_COLOR_BIT,0,0,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,DiagReason::Present);g.swapLayouts[index]=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkSemaphore rendered=g.rendered[index];submit(false,true,f.acquired,rendered);
        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};present.waitSemaphoreCount=1;present.pWaitSemaphores=&rendered;present.swapchainCount=1;present.pSwapchains=&g.swap;present.pImageIndices=&index;
        result=diag_timed(&DiagFrame::queuePresentNs,[&]{return vkQueuePresentKHR(g.queue,&present);});if(result==VK_ERROR_OUT_OF_DATE_KHR)g.swapDirty=true;else if(result==VK_SUBOPTIMAL_KHR)g.swapReview=true;else check(result,"vkQueuePresentKHR");
        static GfxPresentRect last{};static uint32_t lastw=0,lasth=0;
        if(std::memcmp(&last,&rect,sizeof rect) || lastw!=g.extent.width || lasth!=g.extent.height){std::fprintf(stderr,"[gfx-vulkan] present content %ux%u -> surface %ux%u rect {%u,%u,%u,%u}\n",backbuffer->w,backbuffer->h,g.extent.width,g.extent.height,rect.x,rect.y,rect.w,rect.h);last=rect;lastw=g.extent.width;lasth=g.extent.height;}
        profile_frame(started);
    });
}
} // extern C
