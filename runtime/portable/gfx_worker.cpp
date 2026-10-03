extern "C" {
#include "gfx.h"
}
#include "gfx_worker.h"
#include "gfx_worker_backend_api.h"
#include "gfx_worker_queue.h"
#include "gfx_worker_snapshot.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <time.h>
#if defined(__ANDROID__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {
using namespace gfxworker;
constexpr size_t ARENA_BYTES=32u<<20,SLOTS=2048;
struct Args {void* a=nullptr;void* b=nullptr;uint64_t q=0;uint32_t n[12]{};float f=0;};
enum Op {RESIZE=1,BUF_DESTROY,BUF_UPLOAD,TEX_DESTROY,TEX_UPLOAD,TEX_UPLOAD_RECT,COPY,TARGETS,CLEAR,DRAW,SCENE,FX_SET,TRACE,FOCUS,SYNC_PIPELINES,PROF_FRONT,PROF_SKIP,PROF_SHIM,PROBE,START,CPU_STATS};
struct Clear {uint32_t count,flags,color,stencil,vp[6];float z;};
struct State {
    std::mutex gate;std::unique_ptr<Queue> queue;
    std::unordered_map<GfxTex*,TextureInfo> textures;
    std::unordered_map<GfxBuf*,uint32_t> buffers;
    bool configured=false,diagnostic=false;
    uint64_t frames=0,drainNs=0,barrierNs=0,barriers=0,fallbacks=0,since=0;
    uint64_t workerCpuStart=0,workerCpuEnd=0,producerCpuStart=0;
};
// A leaked process-lifetime state avoids destruction-order hazards with Vulkan
// globals. Explicit Android entrypoint/atexit shutdown owns drain+join.
State& state(){static State* s=new State;return *s;}
std::atomic<uint32_t> workerErrors{0};
uint64_t now(){struct timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000ull+uint64_t(t.tv_nsec);}
uint64_t thread_cpu(){struct timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return uint64_t(t.tv_sec)*1000000000ull+uint64_t(t.tv_nsec);}
bool enabled(const char* key){const char* p=std::getenv(key);return p && p[0]=='1' && !p[1];}
void execute(void*,uint32_t op,const void* payload,size_t) {
    const auto* a=static_cast<const Args*>(payload);
    const auto* data=static_cast<const uint8_t*>(payload)+sizeof(Args);
    switch(op) {
    case RESIZE:gfx_backend_resize(a->n[0],a->n[1]);break;
    case BUF_DESTROY:gfx_backend_buf_destroy(static_cast<GfxBuf*>(a->a));break;
    case BUF_UPLOAD:gfx_backend_buf_upload(static_cast<GfxBuf*>(a->a),data,a->n[0]);break;
    case TEX_DESTROY:gfx_backend_tex_destroy(static_cast<GfxTex*>(a->a));break;
    case TEX_UPLOAD:gfx_backend_tex_upload(static_cast<GfxTex*>(a->a),a->n[0],a->n[1],data,a->n[2]);break;
    case TEX_UPLOAD_RECT:gfx_backend_tex_upload_rect(static_cast<GfxTex*>(a->a),a->n[0],a->n[1],a->n[2],a->n[3],a->n[4],a->n[5],data,a->n[6]);break;
    case COPY:gfx_backend_copy(static_cast<GfxTex*>(a->a),a->n[0],a->n[1],a->n[2],a->n[3],a->n[4],a->n[5],static_cast<GfxTex*>(a->b),a->n[6],a->n[7],a->n[8],a->n[9]);break;
    case TARGETS:gfx_backend_set_targets(static_cast<GfxTex*>(a->a),a->n[0],a->n[1],static_cast<GfxTex*>(a->b));break;
    case CLEAR:{const auto* c=static_cast<const Clear*>(payload);gfx_backend_clear(c->count,c->count?reinterpret_cast<const int32_t*>(c+1):nullptr,c->flags,c->color,c->z,c->stencil,c->vp);break;}
    case DRAW:gfx_backend_draw(static_cast<const GfxDraw*>(payload));break;
    case SCENE:gfx_backend_scene_done(static_cast<GfxTex*>(a->a),a->n[0]?reinterpret_cast<const GfxScene*>(data):nullptr);break;
    case FX_SET:gfx_backend_fx_set(reinterpret_cast<const char*>(data),a->f);break;
    case TRACE:gfx_backend_trace_dump(a->n[0]?reinterpret_cast<const char*>(data):nullptr);break;
    case FOCUS:gfx_backend_set_focus(a->n[0]?reinterpret_cast<const float*>(data):nullptr);break;
    case SYNC_PIPELINES:gfx_backend_set_sync_pipelines(int(a->n[0]));break;
    case PROF_FRONT:gfx_backend_prof_front(a->q);break;
    case PROF_SKIP:gfx_backend_prof_skip(int(a->n[0]));break;
    case PROF_SHIM:gfx_backend_prof_shim(a->q);break;
    case PROBE:gfx_backend_android_probe_read(int(a->n[0]),int(a->n[1]));break;
    case START:{
#if defined(__ANDROID__)
        long tid=syscall(SYS_gettid);
#else
        long tid=0;
#endif
        std::fprintf(stderr,"[gfx-worker] native worker start tid=%ld; exact FIFO snapshots\n",tid);
        if(a->a)*static_cast<uint64_t*>(a->a)=thread_cpu();break;
    }
    case CPU_STATS:*static_cast<uint64_t*>(a->a)=thread_cpu();break;
    default:throw std::runtime_error("unknown render-worker command");
    }
}
void drain(State& s,bool barrier=false) {
    if(!s.queue)return;uint64_t t=s.diagnostic?now():0;
    try{s.queue->drain();}catch(const std::exception& e){++workerErrors;std::fprintf(stderr,"[gfx-worker] fatal ordered execution failure: %s\n",e.what());std::abort();}
    if(s.diagnostic){uint64_t n=now()-t;if(barrier){s.barrierNs+=n;++s.barriers;}else s.drainNs+=n;}
}
template<class Fill,class Direct> void enqueue(State& s,uint32_t op,size_t bytes,Fill&& fill,Direct&& direct) {
    if(!s.queue){direct();return;}
    try {if(s.queue->put(op,bytes,std::forward<Fill>(fill)))return;}
    catch(const std::exception& e){std::fprintf(stderr,"[gfx-worker] fatal queue failure: %s\n",e.what());std::abort();}
    ++s.fallbacks;drain(s);direct();
}
template<class Direct> void args(State& s,uint32_t op,const Args& a,Direct&& direct) {
    enqueue(s,op,sizeof a,[&](void* p){new(p)Args(a);},std::forward<Direct>(direct));
}
template<class Direct> void blob(State& s,uint32_t op,const Args& a,const void* source,size_t bytes,Direct&& direct) {
    if(bytes>SIZE_MAX-sizeof a){drain(s);++s.fallbacks;direct();return;}
    enqueue(s,op,sizeof a+bytes,[&](void* p){new(p)Args(a);if(bytes)std::memcpy(static_cast<uint8_t*>(p)+sizeof a,source,bytes);},std::forward<Direct>(direct));
}
void report(State& s) {
    if(!s.diagnostic || !s.queue)return;++s.frames;uint64_t t=now();if(!s.since)s.since=t;if(t-s.since<2000000000ull)return;
    auto c=s.queue->counters();std::fprintf(stderr,"[gfx-worker] cumulative frames %llu worker_cpu_ms %.3f producer_cpu_ms %.3f drain_ms %.3f result_barriers %llu wait_ms %.3f queued %llu bytes %llu backpressure %llu wakes %llu fallback %llu\n",(unsigned long long)s.frames,double(s.workerCpuEnd-s.workerCpuStart)*1e-6,double(thread_cpu()-s.producerCpuStart)*1e-6,double(s.drainNs)*1e-6,(unsigned long long)s.barriers,double(s.barrierNs)*1e-6,(unsigned long long)c.commands,(unsigned long long)c.bytes,(unsigned long long)c.backpressure,(unsigned long long)c.wakes,(unsigned long long)s.fallbacks);s.since=t;
}
}
extern "C" {
int gfx_init(void* window,int vsync) {
    State& s=state();std::lock_guard<std::mutex> lock(s.gate);drain(s);int result=gfx_backend_init(window,vsync);
    if(result && !s.configured){s.configured=true;s.diagnostic=enabled("FFXI_RENDER_WORKER_DIAGNOSTICS");if(enabled("FFXI_RENDER_WORKER")) {
        try{s.queue=std::make_unique<Queue>(ARENA_BYTES,SLOTS,execute,nullptr);std::atexit(gfx_worker_shutdown);std::fprintf(stderr,"[gfx-worker] enabled; %zu bytes, %zu commands; Present/result drain\n",ARENA_BYTES,SLOTS);}
        catch(const std::exception& e){std::fprintf(stderr,"[gfx-worker] disabled: %s\n",e.what());}
        if(s.queue){Args a;a.a=s.diagnostic?&s.workerCpuStart:nullptr;args(s,START,a,[]{});drain(s);if(s.diagnostic)s.producerCpuStart=thread_cpu();}
    }}return result;
}
void gfx_worker_shutdown(){State& s=state();std::lock_guard<std::mutex> lock(s.gate);drain(s);if(s.queue){s.queue->stop();s.queue.reset();}s.textures.clear();s.buffers.clear();}
void gfx_resize(uint32_t w,uint32_t h){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.n[0]=w;a.n[1]=h;args(s,RESIZE,a,[&]{gfx_backend_resize(w,h);});}
GfxBuf* gfx_buf_create(uint32_t size){State& s=state();std::lock_guard<std::mutex> lock(s.gate);drain(s,true);GfxBuf* b=gfx_backend_buf_create(size);if(s.queue && b){try{s.buffers[b]=size;}catch(const std::bad_alloc&){drain(s);s.queue->stop();s.queue.reset();s.buffers.clear();s.textures.clear();std::fprintf(stderr,"[gfx-worker] metadata allocation failed; remaining calls direct\n");}}return b;}
void gfx_buf_destroy(GfxBuf* b){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.a=b;args(s,BUF_DESTROY,a,[&]{gfx_backend_buf_destroy(b);});s.buffers.erase(b);}
void gfx_buf_upload(GfxBuf* b,const void* data,uint32_t size){State& s=state();std::lock_guard<std::mutex> lock(s.gate);auto it=s.buffers.find(b);if(!s.queue || !data || it==s.buffers.end() || size>it->second){drain(s,true);gfx_backend_buf_upload(b,data,size);return;}Args a;a.a=b;a.n[0]=size;blob(s,BUF_UPLOAD,a,data,size,[&]{gfx_backend_buf_upload(b,data,size);});}
GfxTex* gfx_tex_create(int type,uint32_t fmt,uint32_t w,uint32_t h,uint32_t levels,int use){State& s=state();std::lock_guard<std::mutex> lock(s.gate);drain(s,true);GfxTex* t=gfx_backend_tex_create(type,fmt,w,h,levels,use);if(s.queue && t){try{s.textures[t]={fmt,w,h,levels,type==GFX_TEX_CUBE?6u:1u,use};}catch(const std::bad_alloc&){drain(s);s.queue->stop();s.queue.reset();s.buffers.clear();s.textures.clear();std::fprintf(stderr,"[gfx-worker] metadata allocation failed; remaining calls direct\n");}}return t;}
void gfx_tex_destroy(GfxTex* t){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.a=t;args(s,TEX_DESTROY,a,[&]{gfx_backend_tex_destroy(t);});s.textures.erase(t);}
static void upload(GfxTex* t,uint32_t face,uint32_t level,uint32_t x,uint32_t y,uint32_t w,uint32_t h,const void* data,uint32_t pitch,bool whole){
    State& s=state();std::lock_guard<std::mutex> lock(s.gate);auto direct=[&]{if(whole)gfx_backend_tex_upload(t,face,level,data,pitch);else gfx_backend_tex_upload_rect(t,face,level,x,y,w,h,data,pitch);};
    auto it=s.textures.find(t);UploadPlan p;
    if(whole && it!=s.textures.end() && level<32){w=it->second.width>>level;h=it->second.height>>level;if(!w)w=1;if(!h)h=1;}
    if(!s.queue || !data || it==s.textures.end() || !upload_plan(it->second,face,level,x,y,w,h,pitch,p)){drain(s,true);direct();return;}
    Args a;a.a=t;a.n[0]=face;a.n[1]=level;if(whole)a.n[2]=p.rowBytes;else {a.n[2]=x;a.n[3]=y;a.n[4]=w;a.n[5]=h;a.n[6]=p.rowBytes;}
    if(p.bytes>SIZE_MAX-sizeof a){drain(s);++s.fallbacks;direct();return;}
    enqueue(s,whole?TEX_UPLOAD:TEX_UPLOAD_RECT,sizeof a+p.bytes,[&](void* out){new(out)Args(a);fill_upload(static_cast<uint8_t*>(out)+sizeof a,data,pitch,p);},direct);
}
void gfx_tex_upload(GfxTex* t,uint32_t f,uint32_t l,const void* src,uint32_t pitch){upload(t,f,l,0,0,0,0,src,pitch,true);}
void gfx_tex_upload_rect(GfxTex* t,uint32_t f,uint32_t l,uint32_t x,uint32_t y,uint32_t w,uint32_t h,const void* src,uint32_t pitch){upload(t,f,l,x,y,w,h,src,pitch,false);}
void gfx_tex_read(GfxTex* t,uint32_t f,uint32_t l,void* dst,uint32_t pitch){State& s=state();std::lock_guard<std::mutex> lock(s.gate);drain(s,true);gfx_backend_tex_read(t,f,l,dst,pitch);}
void gfx_tex_read_async(GfxTex* t,uint32_t f,uint32_t l,void* dst,uint32_t pitch){State& s=state();std::lock_guard<std::mutex> lock(s.gate);drain(s,true);gfx_backend_tex_read_async(t,f,l,dst,pitch);}
void gfx_tex_read_async_keyed(GfxTex* t,uint32_t f,uint32_t l,void* dst,uint32_t pitch,uint64_t key,uint32_t age){State& s=state();std::lock_guard<std::mutex> lock(s.gate);drain(s,true);gfx_backend_tex_read_async_keyed(t,f,l,dst,pitch,key,age);}
void gfx_android_probe_read(int known,int async){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.n[0]=uint32_t(known);a.n[1]=uint32_t(async);args(s,PROBE,a,[&]{gfx_backend_android_probe_read(known,async);});}
void gfx_copy(GfxTex* src,uint32_t sf,uint32_t sl,uint32_t sx,uint32_t sy,uint32_t w,uint32_t h,GfxTex* dst,uint32_t df,uint32_t dl,uint32_t dx,uint32_t dy){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.a=src;a.b=dst;uint32_t n[]{sf,sl,sx,sy,w,h,df,dl,dx,dy};std::memcpy(a.n,n,sizeof n);args(s,COPY,a,[&]{gfx_backend_copy(src,sf,sl,sx,sy,w,h,dst,df,dl,dx,dy);});}
void gfx_set_targets(GfxTex* c,uint32_t f,uint32_t l,GfxTex* d){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.a=c;a.b=d;a.n[0]=f;a.n[1]=l;args(s,TARGETS,a,[&]{gfx_backend_set_targets(c,f,l,d);});}
void gfx_clear(uint32_t n,const int32_t* rects,uint32_t flags,uint32_t color,float z,uint32_t stencil,const uint32_t vp[6]){State& s=state();std::lock_guard<std::mutex> lock(s.gate);auto direct=[&]{gfx_backend_clear(n,rects,flags,color,z,stencil,vp);};if(!s.queue || !vp || (n && !rects) || uint64_t(n)*16>uint64_t(SIZE_MAX)-sizeof(Clear)){drain(s);direct();return;}Clear c{n,flags,color,stencil,{},z};std::memcpy(c.vp,vp,sizeof c.vp);enqueue(s,CLEAR,sizeof c+size_t(n)*16,[&](void* p){new(p)Clear(c);if(n)std::memcpy(static_cast<uint8_t*>(p)+sizeof c,rects,size_t(n)*16);},direct);}
void gfx_draw(const GfxDraw* d){State& s=state();std::lock_guard<std::mutex> lock(s.gate);if(!s.queue || !d){drain(s);gfx_backend_draw(d);return;}DrawPlan p;
    uint32_t vsWords=d->vs_token_count,psWords=d->ps_token_count;
    if(!draw_plan(*d,vsWords,psWords,p)){drain(s);++s.fallbacks;gfx_backend_draw(d);return;}enqueue(s,DRAW,p.bytes,[&](void* out){fill_draw(out,*d,p);},[&]{gfx_backend_draw(d);});}
void gfx_scene_done(GfxTex* color,const GfxScene* scene){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.a=color;a.n[0]=scene!=nullptr;blob(s,SCENE,a,scene,scene?sizeof *scene:0,[&]{gfx_backend_scene_done(color,scene);});}
void gfx_fx_set(const char* key,float v){State& s=state();std::lock_guard<std::mutex> lock(s.gate);if(!key){drain(s);gfx_backend_fx_set(key,v);return;}Args a;a.f=v;blob(s,FX_SET,a,key,std::strlen(key)+1,[&]{gfx_backend_fx_set(key,v);});}
float gfx_fx_get(const char* key){State& s=state();std::lock_guard<std::mutex> lock(s.gate);drain(s,true);return gfx_backend_fx_get(key);}
void gfx_trace_dump(const char* path){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.n[0]=path!=nullptr;blob(s,TRACE,a,path,path?std::strlen(path)+1:0,[&]{gfx_backend_trace_dump(path);});}
void gfx_set_focus(const float* pos){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.n[0]=pos!=nullptr;blob(s,FOCUS,a,pos,pos?sizeof(float)*3:0,[&]{gfx_backend_set_focus(pos);});}
void gfx_present(GfxTex* t){State& s=state();std::lock_guard<std::mutex> lock(s.gate);if(s.diagnostic && s.queue){Args a;a.a=&s.workerCpuEnd;args(s,CPU_STATS,a,[]{});}drain(s);gfx_backend_present(t);report(s);}
void gfx_finish(){State& s=state();std::lock_guard<std::mutex> lock(s.gate);drain(s,true);gfx_backend_finish();}
void gfx_set_sync_pipelines(int on){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.n[0]=uint32_t(on);args(s,SYNC_PIPELINES,a,[&]{gfx_backend_set_sync_pipelines(on);});}
uint32_t gfx_failures(){State& s=state();std::lock_guard<std::mutex> lock(s.gate);drain(s,true);return gfx_backend_failures()+workerErrors.load();}
void gfx_prof_front(uint64_t ns){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.q=ns;args(s,PROF_FRONT,a,[&]{gfx_backend_prof_front(ns);});}
void gfx_prof_skip(int reason){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.n[0]=uint32_t(reason);args(s,PROF_SKIP,a,[&]{gfx_backend_prof_skip(reason);});}
void gfx_prof_shim(uint64_t ns){State& s=state();std::lock_guard<std::mutex> lock(s.gate);Args a;a.q=ns;args(s,PROF_SHIM,a,[&]{gfx_backend_prof_shim(ns);});}
}
