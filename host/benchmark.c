/* Scoped diagnostic collector for the local hxitest performance harness.
 * MIT, like xi-on-mac. No credentials, process arguments, readbacks or per-draw
 * hooks. One monotonic timestamp and buffered numeric record per Present. */
#include "benchmark.h"
#include "plat.h"
#include "runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#if defined(FFXI_ANDROID_DIAGNOSTIC)
#include "gthread.h"
#include "xi_cpu_sample.h"
#include <unistd.h>
extern int FFXI_VulkanDiagnosticArm(const char*, uint32_t);
#endif

#if defined(__GNUC__)
#define BENCH_EXPORT __attribute__((visibility("default"), used))
#else
#define BENCH_EXPORT
#endif

static PlatFile *frames, *markers;
static uint64_t anchor_ns, anchor_ms, previous_ns, flush_ns, frame_count;
static uint32_t current_zone;
static char buffer[65536];
static uint32_t buffered;
static int active, initialized, registered;
#if defined(FFXI_ANDROID_DIAGNOSTIC)
static int diagnostic, sampler_armed, diagnostic_attempted;
static int sampler_begin_result, sampler_end_result;
static uint32_t present_tid, present_tid_mismatches;
static uint64_t present_tid_checks;
static FILE *sampler_phase;
static void diagnostic_record(const char *event)
{
    if (!sampler_phase) return;
    GThread *self=gt_self();
    fprintf(sampler_phase,"{\"event\":\"%s\",\"phase\":\"mixed-32\",\"zone\":234,\"fixture_count\":32,\"monotonic_ns\":%" PRIu64 ",\"tid\":%u,\"present_tid\":%u,\"present_tid_mismatches\":%u,\"present_tid_checks\":%" PRIu64 ",\"frame\":%" PRIu64 ",\"fcw\":%u,\"sampler_begin_result\":%d,\"sampler_end_result\":%d,\"cleanup_ok\":%d}\n",
            event,rt_monotonic_ns(),(uint32_t)gettid(),present_tid,present_tid_mismatches,present_tid_checks,frame_count,self ? self->g.fcw : 0,sampler_begin_result,sampler_end_result,xi_cpu_sample_cleanup_ok());
    fflush(sampler_phase);
}
static void diagnostic_end(const char *event)
{
    if (sampler_armed) {
        int result=xi_cpu_sample_end(); sampler_armed=0; sampler_end_result=result;
        rt_log("[android-diag] sampler end=%d cleanup=%d\n",result,xi_cpu_sample_cleanup_ok());
        if (!xi_cpu_sample_cleanup_ok()) _Exit(71);
    }
    diagnostic_record(event);
    if (sampler_phase) fclose(sampler_phase),sampler_phase=NULL;
}
static void diagnostic_begin(void)
{
    if (diagnostic_attempted) return;
    diagnostic_attempted=1;
    if (!present_tid || present_tid!=(uint32_t)gettid() || present_tid_mismatches) {
        rt_log("[android-diag] phase owner differs from Present; capture rejected\n"); return;
    }
    const char *dir=getenv("FFXI_BENCH_DIR");
    char phase_path[2048],cpu_path[2048],gpu_path[2048];
    int np=snprintf(phase_path,sizeof phase_path,"%s/sampler-phase.jsonl",dir);
    const char *cpu_dir=getenv("FFXI_NATIVE_CPU_DIR");
    const char *label=strrchr(dir,'/');
    if (!cpu_dir || !label || !label[1]) return;
    int nc=snprintf(cpu_path,sizeof cpu_path,"%s/xi-cpu-%s.jsonl",cpu_dir,label+1);
    int ng=snprintf(gpu_path,sizeof gpu_path,"%s/gpu-ledger.jsonl",dir);
    if (np<0 || np>=(int)sizeof phase_path || nc<0 || nc>=(int)sizeof cpu_path || ng<0 || ng>=(int)sizeof gpu_path) return;
    sampler_phase=fopen(phase_path,"wx"); if (!sampler_phase) return;
    diagnostic_record("phase-start");
    const char *hz=getenv("FFXI_NATIVE_CPU_HZ");
    uint32_t frequency=hz ? (uint32_t)strtoul(hz,NULL,10) : 49;
    if (frequency!=49 && frequency!=53 && frequency!=199 && frequency!=211) frequency=49;
    int result=xi_cpu_sample_begin(cpu_path,frequency);
    sampler_armed=result==1; sampler_begin_result=result;
    int gpu=FFXI_VulkanDiagnosticArm(gpu_path,12);
    rt_log("[android-diag] sampler begin=%d gpu arm=%d owner=%u\n",result,gpu,present_tid);
    if (!xi_cpu_sample_cleanup_ok()) _Exit(71);
}
#endif


static int append(PlatFile* file, const char* bytes, uint32_t count)
{
    return file && plat_file_write(file, bytes, count) == (int64_t)count;
}

static int flush_frames(void)
{
    if (!buffered)
        return 1;
    if (!append(frames, buffer, buffered))
        return 0;
    buffered = 0;
    return 1;
}

void benchmark_close(void)
{
#if defined(FFXI_ANDROID_DIAGNOSTIC)
    diagnostic_end("capture-abort");
#endif
    active = 0;
    if (frames)
    {
        flush_frames();
        plat_file_flush(frames);
        plat_file_close(frames);
        frames = NULL;
    }
    if (markers)
    {
        plat_file_flush(markers);
        plat_file_close(markers);
        markers = NULL;
    }
}

static void failed(void)
{
    rt_log("[benchmark] output failed; timestamps are incomplete and cannot be admitted\n");
    benchmark_close();
}

int benchmark_init(const char* account)
{
    if (initialized)
        return active ? 1 : 0;
    initialized = 1;
#if defined(FFXI_ANDROID_DIAGNOSTIC)
    const char *diag=getenv("FFXI_ANDROID_DIAGNOSTICS");
    diagnostic=diag && !strcmp(diag,"1");
#endif
    const char* dir = getenv("FFXI_BENCH_DIR");
    if (!dir || !*dir)
        return 0;
    if (!account || strcmp(account, "hxitest"))
    {
        rt_log("[benchmark] only the explicitly supplied local hxitest account is admitted\n");
        return -1;
    }
    char frame_path[2048], marker_path[2048];
    int nf = snprintf(frame_path, sizeof frame_path, "%s/frames.csv", dir);
    int nm = snprintf(marker_path, sizeof marker_path, "%s/markers.jsonl", dir);
    if (nf < 0 || nf >= (int)sizeof frame_path || nm < 0 || nm >= (int)sizeof marker_path)
        return -1;
    frames = plat_file_open(frame_path, PLAT_WRITE | PLAT_CREATE | PLAT_EXCL);
    if (!frames)
    {
        rt_log("[benchmark] frames.csv must be absent and its directory writable\n");
        return -1;
    }
    markers = plat_file_open(marker_path, PLAT_WRITE | PLAT_CREATE | PLAT_EXCL);
    if (!markers)
    {
        plat_file_close(frames);
        frames = NULL;
        /* This invocation exclusively created this still-empty file. */
        plat_unlink(frame_path);
        rt_log("[benchmark] markers.jsonl must be absent and its directory writable\n");
        return -1;
    }
    anchor_ns = rt_monotonic_ns();
    anchor_ms = plat_wall_ms();
    previous_ns = flush_ns = anchor_ns;
    frame_count = buffered = current_zone = 0;
    static const char header[] = "epoch,frame_ms,zone,frame,monotonic_ns\n";
    if (!append(frames, header, sizeof header - 1))
    {
        failed();
        return -1;
    }
    active = 1;
    if (!registered)
        atexit(benchmark_close), registered = 1;
    return FFXI_BenchmarkMark("collector initialized", "setup", 0, 0) ? 1 : -1;
}

int benchmark_enabled(void) { return active; }

void benchmark_frame(void)
{
    if (!active)
        return;
#if defined(FFXI_ANDROID_DIAGNOSTIC)
    if (diagnostic) {
        uint32_t tid=(uint32_t)gettid(); ++present_tid_checks;
        if (!present_tid) present_tid=tid;
        else if (present_tid!=tid) ++present_tid_mismatches;
    }
#endif
    uint64_t now = rt_monotonic_ns();
    if (now < previous_ns || now < anchor_ns)
    {
        failed();
        return;
    }
    double epoch = (double)anchor_ms / 1000.0 + (double)(now - anchor_ns) / 1e9;
    double dt = (double)(now - previous_ns) / 1e6;
    char row[192];
    int n = snprintf(row, sizeof row, "%.6f,%.6f,%u,%" PRIu64 ",%" PRIu64 "\n",
                     epoch, dt, current_zone, frame_count++, now);
    if (n < 0 || n >= (int)sizeof row ||
        (buffered + (uint32_t)n > sizeof buffer && !flush_frames()))
    {
        failed();
        return;
    }
    memcpy(buffer + buffered, row, (size_t)n);
    buffered += (uint32_t)n;
    previous_ns = now;
    if (now - flush_ns >= 1000000000ull)
    {
        if (!flush_frames())
            failed();
        flush_ns = now;
    }
}

static int safe_label(const char* s)
{
    if (!s)
        return 0;
    for (unsigned n = 0; n < 160; ++n)
    {
        unsigned char c = (unsigned char)s[n];
        if (!c)
            return 1;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == ' ' || c == '_' || c == '.' || c == '-' || c == ':'))
            return 0;
    }
    return 0;
}

BENCH_EXPORT int FFXI_BenchmarkMark(const char* label, const char* phase,
                                  uint32_t zone, uint32_t fixture_count)
{
    if (!active || !safe_label(label) || !safe_label(phase) || zone > 65535 || fixture_count > 2304)
        return 0;
#if defined(FFXI_ANDROID_DIAGNOSTIC)
    if (diagnostic && !strcmp(label,"stress phase end") && !strcmp(phase,"mixed-32") && zone==234 && fixture_count==32)
        diagnostic_end("phase-end");
#endif
    uint64_t now = rt_monotonic_ns();
    if (now < anchor_ns)
        return 0;
    current_zone = zone;
    char row[768];
    double epoch = (double)anchor_ms / 1000.0 + (double)(now - anchor_ns) / 1e9;
    int n = snprintf(row, sizeof row,
        "{\"epoch\":%.6f,\"monotonic_ns\":%" PRIu64 ",\"label\":\"%s\",\"phase\":\"%s\","
        "\"zone\":%u,\"fixture_count\":%u,\"frame\":%" PRIu64 "}\n",
        epoch, now, label, phase, zone, fixture_count, frame_count);
    if (n < 0 || n >= (int)sizeof row || !flush_frames() || !append(markers, row, (uint32_t)n))
    {
        failed();
        return 0;
    }
#if defined(FFXI_ANDROID_DIAGNOSTIC)
    if (diagnostic && !strcmp(label,"stress phase start") && !strcmp(phase,"mixed-32") && zone==234 && fixture_count==32)
        diagnostic_begin();
#endif
    return 1;
}

BENCH_EXPORT void FFXI_BenchmarkSetZone(uint32_t zone)
{
    if (zone <= 65535)
        current_zone = zone;
}
