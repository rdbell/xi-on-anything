#define _GNU_SOURCE 1
#include "xi_cpu_sample.h"
#include <stdatomic.h>
#include <stdint.h>
#include <limits.h>

#define XI_CPU_SAMPLE_CAP 32768u
#define XI_CPU_MODULE_CAP 512u
#define XI_CPU_SEGMENT_CAP 4u
#define XI_CPU_NAME_CAP 384u
_Static_assert(ATOMIC_LLONG_LOCK_FREE == 2, "64-bit sample atomics must be lock-free");
_Static_assert(ATOMIC_LONG_LOCK_FREE == 2, "ARM64 uint64_t atomics must be lock-free");
_Static_assert(ATOMIC_INT_LOCK_FREE == 2, "32-bit sample atomics must be lock-free");

/* Signal handlers may access only lock-free atomic state. Timer cookie storage
 * stays alive until process exit. There is no stack walk or guest-memory read. */
typedef struct Sample {
    _Atomic uint64_t pc, monotonic_ns;
    _Atomic uint32_t tid, overrun;
} Sample;
static Sample samples[XI_CPU_SAMPLE_CAP];
static _Atomic uint32_t active, owner_tid, sample_count, drops, bad_signal, bad_tid, clock_errors;
static _Atomic uint64_t kernel_overruns;
static int timer_cookie;

static void record_sample(uint64_t pc, uint32_t tid, uint64_t ns, uint32_t overrun)
{
    if (!atomic_load_explicit(&active, memory_order_relaxed)) return;
    if (tid != atomic_load_explicit(&owner_tid, memory_order_relaxed)) {
        atomic_fetch_add_explicit(&bad_tid, 1, memory_order_relaxed); return;
    }
    atomic_fetch_add_explicit(&kernel_overruns, overrun, memory_order_relaxed);
    uint32_t n = atomic_load_explicit(&sample_count, memory_order_relaxed);
    if (n >= XI_CPU_SAMPLE_CAP) {
        atomic_fetch_add_explicit(&drops, 1, memory_order_relaxed); return;
    }
    atomic_store_explicit(&samples[n].pc, pc, memory_order_relaxed);
    atomic_store_explicit(&samples[n].monotonic_ns, ns, memory_order_relaxed);
    atomic_store_explicit(&samples[n].tid, tid, memory_order_relaxed);
    atomic_store_explicit(&samples[n].overrun, overrun, memory_order_relaxed);
    atomic_store_explicit(&sample_count, n + 1, memory_order_release);
}

#if defined(__ANDROID__) && defined(__aarch64__)
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <link.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

typedef struct Segment { uint64_t lo, hi; } Segment;
typedef struct Module {
    uint64_t base;
    uint32_t nsegments, segment_overflow, name_truncated;
    Segment segments[XI_CPU_SEGMENT_CAP];
    char name[XI_CPU_NAME_CAP];
} Module;
typedef struct Modules { uint32_t count, overflow; Module module[XI_CPU_MODULE_CAP]; } Modules;
static Modules modules_before, modules_after;
static timer_t cpu_timer;
static struct sigaction previous_action;
static sigset_t previous_mask;
static int attempted, armed, cleanup_ok = 1, output_fd = -1;
static uint64_t start_mono, end_mono, start_cpu, end_cpu, interval_ns;
static uint32_t requested_hz;

static uint64_t read_clock(clockid_t clock)
{
    struct timespec t;
    if (clock_gettime(clock, &t)) return 0;
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

static void on_sample(int signo, siginfo_t *info, void *context)
{
    int saved_errno = errno;
    if (signo != SIGPROF || !info || info->si_code != SI_TIMER ||
        info->si_value.sival_ptr != &timer_cookie || !context) {
        atomic_fetch_add_explicit(&bad_signal, 1, memory_order_relaxed);
        /* The accepted original action is SIG_DFL. Preserve its fatal behavior
         * for any foreign SIGPROF; do not silently consume another producer. */
        struct sigaction fatal = {0}; fatal.sa_handler = SIG_DFL;
        sigemptyset(&fatal.sa_mask);
        if (sigaction(SIGPROF, &fatal, NULL) || raise(SIGPROF)) _exit(128 + SIGPROF);
        errno = saved_errno; return;
    }
    if (atomic_load_explicit(&active, memory_order_relaxed)) {
        struct timespec t;
        if (clock_gettime(CLOCK_MONOTONIC, &t)) {
            atomic_fetch_add_explicit(&clock_errors, 1, memory_order_relaxed);
        } else {
            const ucontext_t *uc = context;
            uint64_t ns = (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
            record_sample(uc->uc_mcontext.pc, (uint32_t)gettid(), ns,
                          info->si_overrun > 0 ? (uint32_t)info->si_overrun : 0);
        }
    }
    errno = saved_errno;
}

static int module_callback(struct dl_phdr_info *info, size_t size, void *data)
{
    (void)size;
    Modules *m = data;
    uint32_t executable = 0;
    for (uint32_t i = 0; i < info->dlpi_phnum; ++i)
        if (info->dlpi_phdr[i].p_type == PT_LOAD && (info->dlpi_phdr[i].p_flags & PF_X)) executable = 1;
    if (!executable) return 0;
    if (m->count == XI_CPU_MODULE_CAP) { ++m->overflow; return 0; }
    Module *d = &m->module[m->count++];
    d->base = (uint64_t)info->dlpi_addr;
    const char *name = info->dlpi_name && *info->dlpi_name ? info->dlpi_name : "<main executable>";
    size_t len = strlen(name);
    if (len >= sizeof d->name) { len = sizeof d->name - 1; d->name_truncated = 1; }
    memcpy(d->name, name, len); d->name[len] = 0;
    for (uint32_t i = 0; i < info->dlpi_phnum; ++i) {
        const ElfW(Phdr) *p = &info->dlpi_phdr[i];
        if (p->p_type != PT_LOAD || !(p->p_flags & PF_X)) continue;
        if (d->nsegments == XI_CPU_SEGMENT_CAP) { ++d->segment_overflow; continue; }
        Segment *s = &d->segments[d->nsegments++];
        s->lo = d->base + p->p_vaddr; s->hi = s->lo + p->p_memsz;
    }
    return 0;
}

static int restore_mask(void)
{
    int e = pthread_sigmask(SIG_SETMASK, &previous_mask, NULL);
    if (e) { cleanup_ok = 0; errno = e; return 0; }
    return 1;
}

static int drain_signal(const sigset_t *set, uint32_t *owned, uint32_t *foreign)
{
    struct timespec no_wait = {0,0}; siginfo_t info;
    for (;;) {
        int r = sigtimedwait(set, &info, &no_wait);
        if (r == SIGPROF) {
            if (info.si_code == SI_TIMER && info.si_value.sival_ptr == &timer_cookie) ++*owned;
            else ++*foreign;
        } else if (r < 0 && errno == EINTR) continue;
        else if (r < 0 && errno == EAGAIN) return 1;
        else { cleanup_ok = 0; errno = EIO; return 0; }
    }
}

static int restore_action_and_foreign(uint32_t foreign)
{
    if (sigaction(SIGPROF, &previous_action, NULL)) { cleanup_ok = 0; return 0; }
    /* Still blocked on the owner: put back the default-disposition signal.
     * Restoring the exact original mask then preserves blocked/fatal behavior. */
    if (foreign && raise(SIGPROF)) { cleanup_ok = 0; return 0; }
    return 1;
}

int xi_cpu_sample_cleanup_ok(void) { return cleanup_ok; }

static void quoted(FILE *f, const char *s)
{
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        if (*p == '"' || *p == '\\') { fputc('\\', f); fputc(*p, f); }
        else if (*p < 32) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
    }
    fputc('"', f);
}

static void write_modules(FILE *f, const Modules *m, const char *when)
{
    for (uint32_t i = 0; i < m->count; ++i) {
        const Module *d = &m->module[i];
        fprintf(f, "{\"event\":\"module\",\"when\":\"%s\",\"base\":\"%016" PRIx64 "\",\"name\":", when, d->base);
        quoted(f, d->name);
        fprintf(f, ",\"name_truncated\":%u,\"segment_overflow\":%u,\"executable\":[", d->name_truncated, d->segment_overflow);
        for (uint32_t k = 0; k < d->nsegments; ++k)
            fprintf(f, "%s[\"%016" PRIx64 "\",\"%016" PRIx64 "\"]", k ? "," : "", d->segments[k].lo, d->segments[k].hi);
        fputs("]}\n", f);
    }
}

int xi_cpu_sample_begin(const char *path, uint32_t hz)
{
    const char *enabled = getenv("FFXI_NATIVE_CPU_SAMPLE");
    if (!enabled || strcmp(enabled, "1")) return 0;
    if (attempted || !path || !*path || hz < 49 || hz > 499) { errno = EINVAL; return -1; }
    attempted = 1;
    sigset_t signal_set;
    sigemptyset(&signal_set); sigaddset(&signal_set, SIGPROF);
    int mask_error = pthread_sigmask(SIG_BLOCK, &signal_set, &previous_mask);
    if (mask_error) { errno = mask_error; return -1; }
    sigset_t pending;
    if (sigpending(&pending)) { int saved = errno; if (restore_mask()) errno = saved; return -1; }
    if (sigismember(&pending, SIGPROF) == 1) { if (restore_mask()) errno = EBUSY; return -1; }
    struct itimerval legacy;
    if (sigaction(SIGPROF, NULL, &previous_action) || getitimer(ITIMER_PROF, &legacy)) {
        int saved = errno; if (restore_mask()) errno = saved; return -1;
    }
    if (previous_action.sa_handler != SIG_DFL || legacy.it_value.tv_sec || legacy.it_value.tv_usec ||
        legacy.it_interval.tv_sec || legacy.it_interval.tv_usec) {
        if (restore_mask()) errno = EBUSY; return -1;
    }
    output_fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (output_fd < 0) {
        int saved = errno; if (restore_mask()) errno = saved; return -1;
    }
    /* Touch every slot before any sampling signal can run. */
    for (uint32_t i = 0; i < XI_CPU_SAMPLE_CAP; ++i) {
        atomic_store_explicit(&samples[i].pc, 0, memory_order_relaxed);
        atomic_store_explicit(&samples[i].monotonic_ns, 0, memory_order_relaxed);
        atomic_store_explicit(&samples[i].tid, 0, memory_order_relaxed);
        atomic_store_explicit(&samples[i].overrun, 0, memory_order_relaxed);
    }
    dl_iterate_phdr(module_callback, &modules_before);
    uint32_t tid = (uint32_t)gettid(); atomic_store_explicit(&owner_tid, tid, memory_order_relaxed);
    struct sigaction action; memset(&action, 0, sizeof action);
    action.sa_sigaction = on_sample; action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask); sigaddset(&action.sa_mask, SIGPROF);
    if (sigaction(SIGPROF, &action, NULL)) goto fail_file;
    struct sigevent notify; memset(&notify, 0, sizeof notify);
    notify.sigev_notify = SIGEV_THREAD_ID; notify.sigev_signo = SIGPROF;
    notify.sigev_value.sival_ptr = &timer_cookie; notify.sigev_notify_thread_id = (pid_t)tid;
    if (timer_create(CLOCK_THREAD_CPUTIME_ID, &notify, &cpu_timer)) goto fail_action;
    requested_hz = hz; interval_ns = UINT64_C(1000000000) / hz;
    struct itimerspec period; memset(&period, 0, sizeof period);
    period.it_value.tv_nsec = period.it_interval.tv_nsec = (long)interval_ns;
    start_mono = read_clock(CLOCK_MONOTONIC); start_cpu = read_clock(CLOCK_THREAD_CPUTIME_ID);
    if (!start_mono || !start_cpu) { errno = EIO; goto fail_timer; }
    atomic_store_explicit(&active, 1, memory_order_release);
    if (timer_settime(cpu_timer, 0, &period, NULL)) goto fail_timer;
    /* SDL blocks SIGPROF in new threads. This explicit opt-in changes only this
     * owner's mask during capture; the original mask is restored at end. */
    mask_error = pthread_sigmask(SIG_UNBLOCK, &signal_set, NULL);
    if (mask_error) { errno = mask_error; goto fail_timer; }
    armed = 1; return 1;
fail_timer: {
    int saved = errno; atomic_store_explicit(&active, 0, memory_order_relaxed);
    struct itimerspec zero; memset(&zero, 0, sizeof zero);
    int disarmed = timer_settime(cpu_timer, 0, &zero, NULL) == 0;
    int deleted = timer_delete(cpu_timer) == 0;
    if (!deleted) cleanup_ok = 0;
    if (!disarmed && !deleted) {
        /* Do not restore SIG_DFL while an owned source might still fire. */
        armed = 1; cleanup_ok = 0; errno = EIO; return -1;
    }
    uint32_t owned = 0, foreign = 0;
    if (!drain_signal(&signal_set, &owned, &foreign) || !restore_action_and_foreign(foreign)) {
        cleanup_ok = 0; errno = EIO; return -1;
    }
    errno = saved; goto fail_file;
}
fail_action: {
    int saved = errno;
    if (!restore_action_and_foreign(0)) { cleanup_ok = 0; errno = EIO; return -1; }
    errno = saved;
}
fail_file: {
    int saved = errno; close(output_fd); output_fd = -1; unlink(path);
    if (restore_mask()) errno = saved;
    return -1;
}
}

int xi_cpu_sample_end(void)
{
    if (!armed) { errno = EPERM; return -1; }
    if ((uint32_t)gettid() != atomic_load_explicit(&owner_tid, memory_order_relaxed)) { cleanup_ok = 0; errno = EPERM; return -1; }
    sigset_t signal_set; sigemptyset(&signal_set); sigaddset(&signal_set, SIGPROF);
    int mask_error = pthread_sigmask(SIG_BLOCK, &signal_set, NULL);
    if (mask_error) { cleanup_ok = 0; errno = mask_error; return -1; }
    atomic_store_explicit(&active, 0, memory_order_release);
    end_mono = read_clock(CLOCK_MONOTONIC); end_cpu = read_clock(CLOCK_THREAD_CPUTIME_ID);
    struct itimerspec zero; memset(&zero, 0, sizeof zero);
    int disarmed = timer_settime(cpu_timer, 0, &zero, NULL) == 0;
    int deleted = timer_delete(cpu_timer) == 0;
    if (!deleted) cleanup_ok = 0;
    if (!disarmed && !deleted) { cleanup_ok = 0; errno = EIO; return -1; } /* Keep owner blocked/handler installed. */
    /* Bionic timer_delete frees its userspace timer_t wrapper. Never let a
     * later restoration error permit another end() to use that old wrapper. */
    armed = 0;
    uint32_t drained = 0, foreign_pending = 0;
    if (!drain_signal(&signal_set, &drained, &foreign_pending) || !restore_action_and_foreign(foreign_pending)) {
        cleanup_ok = 0; errno = EIO; return -1;
    }
    if (!restore_mask()) return -1;
    armed = 0;
    dl_iterate_phdr(module_callback, &modules_after);
    FILE *f = fdopen(output_fd, "w");
    if (!f) { int saved = errno; close(output_fd); output_fd = -1; errno = saved; return -1; }
    output_fd = -1;
    uint32_t count = atomic_load_explicit(&sample_count, memory_order_acquire);
    uint32_t badsig = atomic_load_explicit(&bad_signal, memory_order_relaxed), badthread = atomic_load_explicit(&bad_tid, memory_order_relaxed);
    uint32_t badclock = atomic_load_explicit(&clock_errors, memory_order_relaxed), dropped = atomic_load_explicit(&drops, memory_order_relaxed);
    int clean = cleanup_ok && deleted && !foreign_pending && !badsig && !badthread && !badclock && !dropped && end_mono >= start_mono && end_cpu >= start_cpu;
    fprintf(f, "{\"event\":\"native-cpu-sampler\",\"complete\":true,\"clean\":%s,\"owner_tid\":%u,\"clock\":\"CLOCK_THREAD_CPUTIME_ID\",\"timestamp_clock\":\"CLOCK_MONOTONIC\",\"hz\":%u,\"interval_ns\":%" PRIu64 ",\"start_monotonic_ns\":%" PRIu64 ",\"end_monotonic_ns\":%" PRIu64 ",\"start_cpu_ns\":%" PRIu64 ",\"end_cpu_ns\":%" PRIu64 ",\"samples\":%u,\"capacity_drops\":%u,\"bad_signal\":%u,\"bad_tid\":%u,\"clock_errors\":%u,\"kernel_overruns\":%" PRIu64 ",\"drained\":%u,\"foreign_pending\":%u,\"timer_disarmed\":%s,\"timer_deleted\":%s,\"original_mask_restored\":true,\"modules_before_overflow\":%u,\"modules_after_overflow\":%u,\"FPS_claim\":false}\n",
        clean ? "true" : "false", atomic_load_explicit(&owner_tid, memory_order_relaxed), requested_hz, interval_ns,
        start_mono, end_mono, start_cpu, end_cpu, count, dropped, badsig, badthread, badclock,
        atomic_load_explicit(&kernel_overruns, memory_order_relaxed), drained, foreign_pending,
        disarmed ? "true" : "false", deleted ? "true" : "false", modules_before.overflow, modules_after.overflow);
    write_modules(f, &modules_before, "before"); write_modules(f, &modules_after, "after");
    for (uint32_t i = 0; i < count; ++i)
        fprintf(f, "{\"event\":\"pc\",\"ordinal\":%u,\"monotonic_ns\":%" PRIu64 ",\"pc\":\"%016" PRIx64 "\",\"tid\":%u,\"overrun\":%u}\n", i,
            atomic_load_explicit(&samples[i].monotonic_ns, memory_order_relaxed), atomic_load_explicit(&samples[i].pc, memory_order_relaxed),
            atomic_load_explicit(&samples[i].tid, memory_order_relaxed), atomic_load_explicit(&samples[i].overrun, memory_order_relaxed));
    int good_output = !ferror(f);
    if (fclose(f)) good_output = 0;
    if (!clean || !good_output) { errno = EIO; return -1; }
    return 1;
}
#elif !defined(XI_CPU_SAMPLE_TEST)
int xi_cpu_sample_begin(const char *path, uint32_t hz) { (void)path; (void)hz; return 0; }
int xi_cpu_sample_end(void) { return -1; }
int xi_cpu_sample_cleanup_ok(void) { return 1; }
#endif
