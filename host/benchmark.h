/* Default-off, local-only Present timestamps. No draw/API profiling. */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* FFXI_BENCH_DIR enables the collector. Directory must exist; frames.csv and
 * markers.jsonl must be absent. Only account hxitest is accepted when enabled.
 * Returns 0 disabled, 1 enabled, -1 rejected. Call once before game startup. */
int benchmark_init(const char* account);
/* At the beginning of the existing host Present hook. Disabled cost: one branch. */
void benchmark_frame(void);
/* Flush/close on ordinary host cleanup; registered with atexit too. */
void benchmark_close(void);
int benchmark_enabled(void);

/* Native LuaJIT ffi exports. Labels/phases accept only ASCII letters, numbers,
 * spaces and _.-:; reject invalid/oversized strings without writing them.
 * Marks use the same monotonic clock/UTC anchor as frame timestamps. */
int FFXI_BenchmarkMark(const char* label, const char* phase, uint32_t zone, uint32_t fixture_count);
void FFXI_BenchmarkSetZone(uint32_t zone);

#ifdef __cplusplus
}
#endif
