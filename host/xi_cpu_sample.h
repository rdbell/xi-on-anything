/* Android-only local diagnostic; no source/build integration is implied. */
#pragma once
#include <stdint.h>
/* Explicit FFXI_NATIVE_CPU_SAMPLE=1 gate is required. Call begin/end on the
 * same verified SDL game thread. path must name a new owned trial output.
 * begin: 0 disabled, 1 armed, -1 rejected/error. end: 1 complete, -1 error.
 * Only one capture attempt is allowed per process. No timer exists by default. */
int xi_cpu_sample_begin(const char *path, uint32_t hz);
int xi_cpu_sample_end(void);
/* If 0 after any error, keep the diagnostic app quarantined and stop it. */
int xi_cpu_sample_cleanup_ok(void);
