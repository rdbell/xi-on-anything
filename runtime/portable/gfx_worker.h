#pragma once
#ifdef __cplusplus
extern "C" {
#endif
// Must run before the Android SDL/native entry point returns. Drains ordered
// work, joins the thread and restores direct calls. No GPU policy changes.
void gfx_worker_shutdown(void);
#ifdef __cplusplus
}
#endif
