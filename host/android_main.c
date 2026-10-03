/* SDL Android entry point. Local test app only; no saved credentials are imported. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <android/log.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#if defined(FFXI_RENDER_WORKER_BUILD)
#include "gfx_worker.h"
#else
static void gfx_worker_shutdown(void) {}
#endif
#ifdef main
#undef main
#endif
int xi_host_main(int argc, char **argv);
int xi_gfx_test_main(int argc, char **argv);
int xi_format_test_main(void);
int xi_state_test_main(void);
int xi_async_test_main(void);
__attribute__((visibility("default"))) int SDL_main(int argc, char **argv)
{
    const char *dir = NULL;
    for (int i=1;i+1<argc;i++) if (!strcmp(argv[i],"--data-dir")) dir=argv[i+1];
    if (!dir) return 2;
    mkdir(dir,0700);
    char log[2048]; snprintf(log,sizeof log,"%s/host64.log",dir);
    freopen(log,"a",stderr); freopen(log,"a",stdout);
    setvbuf(stderr,NULL,_IONBF,0); setvbuf(stdout,NULL,_IOLBF,0);
    setenv("FFXI_PASSWORD","hxitest1",1);
    setenv("FFXI_ADDONS","0",1); setenv("FFXI_PROFILE","0",1); setenv("FFXI_FX","0",1);
    setenv("FFXI_DISCORD","0",1); setenv("FFXI_VSYNC","0",1);
    /* First native baseline reads original occlusion pixels; no forced-visible answer. */
    setenv("FFXI_PROBE","gpu",1); setenv("FFXI_ASYNC_READBACK","0",1);
    setenv("FFXI_NATIVE_CPU_SAMPLE","0",1); setenv("FFXI_ANDROID_DIAGNOSTICS","0",1);
    setenv("FFXI_FPS","0",1); setenv("FFXI_ADDON_DIAGNOSTICS","0",1);
    setenv("FFXI_RENDER_WORKER","0",1); setenv("FFXI_RENDER_WORKER_DIAGNOSTICS","0",1);
    const char *trial=getenv("FFXI_BENCHMARK_ID"); (void)trial;
    for (int i=1;i+1<argc;i++) if (!strcmp(argv[i],"--user") && strcmp(argv[i+1],"hxitest")) {
        fprintf(stderr,"[android] local test app only accepts hxitest\n"); return 3;
    }
    /* Android-only control options are consumed before the portable host sees argv. */
    for (int i=1;i+1<argc;) {
        const char *key=NULL;
        if (!strcmp(argv[i],"--android-diagnostics")) {
#if defined(FFXI_ANDROID_DIAGNOSTIC)
            if (strcmp(argv[i+1],"0") && strcmp(argv[i+1],"1")) return 4;
            setenv("FFXI_NATIVE_CPU_SAMPLE",argv[i+1],1);
            key="FFXI_ANDROID_DIAGNOSTICS";
#else
            fprintf(stderr,"[android] this APK has no diagnostics; use explicit host-diag build\n"); return 4;
#endif
        }
        else if (!strcmp(argv[i],"--android-sample-hz")) {
            if (strcmp(argv[i+1],"49") && strcmp(argv[i+1],"53") && strcmp(argv[i+1],"199") && strcmp(argv[i+1],"211")) return 4;
            key="FFXI_NATIVE_CPU_HZ";
        }
        else if (!strcmp(argv[i],"--android-render-worker") || !strcmp(argv[i],"--android-worker-stats")) {
            if (strcmp(argv[i+1],"0") && strcmp(argv[i+1],"1")) {
                fprintf(stderr,"[android] worker options require0 or1\n"); return 4;
            }
#if !defined(FFXI_RENDER_WORKER_BUILD)
            if (!strcmp(argv[i+1],"1")) {fprintf(stderr,"[android] this build has no render worker\n"); return 4;}
#endif
            key=!strcmp(argv[i],"--android-render-worker") ? "FFXI_RENDER_WORKER" : "FFXI_RENDER_WORKER_DIAGNOSTICS";
        }
        else if (!strcmp(argv[i],"--android-bench-dir")) key="FFXI_BENCH_DIR";
        else if (!strcmp(argv[i],"--android-addons")) key="FFXI_ADDONS";
        else if (!strcmp(argv[i],"--android-perf-env")) key="FFXI_PERF_ENV";
        else if (!strcmp(argv[i],"--android-readback")) {
            if (strcmp(argv[i+1],"0") && strcmp(argv[i+1],"1")) {
                fprintf(stderr,"[android] readback must be0(sync) or1(async)\n"); return 4;
            }
            key="FFXI_ASYNC_READBACK";
        }
        if (key) {
            setenv(key,argv[i+1],1);
            memmove(argv+i,argv+i+2,(argc-i-1)*sizeof *argv); argc-=2;
        } else ++i;
    }
    for (int i=1;i<argc;i++) if (!strcmp(argv[i],"--android-render-worker") || !strcmp(argv[i],"--android-worker-stats")) {
        fprintf(stderr,"[android] worker option requires a value\n"); return 4;
    }
    /* Physical keyboard/controller baseline: do not open IME at every game window. */
    SDL_SetHint(SDL_HINT_ENABLE_SCREEN_KEYBOARD,"0");
    SDL_SetMainReady();
#if defined(FFXI_ANDROID_DIAGNOSTIC)
    const char *internal=SDL_GetAndroidInternalStoragePath();
    if (internal) setenv("FFXI_NATIVE_CPU_DIR",internal,1);
#endif
    for (int i=1;i<argc;i++) if (!strcmp(argv[i],"--android-format-test") || !strcmp(argv[i],"--android-state-test") || !strcmp(argv[i],"--android-async-test")) {
        int result=!strcmp(argv[i],"--android-format-test") ? xi_format_test_main() :
                   !strcmp(argv[i],"--android-state-test") ? xi_state_test_main() : xi_async_test_main();
        fprintf(stderr,"[android] graphics gate exit %d (%s)\n",result,argv[i]);
        gfx_worker_shutdown();
        return result;
    }
    for (int i=1;i<argc;i++) if (!strcmp(argv[i],"--android-gfx-test")) {
        char *args[] = { "gfx_test", "--window", NULL };
        int result = xi_gfx_test_main(2,args);
        fprintf(stderr,"[android] graphics gate exit %d\n",result);
        gfx_worker_shutdown();
        return result;
    }
    fprintf(stderr,"[android] native ARM64 local test entry; real GPU probes, async-readback=%s\n",getenv("FFXI_ASYNC_READBACK"));
    int ret=xi_host_main(argc,argv);
    fprintf(stderr,"[android] host exit %d\n",ret);
    gfx_worker_shutdown();
    return ret;
}
