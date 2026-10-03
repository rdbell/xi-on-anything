#!/usr/bin/env python3
"""Isolated NDK build of the native host; uses this install's existing generated C.

Does not regenerate/overwrite the Mac build or copy credentials. SDL3 must be a complete
upstream Android-capable source tree (the vendored Mac subset lacks Android sources).
"""
import argparse, concurrent.futures, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import build
import android_translation
NDK=Path(os.environ.get('ANDROID_NDK_HOME',str(Path.home()/'Library/Android/sdk/ndk/27.0.12077973')))
TC=NDK/'toolchains/llvm/prebuilt/darwin-x86_64/bin'
CC=str(TC/'aarch64-linux-android28-clang'); CXX=CC+'++'; AR=str(TC/'llvm-ar')
OUT=ROOT/'build/android'
CACHE=ROOT.parent/'android/cache/xi-native-android'
SDL=CACHE/'SDL3-3.4.16'
FLAGS=['-O2','-g','-fPIC','-ffixed-x18','-fno-strict-aliasing','-DRT_GUEST_WINDOW','-DFFXI_ANDROID_VULKAN','-I'+str(ROOT/'runtime'),'-I'+str(ROOT/'runtime/portable'),'-I'+str(OUT/'generated'),'-I'+str(ROOT/'generated'),'-I'+str(ROOT/'third_party/stb'),'-I'+str(SDL/'include')]
PORTABLE=[x.replace('\\','/').replace('plat_win.c','plat_posix.c') for x in build.PORTABLE]
HOST=['runtime/portable/user32.c','runtime/portable/d3d8.c','runtime/portable/dsound.c','runtime/portable/input.c','runtime/portable/dinput.c','runtime/portable/ws2.c','host/host64.c','host/lsb_login.c','host/datui.c','host/uidraw.c','host/modern.c','host/discord.c','host/signin.c','host/sewave.c','host/ui_art.c','host/keychain.c','host/appdefaults.c']
ADDONS=sorted(str(x.relative_to(ROOT)) for x in (ROOT/'host/addons').iterdir() if x.suffix in ('.c','.cpp'))+['generated/addons_lua.c']
WARN=['-Wno-unused-label','-Wno-unused-variable','-Wno-unused-but-set-variable','-Wno-unused-function','-Wno-parentheses-equality','-Wno-unreachable-code']
def run(args,**kw):
    subprocess.run(list(map(str,args)),cwd=ROOT,check=True,**kw)
def compile_one(job):
    src,obj,extra=job
    s=Path(src); s=s if s.is_absolute() else ROOT/s
    obj.parent.mkdir(parents=True,exist_ok=True)
    flags=FLAGS+extra+(['-std=c++17'] if s.suffix=='.cpp' else ['-std=c11'])
    if str(src).startswith('generated/'): flags+=WARN
    key=hashlib.sha256((s.read_bytes()+json.dumps(flags).encode())).hexdigest()
    stamp=obj.with_suffix('.stamp')
    # Build files on every script invocation when own headers changed; compiler depfile tracks this.
    deps=obj.with_suffix('.d')
    stale=not obj.exists() or not stamp.exists() or stamp.read_text()!=key
    if not stale and deps.exists():
        for d in deps.read_text().replace('\\\n',' ').split(':',1)[-1].split():
            p=Path(d)
            if p.exists() and p.stat().st_mtime>obj.stat().st_mtime: stale=True;break
    if stale:
        cmd=[CXX if s.suffix=='.cpp' else CC]+flags+['-MMD','-MF',str(deps),'-c',str(s),'-o',str(obj)]
        r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        if r.returncode:return str(s),r.stdout+r.stderr
        stamp.write_text(key)
    return str(obj),None
def compile_many(srcs,group,extra=[]):
    jobs=[(s,OUT/'obj'/group/(str(s).replace('/','_').replace('\\','_')+'.o'),extra) for s in srcs]
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for name,err in ex.map(compile_one,jobs):
            if err:errors.append((name,err))
    if errors:
        for name,err in errors:print(name+'\n'+err[-6500:],file=sys.stderr)
        raise SystemExit(1)
    return [j[1] for j in jobs]
def library(name):
    lib=ROOT/'third_party'/name
    m=json.loads((lib/'manifest.json').read_text())
    archive=OUT/'lib'/('lib'+name+'.a');archive.parent.mkdir(parents=True,exist_ok=True)
    if name=='luajit':
        work=OUT/'luajit-src'
        if not work.exists():shutil.copytree(lib,work)
        run(['make','-C',work/'src','-j6','BUILDMODE=static','TARGET_SYS=Linux','HOST_CC=clang','CROSS='+str(TC)+'/',
             'CC='+CC,'STATIC_CC='+CC,'DYNAMIC_CC='+CC+' -fPIC','TARGET_LD='+CC,'TARGET_AR='+AR+' rcus','TARGET_STRIP='+str(TC/'llvm-strip'),
             'XCFLAGS=-DLUAJIT_ENABLE_LUA52COMPAT -fPIC','libluajit.a'])
        shutil.copyfile(work/'src/libluajit.a',archive)
    else:
        objs=[]
        for i,g in enumerate(m['groups']):
            extra=g['flags']+['-I'+str(lib/d) for d in g['include']]+sum((['-idirafter',str(lib/d)] for d in g['idirafter']),[])
            objs+=compile_many([str(lib/s) for s in g['sources']],name+str(i),extra)
        run([AR,'rcs',archive]+objs)
    return archive,['-I'+str(lib/d) for d in m['public']]
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('target',choices=['deps','boot','host','host-diag','gfxtest']);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'generated').mkdir(exist_ok=True)
    build.BUILD=android_translation.android_build(build.BUILD)
    build.BUILD_H=str(OUT/'generated/build.h');build.write_build_h()
    if a.target=='boot':
        generated=sorted(str(x.relative_to(ROOT)) for x in (ROOT/'generated/all').glob('*.c'))
        objs=compile_many(generated,'game',['-I'+str(ROOT/'generated/all')])+compile_many(PORTABLE+['tests/boot64.c'],'boot')
        run([CC,'-o',OUT/'boot64']+objs+['-lm','-ldl'])
        return
    libs=[];extra=['-I'+str(ROOT/'host/addons'),'-DIMGUI_USER_CONFIG="imconfig_xi.h"']
    for n in ['mbedtls','luajit','imgui','luasocket','lfs','sqlite']:
        lib,inc=library(n);libs.append(lib);extra+=inc
    if a.target=='deps':return
    # Shader agent supplies Android glslang archive flags in this manifest.
    spv=json.loads((CACHE/'spirv-build.json').read_text())
    extra+=spv['cflags']
    gfx=['runtime/portable/gfx_hlsl.c','runtime/portable/gfx_hlsl_shaders.c','runtime/portable/gfx_spirv.c','runtime/portable/gfx_vulkan.cpp']
    if a.target=='host-diag': gfx[-1]='host/gfx_vulkan_diagnostic.cpp'
    else:
        gfx+=['runtime/portable/gfx_worker.cpp']
        extra+=['-DFFXI_RENDER_WORKER_BUILD']
    sdl_lib=CACHE/'sdl-build/libSDL3.so'
    if a.target=='gfxtest':
        objs=compile_many(gfx+['tests/gfx_test.c'],'gfxtest',extra)
        run([CXX,'-Wl,-z,max-page-size=16384','-static-libstdc++','-o',OUT/'gfx_test']+objs+['-Wl,--start-group']+spv['libs']+['-Wl,--end-group',str(sdl_lib),'-lvulkan','-llog','-landroid','-lm','-ldl'])
        return
    objs=[]
    for part in ['android-all','ffxi']:
        objs+=compile_many(sorted(str(x.relative_to(ROOT)) for x in (ROOT/'generated'/part).glob('*.c')),part,['-I'+str(ROOT/'generated'/part)])
    host=PORTABLE+HOST+ADDONS+gfx+['host/android_main.c']
    if (ROOT/'host/benchmark.c').exists():host+=['host/benchmark.c']
    diagflags=['-DFFXI_ANDROID_DIAGNOSTIC'] if a.target=='host-diag' else []
    objs+=compile_many(host,'host',extra+['-Dmain=xi_host_main']+diagflags)
    if a.target=='host-diag':
        objs+=compile_many(['host/xi_cpu_sample.c'],'cpu-diagnostic',extra+['-mno-outline-atomics'])
    objs+=compile_many(['tests/gfx_test.c'],'host-gfx-test',extra+['-Dmain=xi_gfx_test_main'])
    objs+=compile_many(['tests/gfx_format_test.c'],'host-format-test',extra+['-Dmain=xi_format_test_main'])
    objs+=compile_many(['tests/gfx_state_test.c'],'host-state-test',extra+['-Dmain=xi_state_test_main'])
    objs+=compile_many(['tests/gfx_async_test.c'],'host-async-test',extra+['-Dmain=xi_async_test_main'])
    run([CXX,'-shared','-Wl,-soname,libmain.so','-Wl,-z,max-page-size=16384','-static-libstdc++','-Wl,--export-dynamic','-o',OUT/'libmain.so']+objs+['-Wl,--start-group']+libs+spv['libs']+['-Wl,--end-group',str(sdl_lib),'-lvulkan','-llog','-landroid','-lm','-ldl'])
    print('built',OUT/'libmain.so')
if __name__=='__main__':main()
