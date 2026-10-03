/* Exact Horizon SetTarget bridge admission without running the game.
 * Reuse the PE loader/memory fixtures; mock only the final dispatch to inspect
 * the real bridge's resolved function, this pointer and ActorPointer argument.
 * clang -std=c11 -O1 -g -Wall -Wextra -DRT_GUEST_WINDOW -Iruntime -Iruntime/portable
 *   -Ihost/addons tests/game_target_test.c host/addons/game.c -o build/game_target_test
 */
#include "gthread.h"
#define main unused_game_pattern_test_main
#include "game_test.c"
#undef main

static GThread thread;
static uint32_t called_fn, called_self, called_args[3];
static unsigned called_nargs, calls;
GThread* gt_self(void) { return &thread; }
uint32_t guest_call(uint32_t fn, unsigned nargs, const uint32_t* args)
{
    (void)fn; (void)nargs; (void)args;
    abort(); /* this fixture must not enter any unrelated fallback */
}
uint32_t guest_thiscall(uint32_t fn, uint32_t self, unsigned nargs, const uint32_t* args)
{
    called_fn=fn;called_self=self;called_nargs=nargs;++calls;
    if(nargs==3)memcpy(called_args,args,sizeof called_args);
    return 1;
}
static int image_bytes(uint32_t address,const uint8_t* want,size_t n)
{
    return address>=img_base && (uint64_t)address+n<=(uint64_t)img_base+img_size && !memcmp(GUEST_PTR(address),want,n);
}
int main(int argc,char** argv)
{
    const char* path=argc>1?argv[1]:"generated/FFXiMain.unpacked.dll";
    rt_guest_base=mmap(NULL,1ull<<32,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE|MAP_NORESERVE,-1,0);
    if(rt_guest_base==MAP_FAILED || !load_image(path))return 2;
    uint32_t site=xi_game_ptr_value(XI_P_SET_TARGET);
    check(site==0x1007a427,"exact Horizon SetTarget site");
    check(xi_game_rd32(site+2)==0x105764e4,"exact Horizon this global");
    check(site+0x1b+xi_game_rd32(site+0x17)==0x10156540,"exact Horizon callee");
    const uint8_t actor_load[]={0x8b,0x80,0xa0,0,0,0};
    const uint8_t actor_push_call[]={0x52,0x51,0x50,0x8b,0xce,0xe8,0x9b,0xfe,0xff,0xff};
    check(image_bytes(0x10156695,actor_load,sizeof actor_load),"original index setter dereferences entity+ActorPointer");
    check(image_bytes(0x1015669b,actor_push_call,sizeof actor_push_call),"original index setter pushes actor and calls exact callee");
    const uint8_t actor_backlink[]={0x8b,0x7d,0x70};
    check(image_bytes(0x1015807d,actor_backlink,sizeof actor_backlink),"callee reads actor+0x70 backlink");
    uint32_t map=xi_game_ptr_value(XI_P_ENTITY_MAP), entity=HEAP+0x1000,actor=HEAP+0x3000,self=HEAP+0x5000,index=7;
    wr32(map+index*4,entity);wr32(entity+0xa0,actor);wr32(actor+0x70,entity);wr32(0x105764e4,self);
    thread.g.ecx=0x12345678;thread.g.esp=HEAP+0x8000;
    check(xi_game_set_target(index)==1 && calls==1,"valid actor dispatches once");
    check(called_fn==0x10156540 && called_self==self && called_nargs==3,"exact callee/self/arity");
    check(called_args[0]==actor && called_args[0]!=entity && called_args[1]==1 && called_args[2]==0,"actor argument category and original flags");
    wr32(entity+0xa0,0);check(xi_game_set_target(index)==0 && calls==1,"empty actor rejects without dispatch");
    wr32(entity+0xa0,0xfffffff0);check(xi_game_set_target(index)==0 && calls==1,"unmapped actor rejects without dispatch");
    wr32(entity+0xa0,actor);wr32(actor+0x70,entity+0x100);check(xi_game_set_target(index)==0 && calls==1,"different entity backlink rejects without dispatch");
    wr32(actor+0x70,entity);wr32(map+index*4,0);check(xi_game_set_target(index)==0 && calls==1,"missing entity rejects without dispatch");
    check(xi_game_set_target(xi_game_entity_count())==0 && calls==1,"out-of-range entity rejects without dispatch");
    wr32(map+index*4,entity);wr32(0x105764e4,0);check(xi_game_set_target(index)==0 && calls==1,"empty target manager rejects without dispatch");
    printf("game_target_test: %d checks, %d failures\n",checks,failures);
    return failures?1:0;
}
