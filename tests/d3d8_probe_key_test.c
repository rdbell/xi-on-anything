/* Actual Android frontend code exercised in an isolated host executable.
 * No GameStart, game/account, GPU, phone or persistent configuration. */
#define RT_GUEST_WINDOW 1
#define FFXI_ANDROID_VULKAN 1
#include "../runtime/portable/d3d8.c"

unsigned char* rt_guest_base;
uint32_t rt_reloc_delta;
GuestFn rt_wrap_probe_ctor,rt_wrap_probe_dtor;
static unsigned ctor_calls,dtor_calls,checks,errors;
static uint32_t blocked_page=UINT32_MAX;
int gwin_is_committed(uint32_t address) {return address<0x11000000u && (address&~0xfffu)!=blocked_page;}
static void original_ctor(Guest* guest) {++ctor_calls;wr32(guest->ecx+4,0);wr16(guest->ecx+2,0);guest->eax=guest->ecx;guest->ecx=0;guest->esp+=4;}
static void original_dtor(Guest* guest) {++dtor_calls;guest->ecx=0;guest->esp+=4;}
const GuestFn rt_orig_probe_ctor=original_ctor,rt_orig_probe_dtor=original_dtor;
static void expect(int value,const char* what) {++checks;if(!value){++errors;fprintf(stderr,"probe-key failure: %s\n",what);}}
int main(void)
{
    rt_guest_base=calloc(0x11000000u,1);if(!rt_guest_base)return 2;
    g_android_probe_hooks=1;
    Obj surface={0};surface.width=surface.height=16;surface.format=FMT_A8R8G8B8;surface.usage=USAGE_RENDERTARGET;surface.guest=0x6000;
    Guest guest={0};guest.esp=0x7000;guest.ebp=0x8000;
    uint32_t rect=0x1032a17c;
    wr32(guest.esp,0x1006c72f);wr32(rect,0);wr32(rect+4,0);wr32(rect+8,16);wr32(rect+12,16);
    expect(!android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY),"unregistered lifetime exact");
    Guest lifetime={0};lifetime.ecx=guest.ebp;lifetime.esp=0x9000;
    android_wrap_probe_ctor(&lifetime);wr32(guest.ebp+4,surface.guest);
    uint64_t first=android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY);
    expect(first!=0,"registered valid leaf admitted");expect(ctor_calls==1 && lifetime.eax==guest.ebp && lifetime.ecx==0 && lifetime.esp==0x9004,"original ctor still executes and keeps its register/stack effects");
    expect((uint32_t)first==guest.ebp,"logical key retains object");
    lifetime.ecx=guest.ebp;lifetime.esp=0x9000;android_wrap_probe_dtor(&lifetime);
    expect(!android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY),"destroyed lifetime exact");expect(dtor_calls==1 && lifetime.ecx==0 && lifetime.esp==0x9004,"original dtor still executes");
    lifetime.ecx=guest.ebp;lifetime.esp=0x9000;android_wrap_probe_ctor(&lifetime);wr32(guest.ebp+4,surface.guest);
    uint64_t second=android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY);
    expect(second && second!=first,"same address new lifetime different key");
    wr32(guest.esp,0x1006c730);expect(!android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY),"unknown return exact");wr32(guest.esp,0x1006c72f);
    wr32(guest.ebp+4,surface.guest+4);expect(!android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY),"surface backlink mismatch exact");wr32(guest.ebp+4,surface.guest);
    expect(!android_surface_probe_key(&guest,&surface,rect,0),"write lock exact");
    expect(!android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY|0x20),"additional flags exact");
    surface.format=22;expect(!android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY),"different format exact");surface.format=21;
    surface.usage=0;expect(!android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY),"nonrender target exact");surface.usage=1;
    surface.width=15;expect(!android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY),"different size exact");surface.width=16;
    expect(!android_surface_probe_key(&guest,&surface,0,LOCK_READONLY),"different rect pointer exact");
    wr32(rect+12,15);expect(!android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY),"partial rect exact");wr32(rect+12,16);
    blocked_page=rect&~0xfffu;expect(!android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY),"unmapped rect exact");blocked_page=UINT32_MAX;
    expect(!android_probe_mapped(UINT32_MAX-3,8),"range wrap rejected");
    expect(!android_probe_mapped(0,8),"null range rejected");
    expect(!android_probe_mapped(0x8000,0),"empty range rejected");
    blocked_page=0x9000;expect(!android_probe_mapped(0x8ffc,8),"range crossing unmapped second page rejected");blocked_page=UINT32_MAX;
    guest.ebp=0xa17f;android_probe_created(guest.ebp);wr32(guest.ebp+4,surface.guest);
    expect(android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY)!=0,"legitimate unaligned effect lifetime admitted safely");
    g_android_probe_hooks=0;expect(!android_surface_probe_key(&guest,&surface,rect,LOCK_READONLY),"missing hooks exact");g_android_probe_hooks=1;
    memset(g_android_probe_lives,0,sizeof g_android_probe_lives);g_android_probe_generation=0;
    for(unsigned n=0;n<ANDROID_PROBE_LIVES;n++)android_probe_created(0x10000+n*4);
    unsigned live=0;for(unsigned n=0;n<ANDROID_PROBE_LIVES;n++)live+=g_android_probe_lives[n].state==1;
    expect(live==ANDROID_PROBE_LIVES,"bounded registry fills exactly");
    android_probe_created(0x20000);expect(android_probe_life(0x20000,0)==NULL,"overflow remains unknown");
    android_probe_destroyed(0x10000);android_probe_created(0x20000);
    expect(android_probe_life(0x20000,0) && android_probe_life(0x20000,0)->state==1,"tombstone reused safely after destruction");
    expect(android_probe_life(0x10004,0) && android_probe_life(0x10004,0)->state==1,"collision search survives tombstone reuse");
    g_android_probe_generation=UINT32_MAX;android_probe_destroyed(0x10004);android_probe_created(0x30000);
    expect(!android_probe_life(0x30000,0),"generation wrap remains unknown instead of aliasing");
    g_android_probe_hooks=1;android_probe_created(0x10008);
    expect(!g_android_probe_hooks && android_probe_life(0x10008,0)->generation==0
        && android_probe_life(0x10008,0)->state==2,"exhausted ctor at existing live address invalidates old lifetime and disables delayed policy");
    free(rt_guest_base);printf("{\"test\":\"Android actual frontend probe lifetime/key\",\"checks\":%u,\"failures\":%u,\"runtime_or_FPS_claim\":false}\n",checks,errors);
    return errors!=0;
}
