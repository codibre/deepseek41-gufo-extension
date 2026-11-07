// Host replica of the EXTENSION's ds4_iq_deq8 IQ1_M path using the extension's own vendored table.
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "src/model/kernels/rocm/mmq/ggml-common.h"

static void ext_iq1m_deq8(const void* blk,int g,float* v){
  const block_iq1_m* b=(const block_iq1_m*)blk;
  const uint16_t* sc=(const uint16_t*)b->scales;
  iq1m_scale_t s; s.u16=(uint16_t)((sc[0]>>12)|((sc[1]>>8)&0x00f0)|((sc[2]>>4)&0x0f00)|(sc[3]&0xf000));
  uint16_t hb; { ggml_half h=s.f16; memcpy(&hb,&h,2); }
  uint32_t sign=(hb>>15)&1u, ex=(hb>>10)&0x1fu, mant=hb&0x3ffu; float d;
  if(ex==0){ d=(float)mant*powf(2.0f,-24); }
  else if(ex==0x1f){ d=__builtin_inf(); if(!mant&&sign)d=-d; }
  else { uint32_t f=(sign<<31)|((ex+112)<<23)|(mant<<13); memcpy(&d,&f,4); }
  const int ib=g>>2,l=g&3;
  const float dl=d*(2.0f*(float)((sc[ib>>1]>>(6*(ib&1)+((l<2)?0:3)))&0x7)+1.0f);
  const uint8_t qhb=b->qh[2*ib+(l>>1)];
  const uint32_t qhl=(uint32_t)qhb>>(4*(l&1));
  const uint32_t idx=(uint32_t)b->qs[4*ib+l]|((qhl&7u)<<8);
  const float delta=-1.0f+IQ1M_DELTA-(float)(qhl&8u)*(2.0f*IQ1M_DELTA/8.0f);
  const uint32_t grid=iq1s_grid_gpu[idx];
  for(int j=0;j<4;++j){const uint32_t byte=(grid>>(8*j))&0xffu;
    v[j]=dl*(float)(byte&0xfu)+dl*delta; v[j+4]=dl*(float)(byte>>4)+dl*delta;}
}

int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); static char buf[56]; fread(buf,1,56,f); fclose(f);
  for(int g=0;g<2;++g){float v[8]; ext_iq1m_deq8(buf,g,v);
    printf("lane%d: ",g); for(int j=0;j<8;++j) printf("%.7g ",v[j]); printf("\n");}
  printf("bytes:"); for(int i=0;i<56;++i) printf("%02x",(uint8_t)buf[i]); printf("\n");
  return 0;
}
