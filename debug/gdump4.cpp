// gdump4: dequant slabs using llama.cpp fork's OWN dequantize_row_* functions. No transcriptions.
// usage: gdump4 in.bin out.f32 iq1m|q5_k|q8_0|bf16|f32 ROWS COLS
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

extern "C" void dequantize_row_iq1_m(const block_iq1_m * x, float * y, int64_t k);
extern "C" void dequantize_row_q5_K  (const block_q5_K  * x, float * y, int64_t k);
extern "C" void dequantize_row_q8_0  (const block_q8_0  * x, float * y, int64_t k);

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr,"usage: gdump4 in out type rows cols\n"); return 1; }
  const char* type = argv[3];
  long rows = atol(argv[4]), cols = atol(argv[5]);
  FILE* f = fopen(argv[1], "rb"); if (!f) { perror("open"); return 1; }
  size_t blen; int per;
  if (!strcmp(type,"iq1m")) { blen=sizeof(block_iq1_m); per=QK_K; }
  else if (!strcmp(type,"q5_k")) { blen=sizeof(block_q5_K); per=QK_K; }
  else if (!strcmp(type,"q8_0")) { blen=sizeof(block_q8_0); per=QK8_0; }
  else if (!strcmp(type,"bf16")) { blen=2; per=1; }
  else if (!strcmp(type,"f32"))  { blen=4; per=1; }
  else { fprintf(stderr,"bad type\n"); return 1; }
  long n = rows*cols;
  std::vector<unsigned char> in(n*blen/per + blen);
  size_t got = fread(in.data(),1,in.size(),f); fclose(f); (void)got;
  std::vector<float> out(n);
  for (long r=0;r<rows;++r) {
    const unsigned char* p = in.data() + r*(cols*blen/per);
    float* y = out.data() + r*cols;
    if (!strcmp(type,"iq1m")) dequantize_row_iq1_m((const block_iq1_m*)p,y,cols);
    else if (!strcmp(type,"q5_k")) dequantize_row_q5_K((const block_q5_K*)p,y,cols);
    else if (!strcmp(type,"q8_0")) dequantize_row_q8_0((const block_q8_0*)p,y,cols);
    else if (!strcmp(type,"bf16")) for(long c=0;c<cols;++c){ uint32_t u=(uint32_t)p[2*c]|((uint32_t)p[2*c+1]<<16); u<<=16; float v; memcpy(&v,&u,4); y[c]=v; }
    else memcpy(y,p,cols*4);
  }
  FILE* o=fopen(argv[2],"wb"); fwrite(out.data(),4,n,o); fclose(o);
  printf("ok %ld rows\n", rows); return 0;
}
