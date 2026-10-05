// badmap: find bad VRAM words on one GPU and tag every 64 KB block so a host-side
// read-only BAR1 scan can map each block to its FB physical address.
//
// Usage: badmap <outdir> <passes>
//   1. Allocates nearly all free VRAM (1 GiB chunks, then smaller).
//   2. Runs <passes> write/read-back passes with rotating patterns over all of it and
//      logs every mismatched 32-bit word to <outdir>/errors.csv.
//   3. Writes a tag at the start of every 64 KB block: {MAGIC0, block index, MAGIC1, ~block index}.
//   4. Writes <outdir>/ready and waits (up to 60 min) for <outdir>/done before freeing.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static const uint32_t MAGIC0 = 0xB4D3A9C1u, MAGIC1 = 0x5EEDF00Du;
static const size_t BLOCK = 64 * 1024, WORDS_PER_BLOCK = BLOCK / 4;
static const unsigned MAX_ERR = 1u << 20;

struct Err { uint64_t word; uint32_t expect, got; uint32_t pass; uint32_t pad; };

__device__ __forceinline__ uint32_t pattern(uint64_t gw, uint32_t pass) {
    switch (pass % 6) {
        case 0: return 0xFFFFFFFFu;
        case 1: return 0x00000000u;
        case 2: return (gw & 1) ? 0xAAAAAAAAu : 0x55555555u;
        case 3: return (gw & 1) ? 0x55555555u : 0xAAAAAAAAu;
        case 4: return (uint32_t)gw ^ (uint32_t)(gw >> 32) ^ (pass * 0x9E3779B9u);   // address pattern
        default: {                                                                     // hash pattern
            uint64_t x = gw * 0x9E3779B97F4A7C15ull + pass * 0xBF58476D1CE4E5B9ull;
            x ^= x >> 31; x *= 0x94D049BB133111EBull; x ^= x >> 29; return (uint32_t)x;
        }
    }
}

__global__ void fill(uint32_t *p, size_t n, uint64_t base, uint32_t pass) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = pattern(base + i, pass);
}

__global__ void check(const uint32_t *p, size_t n, uint64_t base, uint32_t pass, Err *errs, unsigned *nerr) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t e = pattern(base + i, pass), g = p[i];
        if (g != e) {
            unsigned k = atomicAdd(nerr, 1u);
            if (k < MAX_ERR) errs[k] = Err{base + i, e, g, pass, 0};
        }
    }
}

__global__ void tag(uint32_t *p, size_t nblocks, uint64_t blockBase) {
    for (size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x; b < nblocks; b += (size_t)gridDim.x * blockDim.x) {
        uint32_t *q = p + b * WORDS_PER_BLOCK; uint32_t idx = (uint32_t)(blockBase + b);
        q[0] = MAGIC0; q[1] = idx; q[2] = MAGIC1; q[3] = ~idx;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: badmap <outdir> <passes>\n"); return 2; }
    std::string out = argv[1]; int passes = atoi(argv[2]);
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device %s, %d SMs, pci %04x:%02x:%02x\n", pr.name, pr.multiProcessorCount, pr.pciDomainID, pr.pciBusID, pr.pciDeviceID);

    Err *errs; unsigned *nerr; CK(cudaMalloc(&errs, sizeof(Err) * MAX_ERR)); CK(cudaMalloc(&nerr, sizeof(unsigned)));
    CK(cudaMemset(nerr, 0, sizeof(unsigned)));

    std::vector<std::pair<uint32_t *, size_t>> chunks;   // (ptr, bytes)
    size_t tryb = 1ull << 30, total = 0;
    while (tryb >= (2ull << 20)) {
        void *p = nullptr;
        if (cudaMalloc(&p, tryb) == cudaSuccess) { chunks.push_back({(uint32_t *)p, tryb}); total += tryb; }
        else { cudaGetLastError(); tryb >>= 1; }
    }
    size_t fr, tt; CK(cudaMemGetInfo(&fr, &tt));
    printf("allocated %zu chunks, %.2f GiB (device total %.2f GiB, left free %.1f MiB)\n",
           chunks.size(), total / 1073741824.0, tt / 1073741824.0, fr / 1048576.0);
    fflush(stdout);

    // Global word / block numbering follows chunk order; chunk sizes are multiples of 64 KB.
    for (int pass = 0; pass < passes; pass++) {
        uint64_t base = 0;
        for (auto &c : chunks) { fill<<<4096, 256>>>(c.first, c.second / 4, base, pass); base += c.second / 4; }
        CK(cudaDeviceSynchronize());
        base = 0;
        for (auto &c : chunks) { check<<<4096, 256>>>(c.first, c.second / 4, base, pass, errs, nerr); base += c.second / 4; }
        CK(cudaDeviceSynchronize());
        if (pass % 10 == 9 || pass == passes - 1) {
            unsigned h; CK(cudaMemcpy(&h, nerr, sizeof h, cudaMemcpyDeviceToHost));
            printf("pass %d: %u errors so far\n", pass + 1, h); fflush(stdout);
        }
    }
    unsigned h; CK(cudaMemcpy(&h, nerr, sizeof h, cudaMemcpyDeviceToHost));
    unsigned keep = h < MAX_ERR ? h : MAX_ERR;
    std::vector<Err> he(keep);
    if (keep) CK(cudaMemcpy(he.data(), errs, sizeof(Err) * keep, cudaMemcpyDeviceToHost));
    FILE *f = fopen((out + "/errors.csv").c_str(), "w");
    fprintf(f, "word,block,word_in_block,expect,got,xor,pass\n");
    for (auto &e : he) fprintf(f, "%llu,%llu,%llu,0x%08x,0x%08x,0x%08x,%u\n", (unsigned long long)e.word,
                               (unsigned long long)(e.word / WORDS_PER_BLOCK), (unsigned long long)(e.word % WORDS_PER_BLOCK),
                               e.expect, e.got, e.expect ^ e.got, e.pass);
    fclose(f);
    printf("total errors %u (stored %u)\n", h, keep);

    uint64_t bbase = 0;
    for (auto &c : chunks) { size_t nb = c.second / BLOCK; tag<<<1024, 256>>>(c.first, nb, bbase); bbase += nb; }
    CK(cudaDeviceSynchronize());
    f = fopen((out + "/chunks.csv").c_str(), "w");
    fprintf(f, "chunk,first_block,blocks,bytes\n"); bbase = 0;
    for (size_t i = 0; i < chunks.size(); i++) { size_t nb = chunks[i].second / BLOCK; fprintf(f, "%zu,%llu,%zu,%zu\n", i, (unsigned long long)bbase, nb, chunks[i].second); bbase += nb; }
    fclose(f);
    printf("tagged %llu blocks; ready for BAR1 scan\n", (unsigned long long)bbase); fflush(stdout);
    fclose(fopen((out + "/ready").c_str(), "w"));
    struct stat st;
    for (int i = 0; i < 3600 && stat((out + "/done").c_str(), &st) != 0; i++) sleep(1);
    printf("done\n");
    return 0;
}
