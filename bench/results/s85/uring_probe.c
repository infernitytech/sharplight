// Disk probe: O_DIRECT reads with io_uring (raw syscalls) at a queue depth, or with N pread threads, or mmap faults.
// usage: uring_probe <file> <mode uring|pread|mmap> <block KiB> <qd/threads> <total MiB> [random 0/1] [offset GiB]
#define _GNU_SOURCE
#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rng = 88172645463325252ull;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static int fd; static size_t bs; static long long nblocks; static int randomize; static off_t base, span;
static atomic_llong next_block;
static off_t offset_of(long long i) {
    if (!randomize) return base + (off_t)i * bs;
    return base + (off_t)(xr() % (span / bs)) * bs;
}

static void *pread_worker(void *arg) {
    void *buf; posix_memalign(&buf, 4096, bs);
    uint64_t local = (uint64_t)(uintptr_t)arg * 2654435761u + 1;
    for (;;) {
        long long i = atomic_fetch_add(&next_block, 1);
        if (i >= nblocks) break;
        off_t off;
        if (randomize) { local ^= local << 13; local ^= local >> 7; local ^= local << 17; off = base + (off_t)(local % (span / bs)) * bs; }
        else off = base + (off_t)i * bs;
        if (pread(fd, buf, bs, off) != (ssize_t)bs) { perror("pread"); exit(1); }
    }
    free(buf); return NULL;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 2; }
    const char *path = argv[1], *mode = argv[2];
    bs = (size_t)atol(argv[3]) << 10; int qd = atoi(argv[4]); long long total = atoll(argv[5]) << 20;
    randomize = argc > 6 ? atoi(argv[6]) : 0; base = argc > 7 ? (off_t)atoll(argv[7]) << 30 : 0;
    struct stat st; stat(path, &st); span = st.st_size - base; if (total > span) total = span / bs * bs;
    nblocks = total / bs;
    double t0, t1;
    if (!strcmp(mode, "mmap")) {
        fd = open(path, O_RDONLY);
        unsigned char *m = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        posix_fadvise(fd, base, total, POSIX_FADV_DONTNEED);
        madvise(m + base, total, MADV_RANDOM);
        volatile unsigned long sum = 0; t0 = now();
        for (long long i = 0; i < nblocks; i++) { off_t o = offset_of(i); for (size_t p = 0; p < bs; p += 4096) sum += m[o + p]; }
        t1 = now();
    } else {
        fd = open(path, O_RDONLY | O_DIRECT);
        if (fd < 0) { perror("open O_DIRECT"); return 1; }
        if (!strcmp(mode, "pread")) {
            pthread_t th[256]; t0 = now();
            for (int i = 0; i < qd; i++) pthread_create(&th[i], NULL, pread_worker, (void *)(uintptr_t)(i + 1));
            for (int i = 0; i < qd; i++) pthread_join(th[i], NULL);
            t1 = now();
        } else {
            struct io_uring_params p; memset(&p, 0, sizeof p);
            int rfd = syscall(__NR_io_uring_setup, qd, &p);
            if (rfd < 0) { perror("io_uring_setup"); return 1; }
            size_t sqsz = p.sq_off.array + p.sq_entries * sizeof(unsigned), cqsz = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
            unsigned char *sq = mmap(0, sqsz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, rfd, IORING_OFF_SQ_RING);
            unsigned char *cq = mmap(0, cqsz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, rfd, IORING_OFF_CQ_RING);
            struct io_uring_sqe *sqes = mmap(0, p.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, rfd, IORING_OFF_SQES);
            unsigned *sqtail = (unsigned *)(sq + p.sq_off.tail), *sqmask = (unsigned *)(sq + p.sq_off.ring_mask), *sqarr = (unsigned *)(sq + p.sq_off.array);
            unsigned *cqhead = (unsigned *)(cq + p.cq_off.head), *cqtail = (unsigned *)(cq + p.cq_off.tail), *cqmask = (unsigned *)(cq + p.cq_off.ring_mask);
            struct io_uring_cqe *cqes = (struct io_uring_cqe *)(cq + p.cq_off.cqes);
            void **bufs = calloc(qd, sizeof(void *));
            for (int i = 0; i < qd; i++) posix_memalign(&bufs[i], 4096, bs);
            long long issued = 0, done = 0; int inflight = 0; int freeidx[1024]; int nfree = qd; for (int i = 0; i < qd; i++) freeidx[i] = i;
            t0 = now();
            while (done < nblocks) {
                unsigned tail = *sqtail; int add = 0;
                while (nfree > 0 && issued < nblocks) {
                    int b = freeidx[--nfree]; unsigned idx = tail & *sqmask; struct io_uring_sqe *e = &sqes[idx]; memset(e, 0, sizeof *e);
                    e->opcode = IORING_OP_READ; e->fd = fd; e->addr = (uintptr_t)bufs[b]; e->len = bs; e->off = offset_of(issued); e->user_data = b;
                    sqarr[idx] = idx; tail++; issued++; add++; inflight++;
                }
                __atomic_store_n(sqtail, tail, __ATOMIC_RELEASE);
                if (syscall(__NR_io_uring_enter, rfd, add, 1, IORING_ENTER_GETEVENTS, NULL, 0) < 0) { perror("enter"); return 1; }
                unsigned h = *cqhead, t = __atomic_load_n(cqtail, __ATOMIC_ACQUIRE);
                for (; h != t; h++) { struct io_uring_cqe *c = &cqes[h & *cqmask]; if (c->res != (int)bs) { fprintf(stderr, "res %d\n", c->res); return 1; } freeidx[nfree++] = (int)c->user_data; done++; inflight--; }
                __atomic_store_n(cqhead, h, __ATOMIC_RELEASE);
            }
            t1 = now();
        }
    }
    double gb = (double)nblocks * bs / 1e9;
    printf("%s bs=%zuKiB qd=%d rand=%d %.2f GB in %.2f s: %.2f GB/s\n", mode, bs >> 10, qd, randomize, gb, t1 - t0, gb / (t1 - t0));
    return 0;
}
