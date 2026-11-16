/* Lock-free single-producer/single-consumer ring buffer of float samples. Internal. */
#ifndef SLA_RING_H
#define SLA_RING_H

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct sla_ring
{
    float* data;
    uint64_t capacity_frames; /* power of two */
    uint64_t mask;
    uint32_t channels;
    _Atomic uint64_t write_index; /* frames; advanced only by the producer */
    _Atomic uint64_t read_index;  /* frames; advanced only by the consumer */
} sla_ring;

static inline int sla_ring_init(sla_ring* ring, uint32_t channels, uint32_t min_frames)
{
    uint64_t frames = 64;
    while (frames < min_frames)
    {
        frames <<= 1;
    }

    ring->data = (float*)calloc((size_t)(frames * channels), sizeof(float));
    if (ring->data == NULL)
    {
        return 0;
    }

    ring->capacity_frames = frames;
    ring->mask = frames - 1;
    ring->channels = channels;
    atomic_init(&ring->write_index, 0);
    atomic_init(&ring->read_index, 0);
    return 1;
}

static inline void sla_ring_free(sla_ring* ring)
{
    free(ring->data);
    ring->data = NULL;
}

/* Frames that can be read (consumer view). */
static inline uint32_t sla_ring_readable(sla_ring* ring)
{
    uint64_t write = atomic_load_explicit(&ring->write_index, memory_order_acquire);
    uint64_t read = atomic_load_explicit(&ring->read_index, memory_order_acquire);
    return (uint32_t)(write - read);
}

/* Frames that can be written (producer view). */
static inline uint32_t sla_ring_writable(sla_ring* ring)
{
    return (uint32_t)(ring->capacity_frames - sla_ring_readable(ring));
}

/* Producer: copies up to frame_count frames in; returns the number accepted. */
static inline uint32_t sla_ring_write(sla_ring* ring, const float* frames, uint32_t frame_count)
{
    uint64_t write = atomic_load_explicit(&ring->write_index, memory_order_relaxed);
    uint64_t read = atomic_load_explicit(&ring->read_index, memory_order_acquire);
    uint64_t free_frames = ring->capacity_frames - (write - read);
    uint32_t count = frame_count < free_frames ? frame_count : (uint32_t)free_frames;
    uint32_t channels = ring->channels;

    for (uint32_t i = 0; i < count; i++)
    {
        memcpy(ring->data + ((write + i) & ring->mask) * channels, frames + (size_t)i * channels, channels * sizeof(float));
    }

    atomic_store_explicit(&ring->write_index, write + count, memory_order_release);
    return count;
}

/* Consumer: copies up to frame_count frames out; returns the number read. */
static inline uint32_t sla_ring_read(sla_ring* ring, float* frames, uint32_t frame_count)
{
    uint64_t read = atomic_load_explicit(&ring->read_index, memory_order_relaxed);
    uint64_t write = atomic_load_explicit(&ring->write_index, memory_order_acquire);
    uint64_t ready = write - read;
    uint32_t count = frame_count < ready ? frame_count : (uint32_t)ready;
    uint32_t channels = ring->channels;

    for (uint32_t i = 0; i < count; i++)
    {
        memcpy(frames + (size_t)i * channels, ring->data + ((read + i) & ring->mask) * channels, channels * sizeof(float));
    }

    atomic_store_explicit(&ring->read_index, read + count, memory_order_release);
    return count;
}

#endif
