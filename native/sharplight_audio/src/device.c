#include "sharplight_audio.h"

#include "miniaudio.h"

#include "internal.h"
#include "ring.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define SLA_MAX_CHANNELS 32u
#define SLA_MAX_RING_FRAMES (1u << 24)

struct sla_context
{
    ma_context context;
};

struct sla_stream
{
    ma_device device;
    sla_ring ring;
    sla_device_kind kind;
    uint32_t sample_rate;
    uint32_t channels;
    int started;
    _Atomic uint64_t frames_transferred;
    _Atomic uint64_t underruns;
    _Atomic uint64_t overruns;
};

sla_result sla_context_create(int32_t use_null_backend, sla_context** out_context)
{
    if (out_context == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    *out_context = NULL;
    sla_context* context = (sla_context*)calloc(1, sizeof(*context));
    if (context == NULL)
    {
        return SLA_ERR_OUT_OF_MEMORY;
    }

    ma_backend null_only[] = { ma_backend_null };
    ma_context_config config = ma_context_config_init();
    ma_result result = use_null_backend != 0 ? ma_context_init(null_only, 1, &config, &context->context) : ma_context_init(NULL, 0, &config, &context->context);
    if (result != MA_SUCCESS)
    {
        free(context);
        return SLA_ERR_NO_BACKEND;
    }

    *out_context = context;
    return SLA_OK;
}

void sla_context_destroy(sla_context* context)
{
    if (context == NULL)
    {
        return;
    }

    ma_context_uninit(&context->context);
    free(context);
}

const char* sla_context_backend_name(const sla_context* context)
{
    if (context == NULL)
    {
        return "";
    }

    return ma_get_backend_name(context->context.backend);
}

static sla_result enumerate(sla_context* context, sla_device_kind kind, ma_device_info** infos, ma_uint32* count)
{
    if (context == NULL || (kind != SLA_DEVICE_PLAYBACK && kind != SLA_DEVICE_CAPTURE))
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    ma_device_info* playback = NULL;
    ma_uint32 playback_count = 0;
    ma_device_info* capture = NULL;
    ma_uint32 capture_count = 0;
    if (ma_context_get_devices(&context->context, &playback, &playback_count, &capture, &capture_count) != MA_SUCCESS)
    {
        return SLA_ERR_DEVICE;
    }

    *infos = kind == SLA_DEVICE_PLAYBACK ? playback : capture;
    *count = kind == SLA_DEVICE_PLAYBACK ? playback_count : capture_count;
    return SLA_OK;
}

sla_result sla_device_count(sla_context* context, sla_device_kind kind, int32_t* out_count)
{
    if (out_count == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    ma_device_info* infos = NULL;
    ma_uint32 count = 0;
    sla_result result = enumerate(context, kind, &infos, &count);
    if (result != SLA_OK)
    {
        return result;
    }

    *out_count = (int32_t)count;
    return SLA_OK;
}

sla_result sla_device_info(sla_context* context, sla_device_kind kind, int32_t index, char* name_buffer, int32_t name_capacity, int32_t* out_is_default)
{
    if (name_buffer == NULL || name_capacity <= 0 || index < 0)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    ma_device_info* infos = NULL;
    ma_uint32 count = 0;
    sla_result result = enumerate(context, kind, &infos, &count);
    if (result != SLA_OK)
    {
        return result;
    }

    if ((ma_uint32)index >= count)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    size_t length = strlen(infos[index].name);
    size_t copy = length < (size_t)(name_capacity - 1) ? length : (size_t)(name_capacity - 1);
    memcpy(name_buffer, infos[index].name, copy);
    name_buffer[copy] = '\0';
    if (out_is_default != NULL)
    {
        *out_is_default = infos[index].isDefault ? 1 : 0;
    }

    return copy < length ? SLA_ERR_BUFFER_TOO_SMALL : SLA_OK;
}

static void data_callback(ma_device* device, void* output, const void* input, ma_uint32 frame_count)
{
    sla_stream* stream = (sla_stream*)device->pUserData;
    if (stream->kind == SLA_DEVICE_PLAYBACK)
    {
        float* destination = (float*)output;
        uint32_t read = sla_ring_read(&stream->ring, destination, frame_count);
        if (read < frame_count)
        {
            memset(destination + (size_t)read * stream->channels, 0, (size_t)(frame_count - read) * stream->channels * sizeof(float));
            atomic_fetch_add_explicit(&stream->underruns, frame_count - read, memory_order_relaxed);
        }

        atomic_fetch_add_explicit(&stream->frames_transferred, read, memory_order_relaxed);
    }
    else
    {
        uint32_t written = sla_ring_write(&stream->ring, (const float*)input, frame_count);
        if (written < frame_count)
        {
            atomic_fetch_add_explicit(&stream->overruns, frame_count - written, memory_order_relaxed);
        }

        atomic_fetch_add_explicit(&stream->frames_transferred, written, memory_order_relaxed);
    }
}

sla_result sla_stream_open(sla_context* context, sla_device_kind kind, int32_t device_index, uint32_t sample_rate, uint32_t channels, uint32_t ring_frames, sla_stream** out_stream)
{
    if (out_stream == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    *out_stream = NULL;
    if (context == NULL || (kind != SLA_DEVICE_PLAYBACK && kind != SLA_DEVICE_CAPTURE) || sample_rate < 8000 || sample_rate > 384000 ||
        channels == 0 || channels > SLA_MAX_CHANNELS || ring_frames == 0 || ring_frames > SLA_MAX_RING_FRAMES)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    ma_device_id selected_id;
    int has_device_id = 0;
    if (device_index >= 0)
    {
        ma_device_info* infos = NULL;
        ma_uint32 count = 0;
        sla_result result = enumerate(context, kind, &infos, &count);
        if (result != SLA_OK)
        {
            return result;
        }

        if ((ma_uint32)device_index >= count)
        {
            return SLA_ERR_INVALID_ARGUMENT;
        }

        selected_id = infos[device_index].id;
        has_device_id = 1;
    }

    sla_stream* stream = (sla_stream*)calloc(1, sizeof(*stream));
    if (stream == NULL)
    {
        return SLA_ERR_OUT_OF_MEMORY;
    }

    if (!sla_ring_init(&stream->ring, channels, ring_frames))
    {
        free(stream);
        return SLA_ERR_OUT_OF_MEMORY;
    }

    stream->kind = kind;
    stream->sample_rate = sample_rate;
    stream->channels = channels;
    atomic_init(&stream->frames_transferred, 0);
    atomic_init(&stream->underruns, 0);
    atomic_init(&stream->overruns, 0);

    ma_device_config config = ma_device_config_init(kind == SLA_DEVICE_PLAYBACK ? ma_device_type_playback : ma_device_type_capture);
    config.sampleRate = sample_rate;
    config.dataCallback = data_callback;
    config.pUserData = stream;
    if (kind == SLA_DEVICE_PLAYBACK)
    {
        config.playback.format = ma_format_f32;
        config.playback.channels = channels;
        config.playback.pDeviceID = has_device_id ? &selected_id : NULL;
    }
    else
    {
        config.capture.format = ma_format_f32;
        config.capture.channels = channels;
        config.capture.pDeviceID = has_device_id ? &selected_id : NULL;
    }

    if (ma_device_init(&context->context, &config, &stream->device) != MA_SUCCESS)
    {
        sla_ring_free(&stream->ring);
        free(stream);
        return SLA_ERR_DEVICE;
    }

    *out_stream = stream;
    return SLA_OK;
}

sla_result sla_stream_start(sla_stream* stream)
{
    if (stream == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    if (stream->started)
    {
        return SLA_OK;
    }

    if (ma_device_start(&stream->device) != MA_SUCCESS)
    {
        return SLA_ERR_DEVICE;
    }

    stream->started = 1;
    return SLA_OK;
}

sla_result sla_stream_stop(sla_stream* stream)
{
    if (stream == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    if (!stream->started)
    {
        return SLA_OK;
    }

    ma_result result = ma_device_stop(&stream->device);
    stream->started = 0;
    return result == MA_SUCCESS ? SLA_OK : SLA_ERR_DEVICE;
}

void sla_stream_close(sla_stream* stream)
{
    if (stream == NULL)
    {
        return;
    }

    ma_device_uninit(&stream->device);
    sla_ring_free(&stream->ring);
    free(stream);
}

sla_result sla_stream_write(sla_stream* stream, const float* frames, uint32_t frame_count, uint32_t* out_frames)
{
    if (stream == NULL || out_frames == NULL || (frames == NULL && frame_count != 0) || stream->kind != SLA_DEVICE_PLAYBACK)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    *out_frames = sla_ring_write(&stream->ring, frames, frame_count);
    return SLA_OK;
}

sla_result sla_stream_read(sla_stream* stream, float* frames, uint32_t frame_count, uint32_t* out_frames)
{
    if (stream == NULL || out_frames == NULL || (frames == NULL && frame_count != 0) || stream->kind != SLA_DEVICE_CAPTURE)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    *out_frames = sla_ring_read(&stream->ring, frames, frame_count);
    return SLA_OK;
}

sla_result sla_stream_available(sla_stream* stream, uint32_t* out_frames)
{
    if (stream == NULL || out_frames == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    *out_frames = stream->kind == SLA_DEVICE_PLAYBACK ? sla_ring_writable(&stream->ring) : sla_ring_readable(&stream->ring);
    return SLA_OK;
}

sla_result sla_stream_stats_get(sla_stream* stream, sla_stream_stats* out_stats)
{
    if (stream == NULL || out_stats == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    out_stats->frames_transferred = atomic_load_explicit(&stream->frames_transferred, memory_order_relaxed);
    out_stats->underruns = atomic_load_explicit(&stream->underruns, memory_order_relaxed);
    out_stats->overruns = atomic_load_explicit(&stream->overruns, memory_order_relaxed);
    return SLA_OK;
}

sla_result sla_stream_format(const sla_stream* stream, uint32_t* out_sample_rate, uint32_t* out_channels)
{
    if (stream == NULL || out_sample_rate == NULL || out_channels == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    *out_sample_rate = stream->sample_rate;
    *out_channels = stream->channels;
    return SLA_OK;
}
