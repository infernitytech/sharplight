/*
 * Sharplight audio: flat C ABI over miniaudio (devices, decoding), libopus/opusfile/libogg (Opus) and libvorbis (Vorbis).
 *
 * Conventions:
 *  - All functions return an sla_result (0 = success, negative = failure) unless stated otherwise.
 *  - Handles are opaque pointers; a handle is owned by the caller until its *_close / *_destroy function is called.
 *  - Strings are UTF-8, NUL terminated. Audio samples are 32-bit float, interleaved.
 *  - No callbacks into managed code. Streams exchange data through a lock-free single-producer/single-consumer
 *    ring buffer inside the library: exactly one thread may call sla_stream_write/sla_stream_read for a given stream.
 *  - A handle must not be closed while another thread is still inside a call using it.
 */
#ifndef SHARPLIGHT_AUDIO_H
#define SHARPLIGHT_AUDIO_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define SLA_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define SLA_API __attribute__((visibility("default")))
#else
#define SLA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version; incremented when an existing function or struct changes incompatibly. */
#define SLA_ABI_VERSION 1

typedef int32_t sla_result;

enum
{
    SLA_OK = 0,
    SLA_ERR_INVALID_ARGUMENT = -1,
    SLA_ERR_OUT_OF_MEMORY = -2,
    SLA_ERR_NO_BACKEND = -3,      /* no audio backend/device available on this machine */
    SLA_ERR_IO = -4,              /* file could not be opened, read or written */
    SLA_ERR_UNSUPPORTED = -5,     /* unknown or unsupported format, sample rate or channel count */
    SLA_ERR_DECODE = -6,          /* corrupt or truncated data */
    SLA_ERR_DEVICE = -7,          /* the device could not be opened or started */
    SLA_ERR_INVALID_STATE = -8,   /* operation not valid in the current state (e.g. writing to a finished encoder) */
    SLA_ERR_BUFFER_TOO_SMALL = -9
};

typedef enum sla_device_kind
{
    SLA_DEVICE_PLAYBACK = 0,
    SLA_DEVICE_CAPTURE = 1
} sla_device_kind;

typedef enum sla_codec
{
    SLA_CODEC_UNKNOWN = 0,
    SLA_CODEC_WAV = 1,
    SLA_CODEC_FLAC = 2,
    SLA_CODEC_MP3 = 3,
    SLA_CODEC_VORBIS = 4,
    SLA_CODEC_OPUS = 5
} sla_codec;

typedef enum sla_encode_format
{
    SLA_ENCODE_WAV_PCM16 = 1,
    SLA_ENCODE_WAV_PCM24 = 2,
    SLA_ENCODE_WAV_FLOAT32 = 3,
    SLA_ENCODE_OGG_OPUS = 4
} sla_encode_format;

typedef struct sla_context sla_context;
typedef struct sla_stream sla_stream;
typedef struct sla_decoder sla_decoder;
typedef struct sla_encoder sla_encoder;

typedef struct sla_audio_info
{
    uint32_t sample_rate;
    uint32_t channels;
    int64_t total_frames;   /* -1 when unknown */
    int32_t codec;          /* sla_codec */
    int32_t reserved;
} sla_audio_info;

typedef struct sla_stream_stats
{
    uint64_t frames_transferred;  /* frames moved between ring buffer and device so far */
    uint64_t underruns;           /* playback: frames the device needed but the ring buffer did not have (silence was played) */
    uint64_t overruns;            /* capture: frames the device delivered but the ring buffer had no room for (dropped) */
} sla_stream_stats;

/* ---- Library ---- */

/* Returns (major << 16) | (minor << 8) | patch of the shim. */
SLA_API int32_t sla_version(void);
SLA_API int32_t sla_abi_version(void);
/* Human readable versions of the bundled libraries, e.g. "miniaudio 0.11.26; libopus 1.6.1". Static storage. */
SLA_API const char* sla_version_string(void);
SLA_API const char* sla_error_string(sla_result result);

/* ---- Devices ---- */

/* use_null_backend != 0 selects miniaudio's null backend (no hardware; used for tests and headless machines). */
SLA_API sla_result sla_context_create(int32_t use_null_backend, sla_context** out_context);
SLA_API void sla_context_destroy(sla_context* context);
/* Name of the active backend (ALSA, PulseAudio, WASAPI, Core Audio, Null...). Static storage. */
SLA_API const char* sla_context_backend_name(const sla_context* context);
SLA_API sla_result sla_device_count(sla_context* context, sla_device_kind kind, int32_t* out_count);
/* Copies the device name into name_buffer (truncated, always NUL terminated). */
SLA_API sla_result sla_device_info(sla_context* context, sla_device_kind kind, int32_t index, char* name_buffer, int32_t name_capacity, int32_t* out_is_default);

/* ---- Streams (playback and capture) ---- */

/* device_index < 0 selects the default device. ring_frames is the ring buffer capacity in frames (rounded up to a power of two). */
SLA_API sla_result sla_stream_open(sla_context* context, sla_device_kind kind, int32_t device_index, uint32_t sample_rate, uint32_t channels, uint32_t ring_frames, sla_stream** out_stream);
SLA_API sla_result sla_stream_start(sla_stream* stream);
SLA_API sla_result sla_stream_stop(sla_stream* stream);
SLA_API void sla_stream_close(sla_stream* stream);
/* Playback: queues up to frame_count frames; returns the number accepted (0 when the ring is full) in out_frames. */
SLA_API sla_result sla_stream_write(sla_stream* stream, const float* frames, uint32_t frame_count, uint32_t* out_frames);
/* Capture: copies up to frame_count captured frames out; returns the number copied in out_frames. */
SLA_API sla_result sla_stream_read(sla_stream* stream, float* frames, uint32_t frame_count, uint32_t* out_frames);
/* Playback: free space in frames. Capture: frames ready to read. */
SLA_API sla_result sla_stream_available(sla_stream* stream, uint32_t* out_frames);
/* Snapshot of the transfer counters (see sla_stream_stats). */
SLA_API sla_result sla_stream_stats_get(sla_stream* stream, sla_stream_stats* out_stats);
/* Actual sample rate / channels of the stream (the device may force other values; the library converts). */
SLA_API sla_result sla_stream_format(const sla_stream* stream, uint32_t* out_sample_rate, uint32_t* out_channels);

/* ---- Decoding (WAV, FLAC, MP3, Ogg Vorbis, Ogg Opus) ---- */

SLA_API sla_result sla_decoder_open_file(const char* utf8_path, sla_decoder** out_decoder);
/* The data is copied; the caller's buffer may be released immediately. */
SLA_API sla_result sla_decoder_open_memory(const void* data, size_t size, sla_decoder** out_decoder);
SLA_API sla_result sla_decoder_info(const sla_decoder* decoder, sla_audio_info* out_info);
/* Reads up to frame_count frames of float interleaved audio; out_frames == 0 at end of stream. */
SLA_API sla_result sla_decoder_read(sla_decoder* decoder, float* frames, uint64_t frame_count, uint64_t* out_frames);
SLA_API sla_result sla_decoder_seek(sla_decoder* decoder, uint64_t frame_index);
SLA_API void sla_decoder_close(sla_decoder* decoder);

/* ---- Encoding (WAV, Ogg Opus) ---- */

/*
 * bitrate is bits per second for Opus (<= 0 selects a default of 64000 per channel); ignored for WAV.
 * Opus requires sample_rate in {8000, 12000, 16000, 24000, 48000} and 1 or 2 channels.
 * The encoded stream is kept in memory until sla_encoder_finish.
 */
SLA_API sla_result sla_encoder_open(sla_encode_format format, uint32_t sample_rate, uint32_t channels, int32_t bitrate, sla_encoder** out_encoder);
SLA_API sla_result sla_encoder_write(sla_encoder* encoder, const float* frames, uint64_t frame_count);
/* Flushes and finalizes the stream; no more writes afterwards. Idempotent. */
SLA_API sla_result sla_encoder_finish(sla_encoder* encoder);
/* Valid after sla_encoder_finish, until sla_encoder_close. */
SLA_API sla_result sla_encoder_data(const sla_encoder* encoder, const uint8_t** out_data, size_t* out_size);
SLA_API sla_result sla_encoder_save(sla_encoder* encoder, const char* utf8_path);
SLA_API void sla_encoder_close(sla_encoder* encoder);

#ifdef __cplusplus
}
#endif

#endif
