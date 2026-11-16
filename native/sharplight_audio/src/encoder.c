#include "sharplight_audio.h"

#include "internal.h"

#include <math.h>
#include <ogg/ogg.h>
#include <opus.h>
#include <stdlib.h>
#include <string.h>

#define OPUS_FRAME_MS_DIVISOR 50u /* 20 ms frames */
#define OPUS_MAX_PACKET 4000
#define WAV_HEADER_MAX 64u

struct sla_encoder
{
    sla_encode_format format;
    uint32_t sample_rate;
    uint32_t channels;
    int finished;

    uint8_t* buffer;
    size_t size;
    size_t capacity;
    int out_of_memory;

    uint64_t frames_written;

    /* Opus */
    OpusEncoder* opus;
    ogg_stream_state ogg;
    int ogg_initialized;
    uint32_t frame_size; /* samples per channel per packet */
    float* pending;
    uint32_t pending_frames;
    uint64_t packet_number;
    uint32_t pre_skip_48k;
    uint32_t units_per_sample; /* 48000 / sample_rate */
};

static int append(sla_encoder* encoder, const void* data, size_t length)
{
    if (encoder->out_of_memory)
    {
        return 0;
    }

    if (encoder->size + length > encoder->capacity)
    {
        size_t capacity = encoder->capacity == 0 ? 65536 : encoder->capacity;
        while (capacity < encoder->size + length)
        {
            capacity *= 2;
        }

        uint8_t* grown = (uint8_t*)realloc(encoder->buffer, capacity);
        if (grown == NULL)
        {
            encoder->out_of_memory = 1;
            return 0;
        }

        encoder->buffer = grown;
        encoder->capacity = capacity;
    }

    memcpy(encoder->buffer + encoder->size, data, length);
    encoder->size += length;
    return 1;
}

static void put_u16(uint8_t* destination, uint32_t value)
{
    destination[0] = (uint8_t)(value & 0xFF);
    destination[1] = (uint8_t)((value >> 8) & 0xFF);
}

static void put_u32(uint8_t* destination, uint32_t value)
{
    destination[0] = (uint8_t)(value & 0xFF);
    destination[1] = (uint8_t)((value >> 8) & 0xFF);
    destination[2] = (uint8_t)((value >> 16) & 0xFF);
    destination[3] = (uint8_t)((value >> 24) & 0xFF);
}

static uint32_t wav_bytes_per_sample(sla_encode_format format)
{
    switch (format)
    {
        case SLA_ENCODE_WAV_PCM16:
            return 2;
        case SLA_ENCODE_WAV_PCM24:
            return 3;
        default:
            return 4;
    }
}

/* Reserves the header region; it is filled in by wav_finish. */
static int wav_begin(sla_encoder* encoder)
{
    uint8_t placeholder[WAV_HEADER_MAX];
    memset(placeholder, 0, sizeof(placeholder));
    return append(encoder, placeholder, 44);
}

/* Scales a sample by 2^(bits-1), rounds to nearest even and clamps to the signed range (-scale .. scale-1). */
static int32_t quantize(float sample, float scale)
{
    float limit = scale;
    float scaled = sample * scale;
    if (scaled != scaled)
    {
        return 0;
    }

    if (scaled > limit)
    {
        scaled = limit;
    }

    if (scaled < -limit)
    {
        scaled = -limit;
    }

    int32_t value = (int32_t)lrintf(scaled);
    int32_t maximum = (int32_t)limit - 1;
    return value > maximum ? maximum : value;
}

static void wav_write_samples(sla_encoder* encoder, const float* frames, uint64_t frame_count)
{
    uint8_t chunk[4096];
    uint32_t bytes = wav_bytes_per_sample(encoder->format);
    uint64_t total = frame_count * encoder->channels;
    size_t used = 0;
    for (uint64_t i = 0; i < total; i++)
    {
        float sample = frames[i];
        if (sample != sample)
        {
            sample = 0.0f;
        }

        if (used + 4 > sizeof(chunk))
        {
            append(encoder, chunk, used);
            used = 0;
        }

        if (encoder->format == SLA_ENCODE_WAV_FLOAT32)
        {
            uint32_t bits;
            memcpy(&bits, &sample, sizeof(bits));
            put_u32(chunk + used, bits);
        }
        else if (encoder->format == SLA_ENCODE_WAV_PCM16)
        {
            /* Same rule as Core's WavFile: scale by 2^(bits-1) (the divisor reading uses), round to nearest even, clamp. */
            put_u16(chunk + used, (uint32_t)(int16_t)quantize(sample, 32768.0f));
        }
        else
        {
            uint32_t bits = (uint32_t)quantize(sample, 8388608.0f);
            chunk[used] = (uint8_t)(bits & 0xFF);
            chunk[used + 1] = (uint8_t)((bits >> 8) & 0xFF);
            chunk[used + 2] = (uint8_t)((bits >> 16) & 0xFF);
        }

        used += bytes;
    }

    append(encoder, chunk, used);
}

static sla_result wav_finish(sla_encoder* encoder)
{
    int is_float = encoder->format == SLA_ENCODE_WAV_FLOAT32;
    size_t header_size = 44;
    uint64_t data_size = (uint64_t)(encoder->size - header_size);
    if (data_size > 0xFFFFFFFFu - 64u)
    {
        return SLA_ERR_UNSUPPORTED;
    }

    uint32_t bytes = wav_bytes_per_sample(encoder->format);
    uint8_t* header = encoder->buffer;
    memcpy(header, "RIFF", 4);
    put_u32(header + 4, (uint32_t)(encoder->size - 8));
    memcpy(header + 8, "WAVE", 4);
    memcpy(header + 12, "fmt ", 4);
    put_u32(header + 16, 16);
    put_u16(header + 20, is_float ? 3 : 1);
    put_u16(header + 22, encoder->channels);
    put_u32(header + 24, encoder->sample_rate);
    put_u32(header + 28, encoder->sample_rate * encoder->channels * bytes);
    put_u16(header + 32, encoder->channels * bytes);
    put_u16(header + 34, bytes * 8);
    /* Same 44-byte layout as Core's WavFile for every format (no fact chunk for float). */
    memcpy(header + 36, "data", 4);
    put_u32(header + 40, (uint32_t)data_size);
    return SLA_OK;
}

static int emit_pages(sla_encoder* encoder, int flush)
{
    ogg_page page;
    for (;;)
    {
        int got = flush ? ogg_stream_flush(&encoder->ogg, &page) : ogg_stream_pageout(&encoder->ogg, &page);
        if (got == 0)
        {
            return 1;
        }

        if (!append(encoder, page.header, (size_t)page.header_len) || !append(encoder, page.body, (size_t)page.body_len))
        {
            return 0;
        }
    }
}

static int opus_begin(sla_encoder* encoder, int32_t bitrate)
{
    int error = 0;
    encoder->opus = opus_encoder_create((opus_int32)encoder->sample_rate, (int)encoder->channels, OPUS_APPLICATION_AUDIO, &error);
    if (encoder->opus == NULL || error != OPUS_OK)
    {
        return 0;
    }

    opus_int32 target = bitrate > 0 ? bitrate : (opus_int32)(64000 * encoder->channels);
    opus_encoder_ctl(encoder->opus, OPUS_SET_BITRATE(target));

    opus_int32 lookahead = 0;
    opus_encoder_ctl(encoder->opus, OPUS_GET_LOOKAHEAD(&lookahead));
    encoder->units_per_sample = 48000u / encoder->sample_rate;
    encoder->pre_skip_48k = (uint32_t)lookahead * encoder->units_per_sample;
    encoder->frame_size = encoder->sample_rate / OPUS_FRAME_MS_DIVISOR;
    encoder->pending = (float*)calloc((size_t)encoder->frame_size * encoder->channels, sizeof(float));
    if (encoder->pending == NULL)
    {
        return 0;
    }

    if (ogg_stream_init(&encoder->ogg, 0x534C4F47) != 0)
    {
        return 0;
    }

    encoder->ogg_initialized = 1;

    uint8_t head[19];
    memcpy(head, "OpusHead", 8);
    head[8] = 1;
    head[9] = (uint8_t)encoder->channels;
    put_u16(head + 10, encoder->pre_skip_48k);
    put_u32(head + 12, encoder->sample_rate);
    put_u16(head + 16, 0);
    head[18] = 0;

    static const char vendor[] = "Sharplight";
    uint8_t tags[8 + 4 + sizeof(vendor) - 1 + 4];
    memcpy(tags, "OpusTags", 8);
    put_u32(tags + 8, (uint32_t)(sizeof(vendor) - 1));
    memcpy(tags + 12, vendor, sizeof(vendor) - 1);
    put_u32(tags + 12 + sizeof(vendor) - 1, 0);

    ogg_packet packet;
    memset(&packet, 0, sizeof(packet));
    packet.packet = head;
    packet.bytes = sizeof(head);
    packet.b_o_s = 1;
    packet.granulepos = 0;
    packet.packetno = (ogg_int64_t)encoder->packet_number++;
    ogg_stream_packetin(&encoder->ogg, &packet);
    if (!emit_pages(encoder, 1))
    {
        return 0;
    }

    memset(&packet, 0, sizeof(packet));
    packet.packet = tags;
    packet.bytes = sizeof(tags);
    packet.granulepos = 0;
    packet.packetno = (ogg_int64_t)encoder->packet_number++;
    ogg_stream_packetin(&encoder->ogg, &packet);
    return emit_pages(encoder, 1);
}

/* Encodes the pending (possibly zero padded) frame as one packet. */
static sla_result opus_encode_pending(sla_encoder* encoder, int last)
{
    uint8_t packet_data[OPUS_MAX_PACKET];
    opus_int32 bytes = opus_encode_float(encoder->opus, encoder->pending, (int)encoder->frame_size, packet_data, OPUS_MAX_PACKET);
    if (bytes < 0)
    {
        return SLA_ERR_UNSUPPORTED;
    }

    /* The granule position counts decoder output samples at 48 kHz including the pre-skip: the end of this packet, but never past
       the real end of the audio (the last packets only flush the encoder's lookahead and padding). */
    uint64_t packet_end = (uint64_t)encoder->frame_size * encoder->units_per_sample * (encoder->packet_number - 1);
    uint64_t audio_end = encoder->frames_written * encoder->units_per_sample + encoder->pre_skip_48k;
    if (last || packet_end > audio_end)
    {
        packet_end = audio_end;
    }

    ogg_packet packet;
    memset(&packet, 0, sizeof(packet));
    packet.packet = packet_data;
    packet.bytes = bytes;
    packet.e_o_s = last ? 1 : 0;
    packet.granulepos = (ogg_int64_t)packet_end;
    packet.packetno = (ogg_int64_t)encoder->packet_number++;
    ogg_stream_packetin(&encoder->ogg, &packet);
    if (!emit_pages(encoder, last))
    {
        return SLA_ERR_OUT_OF_MEMORY;
    }

    return SLA_OK;
}

static sla_result opus_write(sla_encoder* encoder, const float* frames, uint64_t frame_count)
{
    uint32_t channels = encoder->channels;
    uint64_t offset = 0;
    while (offset < frame_count)
    {
        if (encoder->pending_frames == encoder->frame_size)
        {
            /* The final packet is produced by finish (so it can carry e_o_s); a full buffer here means more data follows. */
            sla_result result = opus_encode_pending(encoder, 0);
            if (result != SLA_OK)
            {
                return result;
            }

            encoder->pending_frames = 0;
        }

        uint32_t room = encoder->frame_size - encoder->pending_frames;
        uint64_t remaining = frame_count - offset;
        uint32_t take = remaining < room ? (uint32_t)remaining : room;
        for (uint32_t i = 0; i < take * channels; i++)
        {
            float sample = frames[offset * channels + i];
            encoder->pending[(size_t)encoder->pending_frames * channels + i] = sample == sample ? sample : 0.0f;
        }

        encoder->pending_frames += take;
        offset += take;
        encoder->frames_written += take;
    }

    return SLA_OK;
}

sla_result sla_encoder_open(sla_encode_format format, uint32_t sample_rate, uint32_t channels, int32_t bitrate, sla_encoder** out_encoder)
{
    if (out_encoder == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    *out_encoder = NULL;
    if (channels == 0 || channels > 8 || sample_rate < 1000 || sample_rate > 384000)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    int is_wav = format == SLA_ENCODE_WAV_PCM16 || format == SLA_ENCODE_WAV_PCM24 || format == SLA_ENCODE_WAV_FLOAT32;
    if (!is_wav && format != SLA_ENCODE_OGG_OPUS)
    {
        return SLA_ERR_UNSUPPORTED;
    }

    if (format == SLA_ENCODE_OGG_OPUS)
    {
        int rate_ok = sample_rate == 8000 || sample_rate == 12000 || sample_rate == 16000 || sample_rate == 24000 || sample_rate == 48000;
        if (!rate_ok || channels > 2)
        {
            return SLA_ERR_UNSUPPORTED;
        }
    }

    sla_encoder* encoder = (sla_encoder*)calloc(1, sizeof(*encoder));
    if (encoder == NULL)
    {
        return SLA_ERR_OUT_OF_MEMORY;
    }

    encoder->format = format;
    encoder->sample_rate = sample_rate;
    encoder->channels = channels;
    int ok = is_wav ? wav_begin(encoder) : opus_begin(encoder, bitrate);
    if (!ok)
    {
        sla_encoder_close(encoder);
        return SLA_ERR_OUT_OF_MEMORY;
    }

    *out_encoder = encoder;
    return SLA_OK;
}

sla_result sla_encoder_write(sla_encoder* encoder, const float* frames, uint64_t frame_count)
{
    if (encoder == NULL || (frames == NULL && frame_count != 0))
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    if (encoder->finished)
    {
        return SLA_ERR_INVALID_STATE;
    }

    if (frame_count == 0)
    {
        return SLA_OK;
    }

    sla_result result = SLA_OK;
    if (encoder->format == SLA_ENCODE_OGG_OPUS)
    {
        result = opus_write(encoder, frames, frame_count);
    }
    else
    {
        wav_write_samples(encoder, frames, frame_count);
        encoder->frames_written += frame_count;
    }

    return encoder->out_of_memory ? SLA_ERR_OUT_OF_MEMORY : result;
}

sla_result sla_encoder_finish(sla_encoder* encoder)
{
    if (encoder == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    if (encoder->finished)
    {
        return SLA_OK;
    }

    encoder->finished = 1;
    if (encoder->out_of_memory)
    {
        return SLA_ERR_OUT_OF_MEMORY;
    }

    if (encoder->format != SLA_ENCODE_OGG_OPUS)
    {
        return wav_finish(encoder);
    }

    /* The encoder delays its output by the lookahead (the pre-skip), so enough silence is encoded after the last real sample to
       push every real sample through. The final granule position records the real length so decoders trim the padding. */
    uint32_t channels = encoder->channels;
    uint64_t lookahead = encoder->pre_skip_48k / encoder->units_per_sample;
    uint64_t needed_packets = (encoder->frames_written + lookahead + encoder->frame_size - 1) / encoder->frame_size;
    if (needed_packets == 0)
    {
        needed_packets = 1;
    }

    memset(encoder->pending + (size_t)encoder->pending_frames * channels, 0, (size_t)(encoder->frame_size - encoder->pending_frames) * channels * sizeof(float));
    while ((encoder->packet_number - 2) + 1 < needed_packets)
    {
        sla_result result = opus_encode_pending(encoder, 0);
        if (result != SLA_OK)
        {
            return result;
        }

        memset(encoder->pending, 0, (size_t)encoder->frame_size * channels * sizeof(float));
    }

    return opus_encode_pending(encoder, 1);
}

sla_result sla_encoder_data(const sla_encoder* encoder, const uint8_t** out_data, size_t* out_size)
{
    if (encoder == NULL || out_data == NULL || out_size == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    if (!encoder->finished)
    {
        return SLA_ERR_INVALID_STATE;
    }

    *out_data = encoder->buffer;
    *out_size = encoder->size;
    return SLA_OK;
}

sla_result sla_encoder_save(sla_encoder* encoder, const char* utf8_path)
{
    if (encoder == NULL || utf8_path == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    sla_result result = sla_encoder_finish(encoder);
    if (result != SLA_OK)
    {
        return result;
    }

    FILE* file = sla_fopen_utf8(utf8_path, "wb");
    if (file == NULL)
    {
        return SLA_ERR_IO;
    }

    size_t written = fwrite(encoder->buffer, 1, encoder->size, file);
    int closed = fclose(file);
    return written == encoder->size && closed == 0 ? SLA_OK : SLA_ERR_IO;
}

void sla_encoder_close(sla_encoder* encoder)
{
    if (encoder == NULL)
    {
        return;
    }

    if (encoder->opus != NULL)
    {
        opus_encoder_destroy(encoder->opus);
    }

    if (encoder->ogg_initialized)
    {
        ogg_stream_clear(&encoder->ogg);
    }

    free(encoder->pending);
    free(encoder->buffer);
    free(encoder);
}
