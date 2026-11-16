#include "sharplight_audio.h"

#include "miniaudio.h"
#include "decoders/libopus/miniaudio_libopus.h"
#include "decoders/libvorbis/miniaudio_libvorbis.h"

#include "internal.h"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <stdlib.h>
#include <string.h>

struct sla_decoder
{
    ma_decoder decoder;
    uint8_t* memory; /* owned copy for memory sources */
    sla_audio_info info;
};

static ma_decoding_backend_vtable* g_backends[2];

static int32_t sniff_codec(const uint8_t* data, size_t size)
{
    if (size >= 12 && memcmp(data, "RIFF", 4) == 0)
    {
        return SLA_CODEC_WAV;
    }

    if (size >= 4 && memcmp(data, "RF64", 4) == 0)
    {
        return SLA_CODEC_WAV;
    }

    if (size >= 4 && memcmp(data, "fLaC", 4) == 0)
    {
        return SLA_CODEC_FLAC;
    }

    if (size >= 4 && memcmp(data, "OggS", 4) == 0)
    {
        size_t limit = size < 128 ? size : 128;
        for (size_t i = 27; i + 8 <= limit; i++)
        {
            if (memcmp(data + i, "OpusHead", 8) == 0)
            {
                return SLA_CODEC_OPUS;
            }

            if (data[i] == 0x01 && memcmp(data + i + 1, "vorbis", 6) == 0)
            {
                return SLA_CODEC_VORBIS;
            }
        }

        return SLA_CODEC_UNKNOWN;
    }

    if (size >= 3 && memcmp(data, "ID3", 3) == 0)
    {
        return SLA_CODEC_MP3;
    }

    if (size >= 2 && data[0] == 0xFF && (data[1] & 0xE0) == 0xE0)
    {
        return SLA_CODEC_MP3;
    }

    return SLA_CODEC_UNKNOWN;
}

static sla_result finish_open(sla_decoder* decoder, ma_result init_result, const uint8_t* header, size_t header_size, sla_decoder** out_decoder)
{
    if (init_result != MA_SUCCESS)
    {
        free(decoder->memory);
        free(decoder);
        return init_result == MA_DOES_NOT_EXIST || init_result == MA_ACCESS_DENIED ? SLA_ERR_IO : SLA_ERR_UNSUPPORTED;
    }

    ma_format format;
    ma_uint32 channels;
    ma_uint32 sample_rate;
    ma_uint64 length = 0;
    if (ma_decoder_get_data_format(&decoder->decoder, &format, &channels, &sample_rate, NULL, 0) != MA_SUCCESS || channels == 0 || sample_rate == 0)
    {
        ma_decoder_uninit(&decoder->decoder);
        free(decoder->memory);
        free(decoder);
        return SLA_ERR_DECODE;
    }

    if (ma_decoder_get_length_in_pcm_frames(&decoder->decoder, &length) != MA_SUCCESS)
    {
        length = 0;
    }

    decoder->info.sample_rate = sample_rate;
    decoder->info.channels = channels;
    decoder->info.total_frames = length > 0 ? (int64_t)length : -1;
    decoder->info.codec = sniff_codec(header, header_size);
    *out_decoder = decoder;
    return SLA_OK;
}

static ma_decoder_config make_config(void)
{
    g_backends[0] = ma_decoding_backend_libopus;
    g_backends[1] = ma_decoding_backend_libvorbis;
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 0, 0);
    config.ppCustomBackendVTables = g_backends;
    config.customBackendCount = 2;
    return config;
}

sla_result sla_decoder_open_file(const char* utf8_path, sla_decoder** out_decoder)
{
    if (utf8_path == NULL || out_decoder == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    *out_decoder = NULL;
    uint8_t header[128];
    size_t header_size = 0;
    FILE* file = sla_fopen_utf8(utf8_path, "rb");
    if (file == NULL)
    {
        return SLA_ERR_IO;
    }

    header_size = fread(header, 1, sizeof(header), file);
    fclose(file);

    sla_decoder* decoder = (sla_decoder*)calloc(1, sizeof(*decoder));
    if (decoder == NULL)
    {
        return SLA_ERR_OUT_OF_MEMORY;
    }

    ma_decoder_config config = make_config();
    ma_result result;
#if defined(_WIN32)
    wchar_t wide_path[1024];
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8_path, -1, wide_path, (int)(sizeof(wide_path) / sizeof(wide_path[0]))) == 0)
    {
        free(decoder);
        return SLA_ERR_IO;
    }

    result = ma_decoder_init_file_w(wide_path, &config, &decoder->decoder);
#else
    result = ma_decoder_init_file(utf8_path, &config, &decoder->decoder);
#endif
    return finish_open(decoder, result, header, header_size, out_decoder);
}

sla_result sla_decoder_open_memory(const void* data, size_t size, sla_decoder** out_decoder)
{
    if (data == NULL || size == 0 || out_decoder == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    *out_decoder = NULL;
    sla_decoder* decoder = (sla_decoder*)calloc(1, sizeof(*decoder));
    if (decoder == NULL)
    {
        return SLA_ERR_OUT_OF_MEMORY;
    }

    decoder->memory = (uint8_t*)malloc(size);
    if (decoder->memory == NULL)
    {
        free(decoder);
        return SLA_ERR_OUT_OF_MEMORY;
    }

    memcpy(decoder->memory, data, size);
    ma_decoder_config config = make_config();
    ma_result result = ma_decoder_init_memory(decoder->memory, size, &config, &decoder->decoder);
    return finish_open(decoder, result, decoder->memory, size < 128 ? size : 128, out_decoder);
}

sla_result sla_decoder_info(const sla_decoder* decoder, sla_audio_info* out_info)
{
    if (decoder == NULL || out_info == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    *out_info = decoder->info;
    return SLA_OK;
}

sla_result sla_decoder_read(sla_decoder* decoder, float* frames, uint64_t frame_count, uint64_t* out_frames)
{
    if (decoder == NULL || out_frames == NULL || (frames == NULL && frame_count != 0))
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    ma_uint64 read = 0;
    ma_result result = ma_decoder_read_pcm_frames(&decoder->decoder, frames, frame_count, &read);
    *out_frames = read;
    if (result == MA_SUCCESS || result == MA_AT_END)
    {
        return SLA_OK;
    }

    return read > 0 ? SLA_OK : SLA_ERR_DECODE;
}

sla_result sla_decoder_seek(sla_decoder* decoder, uint64_t frame_index)
{
    if (decoder == NULL)
    {
        return SLA_ERR_INVALID_ARGUMENT;
    }

    return ma_decoder_seek_to_pcm_frame(&decoder->decoder, frame_index) == MA_SUCCESS ? SLA_OK : SLA_ERR_UNSUPPORTED;
}

void sla_decoder_close(sla_decoder* decoder)
{
    if (decoder == NULL)
    {
        return;
    }

    ma_decoder_uninit(&decoder->decoder);
    free(decoder->memory);
    free(decoder);
}
