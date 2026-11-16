/* Native smoke, round-trip and fuzz tests of the Sharplight audio library. Usage: sla_test <fixtures-directory> [fuzz-iterations] */
#define _POSIX_C_SOURCE 200809L
#include "sharplight_audio.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_failures = 0;

#define CHECK(condition)                                                                    \
    do                                                                                      \
    {                                                                                       \
        if (!(condition))                                                                   \
        {                                                                                   \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);            \
            g_failures++;                                                                   \
        }                                                                                   \
    } while (0)

static void sleep_ms(long milliseconds)
{
    struct timespec span;
    span.tv_sec = milliseconds / 1000;
    span.tv_nsec = (milliseconds % 1000) * 1000000L;
    nanosleep(&span, NULL);
}

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;

static uint64_t next_random(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

static uint8_t* read_file(const char* path, size_t* size)
{
    FILE* file = fopen(path, "rb");
    if (file == NULL)
    {
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc((size_t)length);
    if (data != NULL && fread(data, 1, (size_t)length, file) != (size_t)length)
    {
        free(data);
        data = NULL;
    }

    fclose(file);
    *size = (size_t)length;
    return data;
}

static float* make_tone(uint32_t sample_rate, uint32_t channels, uint32_t frames)
{
    float* samples = (float*)malloc((size_t)frames * channels * sizeof(float));
    for (uint32_t i = 0; i < frames; i++)
    {
        for (uint32_t c = 0; c < channels; c++)
        {
            samples[(size_t)i * channels + c] = (float)(0.5 * sin(2.0 * 3.14159265358979 * (440.0 + 220.0 * c) * i / sample_rate));
        }
    }

    return samples;
}

static void test_version(void)
{
    CHECK(sla_abi_version() == SLA_ABI_VERSION);
    CHECK(sla_version() > 0);
    CHECK(strstr(sla_version_string(), "miniaudio") != NULL);
    CHECK(strstr(sla_version_string(), "libopus") != NULL);
}

static void test_null_playback_and_capture(void)
{
    sla_context* context = NULL;
    CHECK(sla_context_create(1, &context) == SLA_OK);
    if (context == NULL)
    {
        return;
    }

    CHECK(strcmp(sla_context_backend_name(context), "Null") == 0);
    int32_t count = 0;
    CHECK(sla_device_count(context, SLA_DEVICE_PLAYBACK, &count) == SLA_OK);
    CHECK(count >= 1);
    char name[128];
    int32_t is_default = 0;
    CHECK(sla_device_info(context, SLA_DEVICE_PLAYBACK, 0, name, sizeof(name), &is_default) == SLA_OK);
    CHECK(name[0] != '\0');
    CHECK(sla_device_info(context, SLA_DEVICE_PLAYBACK, 9999, name, sizeof(name), &is_default) == SLA_ERR_INVALID_ARGUMENT);

    sla_stream* playback = NULL;
    CHECK(sla_stream_open(context, SLA_DEVICE_PLAYBACK, -1, 48000, 2, 24000, &playback) == SLA_OK);
    float* tone = make_tone(48000, 2, 4800);
    uint32_t accepted = 0;
    CHECK(sla_stream_write(playback, tone, 4800, &accepted) == SLA_OK);
    CHECK(accepted == 4800);
    CHECK(sla_stream_start(playback) == SLA_OK);
    sleep_ms(400);
    sla_stream_stats stats;
    CHECK(sla_stream_stats_get(playback, &stats) == SLA_OK);
    CHECK(stats.frames_transferred == 4800);
    CHECK(stats.underruns > 0); /* the queue ran dry after 100 ms */
    uint32_t available = 0;
    CHECK(sla_stream_available(playback, &available) == SLA_OK);
    CHECK(available >= 24000 - 4800);
    CHECK(sla_stream_stop(playback) == SLA_OK);
    sla_stream_close(playback);
    free(tone);

    sla_stream* capture = NULL;
    CHECK(sla_stream_open(context, SLA_DEVICE_CAPTURE, -1, 16000, 1, 1024, &capture) == SLA_OK);
    CHECK(sla_stream_start(capture) == SLA_OK);
    sleep_ms(600);
    float buffer[512];
    uint32_t read_total = 0;
    uint32_t read = 0;
    while (sla_stream_read(capture, buffer, 512, &read) == SLA_OK && read > 0)
    {
        read_total += read;
    }

    CHECK(read_total > 0);
    CHECK(sla_stream_stats_get(capture, &stats) == SLA_OK);
    CHECK(stats.frames_transferred >= read_total);
    CHECK(sla_stream_read(capture, buffer, 512, &read) == SLA_OK);
    sla_stream_close(capture);

    CHECK(sla_stream_open(context, SLA_DEVICE_PLAYBACK, -1, 48000, 0, 100, &playback) == SLA_ERR_INVALID_ARGUMENT);
    sla_context_destroy(context);
}

static double snr_db(const float* reference, const float* actual, size_t count)
{
    double signal = 0.0;
    double noise = 0.0;
    for (size_t i = 0; i < count; i++)
    {
        double difference = (double)actual[i] - (double)reference[i];
        signal += (double)reference[i] * (double)reference[i];
        noise += difference * difference;
    }

    return noise == 0.0 ? 200.0 : 10.0 * log10(signal / noise);
}

static size_t decode_all(sla_decoder* decoder, float** samples, sla_audio_info* info)
{
    CHECK(sla_decoder_info(decoder, info) == SLA_OK);
    size_t capacity = 1 << 16;
    size_t frames = 0;
    float* data = (float*)malloc(capacity * info->channels * sizeof(float));
    for (;;)
    {
        if (frames + 4096 > capacity)
        {
            capacity *= 2;
            data = (float*)realloc(data, capacity * info->channels * sizeof(float));
        }

        uint64_t read = 0;
        if (sla_decoder_read(decoder, data + frames * info->channels, 4096, &read) != SLA_OK || read == 0)
        {
            break;
        }

        frames += (size_t)read;
    }

    *samples = data;
    return frames;
}

static void test_wav_round_trip(sla_encode_format format, double minimum_snr)
{
    const uint32_t rate = 22050;
    const uint32_t frames = 5000;
    float* tone = make_tone(rate, 2, frames);
    sla_encoder* encoder = NULL;
    CHECK(sla_encoder_open(format, rate, 2, 0, &encoder) == SLA_OK);
    CHECK(sla_encoder_write(encoder, tone, 1234) == SLA_OK);
    CHECK(sla_encoder_write(encoder, tone + 1234 * 2, frames - 1234) == SLA_OK);
    CHECK(sla_encoder_finish(encoder) == SLA_OK);
    CHECK(sla_encoder_write(encoder, tone, 1) == SLA_ERR_INVALID_STATE);
    const uint8_t* data = NULL;
    size_t size = 0;
    CHECK(sla_encoder_data(encoder, &data, &size) == SLA_OK);

    sla_decoder* decoder = NULL;
    CHECK(sla_decoder_open_memory(data, size, &decoder) == SLA_OK);
    if (decoder != NULL)
    {
        float* decoded = NULL;
        sla_audio_info info;
        size_t decoded_frames = decode_all(decoder, &decoded, &info);
        CHECK(info.codec == SLA_CODEC_WAV);
        CHECK(info.sample_rate == rate && info.channels == 2);
        CHECK(info.total_frames == frames);
        CHECK(decoded_frames == frames);
        CHECK(snr_db(tone, decoded, (size_t)decoded_frames * 2) >= minimum_snr);
        if (format == SLA_ENCODE_WAV_FLOAT32)
        {
            CHECK(memcmp(tone, decoded, (size_t)frames * 2 * sizeof(float)) == 0);
        }

        CHECK(sla_decoder_seek(decoder, 100) == SLA_OK);
        uint64_t read = 0;
        float one[2];
        CHECK(sla_decoder_read(decoder, one, 1, &read) == SLA_OK && read == 1);
        free(decoded);
        sla_decoder_close(decoder);
    }

    sla_encoder_close(encoder);
    free(tone);
}

static void test_opus_round_trip(void)
{
    const uint32_t rate = 48000;
    const uint32_t frames = 48000;
    float* tone = make_tone(rate, 2, frames);
    sla_encoder* encoder = NULL;
    CHECK(sla_encoder_open(SLA_ENCODE_OGG_OPUS, rate, 2, 96000, &encoder) == SLA_OK);
    for (uint32_t offset = 0; offset < frames; offset += 777)
    {
        uint32_t chunk = frames - offset < 777 ? frames - offset : 777;
        CHECK(sla_encoder_write(encoder, tone + (size_t)offset * 2, chunk) == SLA_OK);
    }

    CHECK(sla_encoder_finish(encoder) == SLA_OK);
    const uint8_t* data = NULL;
    size_t size = 0;
    CHECK(sla_encoder_data(encoder, &data, &size) == SLA_OK);
    CHECK(size > 1000 && size < 40000);

    sla_decoder* decoder = NULL;
    CHECK(sla_decoder_open_memory(data, size, &decoder) == SLA_OK);
    if (decoder != NULL)
    {
        float* decoded = NULL;
        sla_audio_info info;
        size_t decoded_frames = decode_all(decoder, &decoded, &info);
        CHECK(info.codec == SLA_CODEC_OPUS);
        CHECK(info.sample_rate == 48000 && info.channels == 2);
        if (decoded_frames != frames)
        {
            fprintf(stderr, "opus decoded %zu frames, expected %u\n", decoded_frames, frames);
        }

        CHECK(decoded_frames == frames); /* pre-skip and end padding are trimmed */
        if (decoded_frames == frames)
        {
            CHECK(snr_db(tone, decoded, (size_t)frames * 2) > 12.0);
        }

        free(decoded);
        sla_decoder_close(decoder);
    }

    sla_encoder_close(encoder);
    free(tone);

    CHECK(sla_encoder_open(SLA_ENCODE_OGG_OPUS, 44100, 2, 0, &encoder) == SLA_ERR_UNSUPPORTED);
    CHECK(sla_encoder_open(SLA_ENCODE_OGG_OPUS, 48000, 3, 0, &encoder) == SLA_ERR_UNSUPPORTED);
}

static const char* const k_fixtures[] = { "tone_s16.wav", "tone_s24.wav", "tone_f32.wav", "tone.flac", "tone.mp3", "tone.ogg", "tone.opus" };
static const sla_codec k_codecs[] = { SLA_CODEC_WAV, SLA_CODEC_WAV, SLA_CODEC_WAV, SLA_CODEC_FLAC, SLA_CODEC_MP3, SLA_CODEC_VORBIS, SLA_CODEC_OPUS };

static void test_fixtures(const char* directory)
{
    for (size_t i = 0; i < sizeof(k_fixtures) / sizeof(k_fixtures[0]); i++)
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", directory, k_fixtures[i]);
        sla_decoder* decoder = NULL;
        sla_result result = sla_decoder_open_file(path, &decoder);
        if (result != SLA_OK)
        {
            fprintf(stderr, "FAIL open %s: %s\n", path, sla_error_string(result));
            g_failures++;
            continue;
        }

        float* samples = NULL;
        sla_audio_info info;
        size_t frames = decode_all(decoder, &samples, &info);
        CHECK(info.codec == (int32_t)k_codecs[i]);
        CHECK(info.channels == 2);
        CHECK(frames >= 7900 && frames <= 9500 + 24000); /* 0.5 s at 16 kHz, or 48 kHz for Opus */
        printf("%-14s codec=%d rate=%u channels=%u frames=%zu\n", k_fixtures[i], info.codec, info.sample_rate, info.channels, frames);
        free(samples);
        sla_decoder_close(decoder);
    }

    sla_decoder* missing = NULL;
    CHECK(sla_decoder_open_file("/nonexistent/file.wav", &missing) == SLA_ERR_IO);
    uint8_t garbage[64];
    memset(garbage, 0x5A, sizeof(garbage));
    CHECK(sla_decoder_open_memory(garbage, sizeof(garbage), &missing) != SLA_OK);
}

/* Recomputes the CRC of every Ogg page, so that mutated payloads get past the container check and reach the codecs. */
static void fix_ogg_checksums(uint8_t* data, size_t length)
{
    size_t position = 0;
    while (position + 27 <= length && memcmp(data + position, "OggS", 4) == 0)
    {
        size_t segments = data[position + 26];
        if (position + 27 + segments > length)
        {
            return;
        }

        size_t body = 0;
        for (size_t i = 0; i < segments; i++)
        {
            body += data[position + 27 + i];
        }

        size_t page = 27 + segments + body;
        if (position + page > length)
        {
            return;
        }

        memset(data + position + 22, 0, 4);
        uint32_t crc = 0;
        for (size_t i = 0; i < page; i++)
        {
            crc ^= (uint32_t)data[position + i] << 24;
            for (int bit = 0; bit < 8; bit++)
            {
                crc = (crc & 0x80000000u) != 0 ? (crc << 1) ^ 0x04C11DB7u : crc << 1;
            }
        }

        data[position + 22] = (uint8_t)(crc & 0xFF);
        data[position + 23] = (uint8_t)((crc >> 8) & 0xFF);
        data[position + 24] = (uint8_t)((crc >> 16) & 0xFF);
        data[position + 25] = (uint8_t)((crc >> 24) & 0xFF);
        position += page;
    }
}

/* Feeds mutated copies of the fixtures (flipped bytes, truncation, junk tails) to the decoders; must neither crash nor hang. */
static void test_fuzz(const char* directory, int iterations)
{
    for (size_t i = 0; i < sizeof(k_fixtures) / sizeof(k_fixtures[0]); i++)
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", directory, k_fixtures[i]);
        size_t size = 0;
        uint8_t* original = read_file(path, &size);
        if (original == NULL)
        {
            g_failures++;
            continue;
        }

        uint8_t* mutated = (uint8_t*)malloc(size + 64);
        int opened = 0;
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            memcpy(mutated, original, size);
            size_t length = size;
            int flips = 1 + (int)(next_random() % 8);
            for (int flip = 0; flip < flips; flip++)
            {
                size_t position = (next_random() % 4 == 0) ? (size_t)(next_random() % (size < 256 ? size : 256)) : (size_t)(next_random() % size);
                mutated[position] = (uint8_t)next_random();
            }

            if (next_random() % 5 == 0)
            {
                length = 1 + (size_t)(next_random() % size);
            }

            if (next_random() % 4 != 0)
            {
                fix_ogg_checksums(mutated, length);
            }

            sla_decoder* decoder = NULL;
            if (sla_decoder_open_memory(mutated, length, &decoder) == SLA_OK)
            {
                opened++;
                sla_audio_info info;
                sla_decoder_info(decoder, &info);
                float buffer[4096 * 8];
                uint64_t total = 0;
                uint64_t read = 0;
                uint64_t per_call = info.channels > 0 && info.channels <= 8 ? 4096 : 0;
                while (per_call > 0 && sla_decoder_read(decoder, buffer, per_call, &read) == SLA_OK && read > 0 && total < (1u << 22))
                {
                    total += read;
                }

                sla_decoder_seek(decoder, next_random() % 20000);
                sla_decoder_read(decoder, buffer, per_call, &read);
                sla_decoder_close(decoder);
            }
        }

        printf("fuzz %-14s %d iterations, %d still opened\n", k_fixtures[i], iterations, opened);
        free(mutated);
        free(original);
    }
}

int main(int argc, char** argv)
{
    const char* fixtures = argc > 1 ? argv[1] : ".";
    int iterations = argc > 2 ? atoi(argv[2]) : 200;
    printf("%s\n", sla_version_string());
    test_version();
    test_null_playback_and_capture();
    test_wav_round_trip(SLA_ENCODE_WAV_PCM16, 80.0);
    test_wav_round_trip(SLA_ENCODE_WAV_PCM24, 120.0);
    test_wav_round_trip(SLA_ENCODE_WAV_FLOAT32, 200.0);
    test_opus_round_trip();
    test_fixtures(fixtures);
    test_fuzz(fixtures, iterations);
    if (g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }

    printf("all native tests passed\n");
    return 0;
}
