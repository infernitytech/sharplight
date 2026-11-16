/* Native tests of the video library: argument validation and VP8 / VP9 encodes of a synthetic moving gradient. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sharplight_video.h"

static int failures;

#define CHECK(condition)                                                                    \
    do {                                                                                    \
        if (!(condition)) {                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, #condition, slv_last_error()); \
            failures++;                                                                     \
        }                                                                                   \
    } while (0)

static void fill_frame(uint8_t *y, uint8_t *u, uint8_t *v, int width, int height, int frame)
{
    int cw = (width + 1) / 2;
    int ch = (height + 1) / 2;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            y[row * width + col] = (uint8_t)(16 + ((col * 3 + row + frame * 4) % 200));
        }
    }
    memset(u, 128, (size_t)cw * (size_t)ch);
    memset(v, 100, (size_t)cw * (size_t)ch);
}

static void encode_clip(int codec, int width, int height, int frames, int bitrate, int threads)
{
    slv_encoder *encoder = NULL;
    CHECK(slv_encoder_create(codec, width, height, 10, 1, bitrate, 70, threads, 0, &encoder) == SLV_OK);
    if (encoder == NULL) {
        return;
    }

    int cw = (width + 1) / 2;
    int ch = (height + 1) / 2;
    uint8_t *y = malloc((size_t)width * (size_t)height);
    uint8_t *u = malloc((size_t)cw * (size_t)ch);
    uint8_t *v = malloc((size_t)cw * (size_t)ch);
    int packets = 0;
    int keyframes = 0;
    int64_t last_pts = -1;
    size_t bytes = 0;
    for (int i = 0; i <= frames; i++) {
        if (i < frames) {
            fill_frame(y, u, v, width, height, i);
            CHECK(slv_encoder_encode(encoder, y, width, u, cw, v, cw, i, 0) == SLV_OK);
        } else {
            CHECK(slv_encoder_flush(encoder) == SLV_OK);
        }

        const uint8_t *data;
        size_t size;
        int64_t pts;
        int key;
        while (slv_encoder_next_packet(encoder, &data, &size, &pts, &key) == 1) {
            CHECK(size > 0);
            CHECK(pts >= last_pts);
            if (packets == 0) {
                CHECK(key == 1);
            }
            last_pts = pts;
            keyframes += key;
            bytes += size;
            packets++;
        }
    }

    printf("codec %d %dx%d: %d packets, %d keyframes, %zu bytes\n", codec, width, height, packets, keyframes, bytes);
    CHECK(packets == frames);
    CHECK(keyframes >= 1);
    free(y);
    free(u);
    free(v);
    slv_encoder_destroy(encoder);
}

int main(void)
{
    CHECK(slv_abi_version() == SLV_ABI_VERSION);
    CHECK(strncmp(slv_version_string(), "libvpx", 6) == 0);

    slv_encoder *encoder = NULL;
    CHECK(slv_encoder_create(SLV_CODEC_VP8, 0, 10, 10, 1, 0, 50, 1, 0, &encoder) == SLV_ERR_INVALID_ARGUMENT);
    CHECK(slv_encoder_create(7, 16, 16, 10, 1, 0, 50, 1, 0, &encoder) == SLV_ERR_UNSUPPORTED);
    CHECK(slv_encoder_create(SLV_CODEC_VP9, 16, 16, 10, 1, 0, 101, 1, 0, &encoder) == SLV_ERR_INVALID_ARGUMENT);
    CHECK(encoder == NULL);
    slv_encoder_destroy(NULL);

    encode_clip(SLV_CODEC_VP8, 64, 48, 10, 0, 1);
    encode_clip(SLV_CODEC_VP9, 64, 48, 10, 0, 1);
    encode_clip(SLV_CODEC_VP8, 33, 17, 8, 200, 2);
    encode_clip(SLV_CODEC_VP9, 33, 17, 8, 200, 2);
    encode_clip(SLV_CODEC_VP9, 320, 240, 12, 0, 4);

    if (failures != 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("all video tests passed\n");
    return 0;
}
