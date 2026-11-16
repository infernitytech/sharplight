#include "sharplight_video.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vpx_version.h"
#include "vpx/vp8cx.h"
#include "vpx/vpx_encoder.h"

#if defined(_MSC_VER)
#define SLV_THREAD_LOCAL __declspec(thread)
#else
#define SLV_THREAD_LOCAL _Thread_local
#endif

#define MAX_DIMENSION 16384

struct slv_encoder {
    vpx_codec_ctx_t context;
    vpx_codec_iter_t iterator;
    int width;
    int height;
    int initialized;
    int flushed;
};

static SLV_THREAD_LOCAL char g_last_error[256];

static int fail(int code, const char *message)
{
    size_t length = strlen(message);
    if (length >= sizeof g_last_error) {
        length = sizeof g_last_error - 1;
    }
    memcpy(g_last_error, message, length);
    g_last_error[length] = '\0';
    return code;
}

static int fail_codec(slv_encoder *encoder, const char *what)
{
    const char *detail = vpx_codec_error_detail(&encoder->context);
    const char *text = vpx_codec_error(&encoder->context);
    char message[sizeof g_last_error];
    snprintf(message, sizeof message, "%s: %s%s%s%s", what, text != NULL ? text : "error", detail != NULL ? " (" : "",
             detail != NULL ? detail : "", detail != NULL ? ")" : "");
    return fail(SLV_ERR_ENCODER, message);
}

int slv_abi_version(void)
{
    return SLV_ABI_VERSION;
}

const char *slv_version_string(void)
{
    return "libvpx " VERSION_STRING_NOSP;
}

const char *slv_last_error(void)
{
    return g_last_error;
}

int slv_encoder_create(int codec, int width, int height, int fps_num, int fps_den, int bitrate_kbps, int quality, int threads,
                       int keyframe_interval, slv_encoder **out_encoder)
{
    if (out_encoder == NULL) {
        return fail(SLV_ERR_INVALID_ARGUMENT, "out_encoder is null");
    }
    *out_encoder = NULL;
    if (codec != SLV_CODEC_VP8 && codec != SLV_CODEC_VP9) {
        return fail(SLV_ERR_UNSUPPORTED, "unknown codec");
    }
    if (width < 1 || height < 1 || width > MAX_DIMENSION || height > MAX_DIMENSION || fps_num < 1 || fps_den < 1 || bitrate_kbps < 0 ||
        quality < 0 || quality > 100 || threads < 0 || keyframe_interval < 0) {
        return fail(SLV_ERR_INVALID_ARGUMENT, "invalid encoder parameter");
    }

    vpx_codec_iface_t *iface = codec == SLV_CODEC_VP9 ? vpx_codec_vp9_cx() : vpx_codec_vp8_cx();
    vpx_codec_enc_cfg_t config;
    if (vpx_codec_enc_config_default(iface, &config, 0) != VPX_CODEC_OK) {
        return fail(SLV_ERR_ENCODER, "no default configuration");
    }

    config.g_w = (unsigned int)width;
    config.g_h = (unsigned int)height;
    config.g_timebase.num = fps_den;
    config.g_timebase.den = fps_num;
    config.g_threads = (unsigned int)(threads > 0 ? threads : 1);
    config.g_error_resilient = 0;
    config.rc_min_quantizer = 0;
    config.rc_max_quantizer = 63;
    config.kf_mode = VPX_KF_AUTO;
    config.kf_min_dist = 0;
    config.kf_max_dist = (unsigned int)(keyframe_interval > 0 ? keyframe_interval : 120);
    int cq_level = (100 - quality) * 63 / 100;
    if (bitrate_kbps > 0) {
        config.rc_end_usage = VPX_VBR;
        config.rc_target_bitrate = (unsigned int)bitrate_kbps;
    } else {
        config.rc_end_usage = VPX_Q;
        /* The cap is generous: in constant-quality mode the quantizer, not the rate, drives the size. */
        config.rc_target_bitrate = (unsigned int)((long long)width * height / 100 + 500);
    }

    slv_encoder *encoder = calloc(1, sizeof *encoder);
    if (encoder == NULL) {
        return fail(SLV_ERR_OUT_OF_MEMORY, "out of memory");
    }
    encoder->width = width;
    encoder->height = height;
    if (vpx_codec_enc_init(&encoder->context, iface, &config, 0) != VPX_CODEC_OK) {
        int result = fail_codec(encoder, "vpx_codec_enc_init");
        free(encoder);
        return result;
    }
    encoder->initialized = 1;

    vpx_codec_control(&encoder->context, VP8E_SET_CPUUSED, 4);
    vpx_codec_control(&encoder->context, VP8E_SET_CQ_LEVEL, (unsigned int)cq_level);
    if (codec == SLV_CODEC_VP9) {
        vpx_codec_control(&encoder->context, VP9E_SET_TILE_COLUMNS, 6);
        vpx_codec_control(&encoder->context, VP9E_SET_ROW_MT, 1);
        vpx_codec_control(&encoder->context, VP8E_SET_ENABLEAUTOALTREF, 1);
    } else {
        /* Hidden alt-ref frames of VP8 are separate packets without a presentation time; not worth the muxing complexity. */
        vpx_codec_control(&encoder->context, VP8E_SET_ENABLEAUTOALTREF, 0);
    }

    *out_encoder = encoder;
    return SLV_OK;
}

int slv_encoder_encode(slv_encoder *encoder, const uint8_t *y, int y_stride, const uint8_t *u, int u_stride, const uint8_t *v, int v_stride,
                       int64_t pts, int force_keyframe)
{
    if (encoder == NULL || y == NULL || u == NULL || v == NULL || y_stride <= 0 || u_stride <= 0 || v_stride <= 0) {
        return fail(SLV_ERR_INVALID_ARGUMENT, "invalid frame argument");
    }
    if (encoder->flushed) {
        return fail(SLV_ERR_INVALID_ARGUMENT, "the encoder was flushed");
    }

    vpx_image_t image;
    memset(&image, 0, sizeof image);
    image.fmt = VPX_IMG_FMT_I420;
    image.cs = VPX_CS_BT_601;
    image.range = VPX_CR_STUDIO_RANGE;
    image.w = (unsigned int)encoder->width;
    image.h = (unsigned int)encoder->height;
    image.d_w = image.w;
    image.d_h = image.h;
    image.bit_depth = 8;
    image.x_chroma_shift = 1;
    image.y_chroma_shift = 1;
    image.bps = 12;
    image.planes[VPX_PLANE_Y] = (unsigned char *)(uintptr_t)y;
    image.planes[VPX_PLANE_U] = (unsigned char *)(uintptr_t)u;
    image.planes[VPX_PLANE_V] = (unsigned char *)(uintptr_t)v;
    image.stride[VPX_PLANE_Y] = y_stride;
    image.stride[VPX_PLANE_U] = u_stride;
    image.stride[VPX_PLANE_V] = v_stride;

    vpx_enc_frame_flags_t flags = force_keyframe ? VPX_EFLAG_FORCE_KF : 0;
    if (vpx_codec_encode(&encoder->context, &image, pts, 1, flags, VPX_DL_GOOD_QUALITY) != VPX_CODEC_OK) {
        return fail_codec(encoder, "vpx_codec_encode");
    }
    encoder->iterator = NULL;
    return SLV_OK;
}

int slv_encoder_flush(slv_encoder *encoder)
{
    if (encoder == NULL) {
        return fail(SLV_ERR_INVALID_ARGUMENT, "encoder is null");
    }
    if (!encoder->flushed) {
        encoder->flushed = 1;
        if (vpx_codec_encode(&encoder->context, NULL, 0, 1, 0, VPX_DL_GOOD_QUALITY) != VPX_CODEC_OK) {
            return fail_codec(encoder, "vpx_codec_encode (flush)");
        }
        encoder->iterator = NULL;
    }
    return SLV_OK;
}

int slv_encoder_next_packet(slv_encoder *encoder, const uint8_t **data, size_t *size, int64_t *pts, int *is_keyframe)
{
    if (encoder == NULL || data == NULL || size == NULL || pts == NULL || is_keyframe == NULL) {
        return -fail(SLV_ERR_INVALID_ARGUMENT, "invalid packet argument");
    }
    const vpx_codec_cx_pkt_t *packet;
    while ((packet = vpx_codec_get_cx_data(&encoder->context, &encoder->iterator)) != NULL) {
        if (packet->kind != VPX_CODEC_CX_FRAME_PKT) {
            continue;
        }
        *data = packet->data.frame.buf;
        *size = packet->data.frame.sz;
        *pts = packet->data.frame.pts;
        *is_keyframe = (packet->data.frame.flags & VPX_FRAME_IS_KEY) != 0;
        return 1;
    }
    return 0;
}

void slv_encoder_destroy(slv_encoder *encoder)
{
    if (encoder == NULL) {
        return;
    }
    if (encoder->initialized) {
        vpx_codec_destroy(&encoder->context);
    }
    free(encoder);
}
