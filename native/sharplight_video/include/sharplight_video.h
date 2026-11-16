/*
 * Sharplight video: VP8 and VP9 encoding (libvpx) behind a small C API. The managed side (src/Sharplight.Video) muxes WebM.
 * Frames go in as planar I420 (8-bit, BT.601 limited range); compressed packets come out in presentation order.
 * Thread safety: one encoder must not be used from two threads at once; different encoders are independent.
 */
#ifndef SHARPLIGHT_VIDEO_H
#define SHARPLIGHT_VIDEO_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define SLV_API __declspec(dllexport)
#else
#define SLV_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SLV_ABI_VERSION 1

/* Result codes: 0 is success. */
#define SLV_OK 0
#define SLV_ERR_INVALID_ARGUMENT 1
#define SLV_ERR_OUT_OF_MEMORY 2
#define SLV_ERR_ENCODER 3
#define SLV_ERR_UNSUPPORTED 4

#define SLV_CODEC_VP8 0
#define SLV_CODEC_VP9 1

typedef struct slv_encoder slv_encoder;

/* Version of this interface; the managed side refuses a library with a different value. */
SLV_API int slv_abi_version(void);

/* "libvpx 1.15.2" (static string). */
SLV_API const char *slv_version_string(void);

/* Text of the last error of the calling thread's most recent failing call (static per thread, valid until the next call). */
SLV_API const char *slv_last_error(void);

/*
 * Creates an encoder.
 *   codec            SLV_CODEC_VP8 or SLV_CODEC_VP9
 *   width, height    pixels, 1..16384
 *   fps_num/fps_den  frame rate as a fraction; packet timestamps are in frames (units of fps_den / fps_num seconds)
 *   bitrate_kbps     > 0: average bitrate target; 0: constant quality from `quality`
 *   quality          0 (smallest) .. 100 (best), used when bitrate_kbps is 0
 *   threads          worker threads, 0 = 1
 *   keyframe_interval maximum frames between keyframes, 0 = default
 */
SLV_API int slv_encoder_create(int codec, int width, int height, int fps_num, int fps_den, int bitrate_kbps, int quality,
                               int threads, int keyframe_interval, slv_encoder **out_encoder);

/*
 * Submits one frame (pts counted in frames, strictly increasing). Planes are 8-bit I420: y is width x height, u and v are
 * ((width+1)/2) x ((height+1)/2); strides are in bytes. Packets become available through slv_encoder_next_packet.
 */
SLV_API int slv_encoder_encode(slv_encoder *encoder, const uint8_t *y, int y_stride, const uint8_t *u, int u_stride, const uint8_t *v,
                               int v_stride, int64_t pts, int force_keyframe);

/* Signals the end of the stream; the remaining packets become available through slv_encoder_next_packet. */
SLV_API int slv_encoder_flush(slv_encoder *encoder);

/*
 * Fetches the next compressed packet. Returns 1 and fills the outputs when one is available, 0 when there is none (submit more
 * frames or flush), negative on error. `data` stays valid until the next call on this encoder.
 */
SLV_API int slv_encoder_next_packet(slv_encoder *encoder, const uint8_t **data, size_t *size, int64_t *pts, int *is_keyframe);

SLV_API void slv_encoder_destroy(slv_encoder *encoder);

#ifdef __cplusplus
}
#endif

#endif
