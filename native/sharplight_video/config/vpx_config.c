/* Copyright (c) 2011 The WebM project authors. All Rights Reserved. */
/*  */
/* Use of this source code is governed by a BSD-style license */
/* that can be found in the LICENSE file in the root of the source */
/* tree. An additional intellectual property rights grant can be found */
/* in the file PATENTS.  All contributing project authors may */
/* be found in the AUTHORS file in the root of the source tree. */
#include "vpx/vpx_codec.h"
static const char* const cfg = "--target=generic-gnu --disable-examples --disable-tools --disable-docs --disable-unit-tests --disable-install-docs --disable-vp8-decoder --disable-vp9-decoder --enable-vp8-encoder --enable-vp9-encoder --disable-runtime-cpu-detect --enable-pic --enable-multithread --disable-webm-io --disable-libyuv --disable-postproc --disable-vp9-postproc --disable-temporal-denoising --disable-vp9-temporal-denoising --disable-install-bins --disable-install-libs --disable-install-srcs";
const char *vpx_codec_build_config(void) {return cfg;}
