/* Compiles the miniaudio implementation with the features Sharplight uses (devices and decoding; Opus and Vorbis through custom backends). */
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_RESOURCE_MANAGER

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "decoders/libopus/miniaudio_libopus.c"
#include "decoders/libvorbis/miniaudio_libvorbis.c"
