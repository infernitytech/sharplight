/* Shared internals of the Sharplight audio library. */
#ifndef SLA_INTERNAL_H
#define SLA_INTERNAL_H

#include "sharplight_audio.h"

#include <stdio.h>

/* Opens a file given a UTF-8 path (wide-character API on Windows). */
FILE* sla_fopen_utf8(const char* utf8_path, const char* mode);

#endif
