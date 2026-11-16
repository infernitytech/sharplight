#include "sharplight_audio.h"

#include "miniaudio.h"

#include "internal.h"

#if defined(_WIN32)
#include <windows.h>
#endif

#define SLA_VERSION_MAJOR 1
#define SLA_VERSION_MINOR 0
#define SLA_VERSION_PATCH 0

/* Keep in sync with third_party/README.md. */
static const char g_version_text[] = "sharplight_audio 1.0.0; miniaudio " MA_VERSION_STRING "; libopus 1.6.1; opusfile 0.12; libogg 1.3.6; libvorbis 1.3.7+git1b75110b";

int32_t sla_version(void)
{
    return (SLA_VERSION_MAJOR << 16) | (SLA_VERSION_MINOR << 8) | SLA_VERSION_PATCH;
}

int32_t sla_abi_version(void)
{
    return SLA_ABI_VERSION;
}

const char* sla_version_string(void)
{
    return g_version_text;
}

const char* sla_error_string(sla_result result)
{
    switch (result)
    {
        case SLA_OK:
            return "success";
        case SLA_ERR_INVALID_ARGUMENT:
            return "invalid argument";
        case SLA_ERR_OUT_OF_MEMORY:
            return "out of memory";
        case SLA_ERR_NO_BACKEND:
            return "no audio backend or device available";
        case SLA_ERR_IO:
            return "I/O error";
        case SLA_ERR_UNSUPPORTED:
            return "unsupported format";
        case SLA_ERR_DECODE:
            return "corrupt or truncated audio data";
        case SLA_ERR_DEVICE:
            return "audio device could not be opened or started";
        case SLA_ERR_INVALID_STATE:
            return "invalid state";
        case SLA_ERR_BUFFER_TOO_SMALL:
            return "buffer too small";
        default:
            return "unknown error";
    }
}

FILE* sla_fopen_utf8(const char* utf8_path, const char* mode)
{
#if defined(_WIN32)
    wchar_t wide_path[MAX_PATH * 4];
    wchar_t wide_mode[8];
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8_path, -1, wide_path, (int)(sizeof(wide_path) / sizeof(wide_path[0]))) == 0)
    {
        return NULL;
    }

    if (MultiByteToWideChar(CP_UTF8, 0, mode, -1, wide_mode, (int)(sizeof(wide_mode) / sizeof(wide_mode[0]))) == 0)
    {
        return NULL;
    }

    return _wfopen(wide_path, wide_mode);
#else
    return fopen(utf8_path, mode);
#endif
}
