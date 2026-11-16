# Third-party notices

The native library shipped in `runtimes/<rid>/native/` (`libsharplight_video`) statically includes the following software. The
sources are vendored unmodified under `third_party/` in the repository, with their license files.

## libvpx 1.15.2 (https://www.webmproject.org/code/)

Copyright (c) 2010, The WebM Project authors. All rights reserved. BSD 3-Clause License:

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions
are met: (1) Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer. (2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution. (3) Neither the name of Google, nor the WebM
Project, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior
written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES
ARE DISCLAIMED; see `third_party/libvpx/LICENSE` for the complete text and `third_party/libvpx/PATENTS` for the additional patent grant.

## ffmpeg (optional, not included)

MP4 output uses an `ffmpeg` executable that the user installs separately; it is not part of this package and is run as a separate process.
