#!/usr/bin/env python3
"""Drop the given files from the OS page cache (posix_fadvise DONTNEED; no root needed) to measure cold-disk loads."""
import os, sys
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
