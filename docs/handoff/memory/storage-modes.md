---
name: storage-modes
description: "User 2026-11-16: three storage modes for ALL files Studio/tools write (models, caches, kernels, config, history, outputs): portable next to the package/executable, system (XDG etc.), custom; switchable in Settings"
metadata:
  type: project
---

User (2026-11-16): "we want 3 modes for all files it writes (models, caches, kernels, shaders, config files, history etc):
a) all in a directory next to the relevant package/executable so everything stays portable (e.g. ./data next to the
AppImage), b) in the system (e.g. ~/.local/share etc.), c) in a custom location. This should easily be configurable in
the settings."

- Portable = next to the *package file* (for an AppImage: dirname($APPIMAGE), never APPDIR, which is the mounted/extracted
  image; for Velopack on Windows: next to the install root, never inside `current/`, which updates replace; tar.gz: next
  to the executable).
- Applies to everything: settings, studio.db, page history/outputs, models, artifact store/kernels, Hub caches, logs.
- Bootstrap: the mode itself must be discoverable before settings load (marker next to the executable for portable, a
  small pointer in the system config dir for custom). Switching offers to move existing data.

**Why:** portable installs (USB, shared machines) vs clean system installs vs big external disks for models.
**How to apply:** one storage-layout service used by every component; no component computes its own path.
Related: [[branching-and-studio-releases]], [[work-dirs-cleanup]].
