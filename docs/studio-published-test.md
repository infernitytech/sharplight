# Test of the published Studio build (2026-11-18)

Rule: test the build users get. This round downloaded the newest unstable Linux x64 AppImage from the update channel's R2 folder and
drove it through its real UI. Screenshots are kept outside the repository (`/workspace/work/studio-s4/pub-*.png` on the test machine).

## Setup

- Download: `https://download.sharplight.dev/studio/SharplightStudio-linux-x64-unstable.AppImage` (128.5 MB, last modified
  2026-11-18 10:14 UTC). The Velopack feed `releases.linux-x64-unstable.json` lists `0.3.30-alpha.ga2085ec9e4` as the newest full
  release, and it matches the newest completed `studio.yml` run on `develop` (commit `a2085ec9`, green). Two newer develop commits
  were still building or not yet published.
- Run: Xvfb (1600x1000), `APPIMAGE_EXTRACT_AND_RUN=1`, `SHARPLIGHT_VK_DEVICES=1`, `libegl1` installed, clicks and typing through `xdotool`
  (Studio has no automation hooks). Storage was run twice: once with `SHARPLIGHT_HOME` in a scratch folder, once as real portable mode
  (an empty `sharplight.portable` next to the AppImage). Portable mode created `data/{cache,config,studio}` next to the AppImage as documented.

## What works

- Starts in about 5 s without errors on stderr, shows the verified catalog (199 models) at once, the window is 1440x900 on a custom title bar.
- Every page renders: Chat, Transcribe, Speak, Search, Audio, Image, Video, Vision, Tasks, Models (Discover with Verified, Sharplight feed
  and Hub search, Library, Resident, Downloads), Server, Statistics and the six Settings tabs.
- Update check: Settings > Updates shows `0.3.30-alpha.ga2085ec9e4`, channel *Preview (unstable)*, "Studio ... is up to date".
- Sharplight feed: lists 5 signed packages (stable channel), installed whisper tiny (with its Silero VAD companion) and Qwen3 0.6B Q8_0
  through the UI; both appear in the Library with a *Feed 1.0.0* badge. Whisper tiny loaded on the RTX GPU and the Transcribe page showed it.
- Chat: Qwen3 0.6B (feed package) loaded on GPU 1 through Vulkan, answered "Say hello in five words." at 337 tok/s, 0.12 s to first token.
- Native libraries: `libsharplight_audio.so`, the FFmpeg libraries and the Slang libraries are in the package (`runtimes/linux-x64/native`); the
  audio library has no missing dependencies. The FFmpeg libraries have no RUNPATH (`ldd` alone does not find their siblings); the loader
  loads them by full path in dependency order, so this is harmless, but it was not exercised by a video run in this round.

## Problems found

| # | Problem | Where | Status |
|---|---|---|---|
| 1 | The "No ... model is loaded" notice on Chat, Transcribe, Speak, Search, Audio, Image, Video and Vision: the *Library* button is painted over the end of the text (the text sits in a horizontal stack that does not wrap) | `Views/Inference/InferenceFrame.axaml`, `Views/ChatPage.axaml` | fixed on this branch (grid, wrapping text) |
| 2 | Sharplight feed list: the badges of a row (*Verified*, version, *Fits your GPU*, *Installed 1.0.0*) overlap the size at the right and each other on narrow rows | `Views/FeedCatalogView.axaml` | fixed on this branch (wrapping badge row) |
| 3 | No log file is written anywhere (`<data>/logs` does not exist, stdout and stderr are empty), so startup problems of a user's machine leave no trace | `StorageLayout.LogsDirectory` has no writer | open; needs a file logger |
| 4 | The package ships 28 `*.xml` API documentation files (unpacked 322 MB) | `packaging/` | open, minor (size) |
| 5 | Installed feed packages pile up as stacked cards above the list in Discover (feed view), pushing the list down | Feed view | open, cosmetic |

No crash, no missing native library, no binding error in the log (none is written, see 3) and no hang was seen. The update check worked.

## Found by the new page screenshot tests (all fixed on this branch)

`PageScreenshotTests` (light and dark, 1440 and 900 px) found three more problems on pages the manual round did not stress:

- Tasks: the *Open / Library / Browse models* buttons of a kind card ran out of the card (a horizontal stack); now they wrap.
- Chat at 900 px: the conversation list (264 px) plus the settings panel (300 px) left about 100 px for the conversation, and the model picker
  and capability badges overlapped; the settings panel now closes once when the page becomes narrower than 1000 px (the header toggle opens it again).
- Statistics: the last tile cell of the uniform row ended a pixel or two outside the row (rounded-up cells); the row is clipped.
- Open (seen in the 900 px screenshots, not caught by the checker): the Image page's *Input image* card wraps its text a word per line (a few
  characters wide) when the window is 900 px wide.

## Not covered

Video and image generation runs, microphone and playback devices (the Xvfb machine has no sound server), the real update download and restart
(the installed build was already the newest), and the stable channel (no stable release exists yet).
