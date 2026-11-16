# Storage locations

Everything Sharplight writes goes to one place: settings and other configuration files, Studio's database, page history and outputs,
the model store, the compiled-kernel artifact store, extracted shader sources and logs. You can choose between three modes. Studio,
the command-line tools and the engine all resolve the same layout (`StorageLayout` in `Sharplight.Core`, namespace
`Sharplight.Storage`):

| Mode | Root | When to use it |
|---|---|---|
| **Portable** | a `data` folder next to the package | Everything stays with the app: carry it on a USB drive, keep several versions side by side, delete the folder to remove all traces. |
| **System** (default) | the platform's per-user folders | A normal installation. The tools of a user share models and caches. |
| **Custom** | one folder you choose | A larger disk, a shared models drive, or a separate profile. |

## Folders per mode

Portable and custom use a single root:

| What | Folder |
|---|---|
| Configuration (`store.json`, Studio's `studio/settings.json`) | `<root>/config` |
| Data (Studio's `studio/studio.db`, `studio/discover-highlights.json`, `studio/pages/`) | `<root>` |
| Model store | `<root>/models` |
| Caches: compiled kernels (`artifacts/`), extracted shader sources (`shaders/`) | `<root>/cache` |
| Logs | `<root>/logs` |

System mode follows the platform conventions. Each folder below gets a `sharplight` subfolder unless the table spells out the full path:

| | Linux | Windows | macOS |
|---|---|---|---|
| Configuration | `$XDG_CONFIG_HOME` or `~/.config` | `%APPDATA%` | `~/Library/Application Support/sharplight/config` |
| Data, models (`models/`) | `$XDG_DATA_HOME` or `~/.local/share` | `%LOCALAPPDATA%` | `~/Library/Application Support` |
| Caches | `$XDG_CACHE_HOME` or `~/.cache` | `%LOCALAPPDATA%\sharplight\cache` | `~/Library/Caches` |
| Logs | `$XDG_STATE_HOME/sharplight/logs` or `~/.local/state/sharplight/logs` | `%LOCALAPPDATA%\sharplight\logs` | `~/Library/Logs` |

Relative `XDG_*` values are ignored, as the XDG specification requires. The system model store
(`~/.local/share/sharplight/models` on Linux) is the per-user store of earlier versions. The settings of earlier versions are in the
same place on Linux and Windows. Compiled kernels move from `~/.local/share/sharplight/artifacts` to the cache folder. The old
folder can be deleted, because the kernels are compiled again when they are needed.

## Where the package is

The portable folder goes next to what you see and move, never into a place that updates replace:

| Distribution | Package folder (`data/` goes inside it) |
|---|---|
| AppImage | the folder of the `.AppImage` file (`$APPIMAGE`), never the mounted image (`$APPDIR`) |
| Velopack on Windows (Setup.exe or portable zip) | the install root, next to `current/` and `Update.exe`, never inside `current/` |
| macOS `.app` | the folder that contains the bundle, never inside the bundle (writing there breaks the signature) |
| tar.gz, zip, build output | the executable's folder |

Portable mode is never used inside an installation: an AppImage's `APPDIR`, a Velopack `current/` folder, the inside of an app
bundle, `/usr`, or Program Files. It is also not used when the folder cannot be written. In those cases the system folders are used
and the layout says why (`StorageLayout.Notice`): Studio shows it on the *Storage* card and the tools print it in `storage`.

## How the mode is chosen

The mode must be known before any settings file is read, so it is never stored in the settings. At start-up the layout is
resolved in this order:

1. `SHARPLIGHT_HOME=<dir>`: a custom root for everything. While it is set, the mode cannot be changed in Studio or with
   `storage use`.
2. The portable marker: a file named `sharplight.portable` in the package folder. It may be empty or contain a JSON object.
   Selecting *Portable* creates it. You can also create it by hand to ship or unpack a portable copy.
3. The pointer file `storage.json` in the system configuration folder (`~/.config/sharplight/storage.json` on Linux):
   `{ "mode": "Custom", "root": "/mnt/big/sharplight" }`. It is shared by every tool of the user. A custom root that cannot be
   created or written (for example an unplugged drive) falls back to the system folders, with a notice.
4. Otherwise the system folders.

Narrower overrides apply on top of the layout:

| Override | Affects |
|---|---|
| `--models-dir DIR`, then `SHARPLIGHT_MODELS`, then the model store choice (`store use`, Studio's *Models* card) | the model store only (see [model-hub.md](model-hub.md#store)) |
| `SHARPLIGHT_COMPILATION_CACHE=<dir>` or `off` | the artifact store of compiled kernels only |
| `SHARPLIGHT_STUDIO_HOME=<dir>` | Studio's own settings and data (one folder, for tests and side-by-side profiles) |
| `DeviceOptions.KernelCacheDirectory` | the artifact store of one device (API) |

## Changing the location

You can change the location in Studio (*Settings > Storage*) or with the tools (`storage use`). Choose how the existing files
are handled:

- **Move** (default): every file is copied and verified, and the new mode is recorded. The originals are deleted at the next
  start, before anything opens them, and only if they have not changed since they were copied and the copy is still there. If
  the model store is on the same volume, it is renamed in one step instead of being copied. That happens right before the commit,
  so a cancelled or failed copy leaves the models in place.
- **Copy**: the same, but the originals are kept.
- **Start fresh**: only the new mode is recorded. The old files stay where they are.
- **Keep the models where they are**: the model store is not moved. It stays in use as a custom model folder, recorded in the new
  location's settings. Models are large, so this is offered separately.

Each copy is first written to a `.sl-partial` file. The original is hashed (SHA-256) while it is read, and the copy is read again and
compared before it gets its final name. A move records the originals in `storage-migration.json` in the new configuration folder.
The journal is committed only after the marker or pointer file has been written. An interruption at any point leaves one of two
states. Either the old location is complete and still in effect: copies made so far are reused by the next attempt, and an
uncommitted journal is discarded. Or the new location is complete and committed. A move never deletes an original before its copy
is verified. A new folder inside a current one, or the other way round, is refused. A custom folder must be empty or an existing
Sharplight storage folder. Files with the same name in an existing folder are replaced.

Studio keeps running in the old location until it restarts: the card offers *Restart now*. The command-line tools finish a move
immediately, because they do not hold the files open.

```
sharplight storage                                # mode, why, and every folder
sharplight storage use portable                   # or: system, or a directory (custom)
sharplight storage use /mnt/big/sharplight --copy # keep the originals
sharplight storage use system --fresh             # only switch, leave the files
sharplight storage use portable --keep-models     # leave the model store where it is
```

`storage` and `storage use` are part of every tool that has the store commands (`sharplight`, `sharplight-server`).

## API

```csharp
var layout = StorageLayout.Resolve(StorageEnvironment.Process);   // or take it from the container:
services.AddSharplightStorage();                                  // part of AddSharplight(); a host may register its own layout first
string models = layout.ModelsDirectory;                           // config, data, models, cache, logs, artifacts, shader sources
var resolution = ModelStore.Resolve(commandLine, settings, layout);
string? kernels = ArtifactStore.DirectoryFor(layout);              // null when SHARPLIGHT_COMPILATION_CACHE=off

// Change it: copy and verify, record the mode, and finish the move at the next start.
var target = StorageLayout.Create(StorageMode.Portable, null, StorageEnvironment.Process);
await new StorageMigration(layout, target, Enum.GetValues<StorageCategory>()).RunAsync(StorageTransferKind.Move, progress);
StorageMigration.CompletePending(StorageLayout.Resolve(StorageEnvironment.Process));   // at start-up
```

`StorageEnvironment` holds the inputs: the application folder, the platform, the home folder, the environment variables and a write
probe. Tests build their own environment to check every platform and distribution layout. It is the only place that asks the
operating system for well-known folders. A repository test (`StorageWritersTests`) fails when another source file calls
`GetFolderPath`, `GetTempPath` or `AppContext.BaseDirectory`. The exceptions are the few files that only read from those folders,
such as the native libraries next to the app.

Not covered: Velopack keeps its own update packages in the install root (Windows) or next to the AppImage while it updates. Real
temporary files (a precompile session's scratch artifact store, files being exported) still go to the system's temporary folder
and are deleted after use.
