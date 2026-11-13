# Sharplight.Samples.Common

Helpers shared by the sample applications (not a library for production use): provider/importer registration, device
selection, the shared `--cpu/--device/--ctx/--batch/--disable-features/--cpu-offload` option set, typed option factories
with validation, `ByteSize` parsing and the `CommandRunner` that maps failures to exit codes. `ConsoleInterrupt` lets Ctrl+C stop one
stretch of work (an answer) instead of the whole command.

`samples/Sharplight.Samples.Hub` adds the model store commands (`pull`, `search`, `list`, `show`, `rm`), the pull options,
console pull progress and model-argument resolution shared by `sharplight` and `sharplight-server`.
