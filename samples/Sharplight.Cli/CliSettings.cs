using Sharplight.Samples;
using Sharplight.Vision;

namespace Sharplight.Cli;

/// <summary>Everything the command line selects, after parsing and validation.</summary>
/// <param name="ModelPath">Model file or directory; null when only <see cref="ListDevices"/> was requested.</param>
/// <param name="ListDevices">Print the available devices first.</param>
/// <param name="Runtime">Device and sizing options shared with the other samples.</param>
/// <param name="SplitWeights">Relative layer share per device slot; one entry per opened device (the same GPU may be opened several times).</param>
/// <param name="GpuMemoryLimitBytes">Hard memory cap per device slot, or null for what the device reports as available.</param>
/// <param name="MmprojPath">Vision projector for GGUF models, or null.</param>
/// <param name="ImagePaths">Images to describe (the prompt is then formatted with the chat template).</param>
/// <param name="ImageTokens">Per-image token budget, or null for the model default.</param>
/// <param name="SavePath">Where to store the compiled model package, or null.</param>
/// <param name="BenchmarkTokens">Prompt length of the benchmark, or null to generate text instead.</param>
/// <param name="LogitsPath">Where to store the last-token logits of the prompt, or null.</param>
/// <param name="Generation">Prompt and sampling choices.</param>
internal sealed record CliSettings(
    string? ModelPath,
    bool ListDevices,
    RuntimeSettings Runtime,
    IReadOnlyList<double> SplitWeights,
    long? GpuMemoryLimitBytes,
    string? MmprojPath,
    IReadOnlyList<string> ImagePaths,
    ImageTokenLimits? ImageTokens,
    string? SavePath,
    int? BenchmarkTokens,
    string? LogitsPath,
    GenerationSettings Generation);
