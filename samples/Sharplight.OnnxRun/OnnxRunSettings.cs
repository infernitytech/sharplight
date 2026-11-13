namespace Sharplight.OnnxRun;

/// <summary>Command-line choices of the ONNX runner.</summary>
/// <param name="ModelPath">The <c>.onnx</c> file, or a package (<c>.slm</c>) converted from one.</param>
/// <param name="Provider">Execution provider that runs the model.</param>
/// <param name="DeviceIndex">Device index within the provider.</param>
/// <param name="BatchSize">Value of the symbolic batch dimension.</param>
/// <param name="TimedRuns">Number of timed runs after one warm-up run.</param>
/// <param name="InputShapes">Full-shape overrides per graph input; a non-positive entry marks the batch axis.</param>
/// <param name="InputFile">Raw float32 file for the first input; other inputs get random data.</param>
/// <param name="OutputFile">Where to store the first output as raw float32, or null.</param>
/// <param name="DumpGraph">Print the imported graph.</param>
internal sealed record OnnxRunSettings(
    string ModelPath,
    OnnxProvider Provider,
    int DeviceIndex,
    int BatchSize,
    int TimedRuns,
    IReadOnlyDictionary<string, long[]> InputShapes,
    string? InputFile,
    string? OutputFile,
    bool DumpGraph);
