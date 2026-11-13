using System.CommandLine;

namespace Sharplight.Samples;

/// <summary>The device and sizing options shared by the LLM samples: <c>--cpu</c>, <c>--device</c>, <c>--ctx</c>, <c>--batch</c>, <c>--disable-features</c>, <c>--cpu-offload</c>.</summary>
public sealed class RuntimeOptionSet
{
    /// <summary>Context length used when <c>--ctx</c> is not given.</summary>
    public const int DefaultContextLength = 4096;

    /// <summary>Forward-pass batch size used when <c>--batch</c> is not given.</summary>
    public const int DefaultBatchTokens = 512;

    /// <summary><c>--cpu</c>.</summary>
    public Option<bool> Cpu { get; } = new("--cpu") { Description = "Run on the CPU provider instead of a GPU." };

    /// <summary><c>--device</c>.</summary>
    public Option<int> Device { get; } = OptionFactory.Integer("--device", "Index of the device to use (see the CLI's --devices).", 0, minimum: 0, helpName: "index");

    /// <summary><c>--ctx</c>.</summary>
    public Option<int> Context { get; } = OptionFactory.Integer("--ctx", "Context length in tokens (per sequence).", DefaultContextLength, helpName: "tokens");

    /// <summary><c>--batch</c>.</summary>
    public Option<int> Batch { get; } = OptionFactory.Integer("--batch", "Maximum rows (prompt tokens) per forward pass.", DefaultBatchTokens, helpName: "rows");

    /// <summary><c>--disable-features</c>.</summary>
    public Option<string[]> DisableFeatures { get; } = OptionFactory.CommaSeparatedList(
        "--disable-features",
        "Comma-separated device features never to use, e.g. matrix_f16,float16 (diagnostics and fallback testing).",
        "features");

    /// <summary><c>--cpu-offload</c>.</summary>
    public Option<bool> CpuOffload { get; } = new("--cpu-offload")
    {
        Description = "Run layers that do not fit in device memory, and ops the device cannot execute, on the CPU.",
    };

    /// <summary>Adds all options to <paramref name="command"/>.</summary>
    public void AddTo(Command command)
    {
        command.Options.Add(Cpu);
        command.Options.Add(Device);
        command.Options.Add(Context);
        command.Options.Add(Batch);
        command.Options.Add(DisableFeatures);
        command.Options.Add(CpuOffload);
    }

    /// <summary>Reads the parsed values.</summary>
    public RuntimeSettings Read(ParseResult parseResult)
    {
        return new RuntimeSettings(
            parseResult.GetValue(Cpu),
            parseResult.GetValue(Device),
            parseResult.GetValue(Context),
            parseResult.GetValue(Batch),
            parseResult.GetValue(DisableFeatures) ?? [],
            parseResult.GetValue(CpuOffload));
    }
}
