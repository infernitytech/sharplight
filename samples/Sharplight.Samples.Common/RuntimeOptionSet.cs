using System.CommandLine;
using Sharplight.Tensors;

namespace Sharplight.Samples;

/// <summary>
/// The device and sizing options shared by the LLM samples: <c>--cpu</c>, <c>--device</c>, <c>--ctx</c>, <c>--batch</c>,
/// <c>--kv-type</c>, <c>--disable-features</c>, <c>--cpu-offload</c>, <c>--reduced-precision</c>, <c>--provider</c>.
/// </summary>
public sealed class RuntimeOptionSet
{
    /// <summary>Context length used when <c>--ctx</c> is not given.</summary>
    public const int DefaultContextLength = 4096;

    /// <summary>Forward-pass batch size used when <c>--batch</c> is not given: 0, chosen from the model and the free device memory.</summary>
    public const int DefaultBatchTokens = 0;

    /// <summary>KV cache element type used when <c>--kv-type</c> is not given.</summary>
    public const string DefaultKvCacheType = "f16";

    /// <summary><c>--cpu</c>.</summary>
    public Option<bool> Cpu { get; } = new("--cpu") { Description = "Run on the CPU provider instead of a GPU." };

    /// <summary><c>--provider</c>.</summary>
    public Option<string> Provider { get; } = new("--provider")
    {
        Description = "GPU provider to run on (metal, vulkan); default: the first available one (Metal on Apple silicon, else Vulkan).",
        HelpName = "name",
    };

    /// <summary><c>--device</c>.</summary>
    public Option<int> Device { get; } = OptionFactory.Integer("--device", "Index of the device to use (see the CLI's --devices).", 0, minimum: 0, helpName: "index");

    /// <summary><c>--ctx</c>.</summary>
    public Option<int> Context { get; } = OptionFactory.Integer("--ctx", "Context length in tokens (per sequence).", DefaultContextLength, helpName: "tokens");

    /// <summary><c>--batch</c>.</summary>
    public Option<int> Batch { get; } = OptionFactory.Integer("--batch", "Maximum rows (prompt tokens) per forward pass (0: up to 2048, chosen from the model's footprint and the free device memory).", DefaultBatchTokens, minimum: 0, helpName: "rows");

    /// <summary><c>--kv-type</c>.</summary>
    public Option<string> KvType { get; } = CreateKvType();

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

    /// <summary><c>--reduced-precision</c>.</summary>
    public Option<bool> ReducedPrecision { get; } = new("--reduced-precision")
    {
        Description = "Opt-in: faster kernels at reduced (f16) precision where the default keeps near-f32 accuracy "
            + "(e.g. one f16 product in the chunked delta rule of Qwen3.5-style models, and their delta-rule state in bf16).",
    };

    /// <summary>Adds all options to <paramref name="command"/>.</summary>
    public void AddTo(Command command)
    {
        command.Options.Add(Cpu);
        command.Options.Add(Provider);
        command.Options.Add(Device);
        command.Options.Add(Context);
        command.Options.Add(Batch);
        command.Options.Add(KvType);
        command.Options.Add(DisableFeatures);
        command.Options.Add(CpuOffload);
        command.Options.Add(ReducedPrecision);
    }

    /// <summary>Reads the parsed values.</summary>
    public RuntimeSettings Read(ParseResult parseResult)
    {
        return new RuntimeSettings(
            parseResult.GetValue(Cpu),
            parseResult.GetValue(Device),
            parseResult.GetValue(Context),
            parseResult.GetValue(Batch),
            DataType.Get(parseResult.GetValue(KvType)!),
            parseResult.GetValue(DisableFeatures) ?? [],
            parseResult.GetValue(CpuOffload),
            parseResult.GetValue(ReducedPrecision),
            parseResult.GetValue(Provider));
    }

    private static Option<string> CreateKvType()
    {
        var option = new Option<string>("--kv-type")
        {
            Description = "KV cache element type: f16, f32, q8_0 (half the memory of f16) or q4_0 (a quarter, less accurate). "
                + "Caches whose rows are not whole 32-value blocks stay f16.",
            DefaultValueFactory = _ => DefaultKvCacheType,
            HelpName = "type",
        };
        option.AcceptOnlyFromAmong("f16", "f32", "q8_0", "q4_0");
        return option;
    }
}
