using System.CommandLine;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Samples;

/// <summary>
/// The device and sizing options shared by the LLM samples: <c>--cpu</c>, <c>--device</c>, <c>--ctx</c>, <c>--batch</c>,
/// <c>--kv-type</c>, <c>--disable-features</c>, <c>--cpu-offload</c>, <c>--experts-on-disk</c>, <c>--expert-cache</c>, <c>--reduced-precision</c>,
/// <c>--fp8-block-scaled</c>, <c>--provider</c>.
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
        Description = "GPU provider to run on (metal, vulkan, cuda); default: the first available one (Metal on Apple silicon, else Vulkan).",
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

    /// <summary><c>--experts-on-disk</c>.</summary>
    public Option<bool> ExpertsOnDisk { get; } = new("--experts-on-disk")
    {
        Description = "With --cpu-offload: routed experts of mixture-of-experts models that fit neither the GPU nor host memory stay in the model file "
            + "and are read from disk on demand into a cache (models larger than RAM + VRAM; decode at the disk's speed).",
    };

    /// <summary><c>--expert-cache</c>.</summary>
    public Option<long?> ExpertCache { get; } = OptionFactory.Size("--expert-cache",
        "With --experts-on-disk: host memory for the cache of experts read from disk, e.g. 24G (default: sized from the free host memory). "
        + "A cache smaller than the experts on the CPU leaves them on disk even when they would fit.");

    /// <summary><c>--reduced-precision</c>.</summary>
    public Option<bool> ReducedPrecision { get; } = new("--reduced-precision")
    {
        Description = "Opt-in: faster kernels at reduced (f16) precision where the default keeps near-f32 accuracy "
            + "(e.g. one f16 product in the chunked delta rule of Qwen3.5-style models, and their delta-rule state in bf16).",
    };

    /// <summary><c>--fp8-block-scaled</c>.</summary>
    public Option<string> BlockScaledFp8 { get; } = CreateBlockScaledFp8();

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
        command.Options.Add(ExpertsOnDisk);
        command.Options.Add(ExpertCache);
        command.Options.Add(ReducedPrecision);
        command.Options.Add(BlockScaledFp8);
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
            DisabledFeatures(parseResult),
            parseResult.GetValue(CpuOffload),
            parseResult.GetValue(ReducedPrecision),
            parseResult.GetValue(Provider),
            parseResult.GetValue(ExpertsOnDisk),
            parseResult.GetValue(ExpertCache));
    }

    private IReadOnlyList<string> DisabledFeatures(ParseResult parseResult)
    {
        var features = parseResult.GetValue(DisableFeatures) ?? [];
        return parseResult.GetValue(BlockScaledFp8) == "off" && !features.Contains(DeviceFeatures.BlockScaledFp8, StringComparer.OrdinalIgnoreCase)
            ? [.. features, DeviceFeatures.BlockScaledFp8]
            : features;
    }

    private static Option<string> CreateBlockScaledFp8()
    {
        var option = new Option<string>("--fp8-block-scaled")
        {
            Description = "Block-scaled FP8 (default on): models with block-scaled E4M3 weights run their prompt-sized products on the "
                + "block-scaled MMA with power-of-two scales where the device has it (CUDA, compute capability 12.x; about 2 % more serving "
                + "throughput, perplexity +0.3 to +0.8 %, the form vLLM uses by default). off keeps the checkpoints' f32 scales "
                + "(the same as --disable-features block_scaled_fp8).",
            DefaultValueFactory = _ => "on",
            HelpName = "on|off",
        };
        option.AcceptOnlyFromAmong("on", "off");
        return option;
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
