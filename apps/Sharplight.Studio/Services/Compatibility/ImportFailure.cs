using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Models;
using Sharplight.Package;

namespace Sharplight.Studio.Services.Compatibility;

/// <summary>
/// Says which compatibility reason explains a failed import or load (unsupported architecture, unsupported weight format), so the
/// library's error names it instead of only the importer's message.
/// </summary>
internal static class ImportFailure
{
    /// <summary>Bytes of a local GGUF file read for its metadata and tensor table.</summary>
    private const int GgufPrefixBytes = 32 << 20;

    /// <summary>The longest original message quoted after the verdict.</summary>
    private const int MaxQuotedMessage = 240;

    /// <summary>
    /// The error text for a failed import or load: the compatibility verdict and its reason when one applies, followed by
    /// <paramref name="message"/>; otherwise <paramref name="message"/>.
    /// </summary>
    /// <param name="error">The failure.</param>
    /// <param name="message">The error text shown without a verdict (and quoted after one).</param>
    /// <param name="path">The model's file or directory, read for the verdict when the error itself does not name one; or null.</param>
    public static string Describe(Exception error, string message, string? path = null)
    {
        ArgumentNullException.ThrowIfNull(error);
        var verdict = FromException(error) ?? (path is null ? null : FromLocalModel(path));
        if (verdict is not { IsUnsupported: true })
        {
            return message;
        }

        // Importer messages for unknown architectures list every supported one; that list does not help here.
        return message.Length <= MaxQuotedMessage ? $"{verdict.Text}. {verdict.Reason} ({message})" : $"{verdict.Text}. {verdict.Reason}";
    }

    /// <summary>The verdict an importer's exception carries (anywhere in the chain of inner exceptions), or null.</summary>
    /// <param name="error">The failure.</param>
    public static ModelCompatibility? FromException(Exception error)
    {
        for (var current = error; current is not null; current = current.InnerException)
        {
            switch (current)
            {
                case UnsupportedArchitectureException architecture:
                    return new ModelCompatibility(CompatibilityState.UnsupportedArchitecture, architecture.Architecture,
                        $"No importer mapping exists for the architecture '{architecture.Architecture}'.");
                case UnsupportedWeightFormatException format:
                    return new ModelCompatibility(CompatibilityState.UnsupportedWeightFormat, format.WeightFormat,
                        $"The weights are stored as {format.WeightFormat}, which Sharplight cannot read.");
                case AggregateException aggregate when aggregate.InnerExceptions.Count > 0:
                    return aggregate.InnerExceptions.Select(FromException).FirstOrDefault(verdict => verdict is not null);
            }
        }

        return null;
    }

    /// <summary>
    /// The verdict for a model on disk: a GGUF file (header and tensor table), a checkpoint directory (<c>config.json</c>) or a
    /// <c>.slm</c> package (its stored weight types against the CPU kernels); null for other files or when nothing can be read.
    /// </summary>
    /// <param name="path">The file or directory.</param>
    public static ModelCompatibility? FromLocalModel(string path)
    {
        try
        {
            if (Directory.Exists(path))
            {
                string config = Path.Combine(path, "config.json");
                string modelOpt = Path.Combine(path, "hf_quant_config.json");
                return File.Exists(config)
                    ? CompatibilityChecker.FromConfig(File.ReadAllText(config), File.Exists(modelOpt) ? File.ReadAllText(modelOpt) : null)
                    : null;
            }

            if (!File.Exists(path))
            {
                return null;
            }

            if (path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            {
                var prefix = ReadPrefix(path, GgufPrefixBytes);
                return CompatibilityChecker.FromGguf(GgufHeaderProbe.ReadMetadata(prefix), GgufHeaderProbe.ReadTensorTypes(prefix));
            }

            if (path.EndsWith(".slm", StringComparison.OrdinalIgnoreCase))
            {
                return FromPackage(ModelPackageReader.Open(path));
            }

            return null;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or InvalidDataException or NotSupportedException)
        {
            return null;
        }
    }

    /// <summary>A package's stored weight types: any without CPU kernels (the fallback of every provider) keeps the model from loading.</summary>
    private static ModelCompatibility? FromPackage(PackageInfo package)
    {
        var unsupported = package.Weights.Where(weight => !CpuExecutionProvider.SupportsWeightType(weight.Type)).ToList();
        if (unsupported.Count == 0)
        {
            return null;
        }

        string name = CompatibilityChecker.TypeName(unsupported[0].Type);
        int count = unsupported.Count(weight => weight.Type == unsupported[0].Type);
        return new ModelCompatibility(CompatibilityState.UnsupportedWeightFormat, name,
            $"{count} weight{(count == 1 ? " is" : "s are")} stored as {name}, which has no compute kernels.");
    }

    private static byte[] ReadPrefix(string path, int count)
    {
        using var file = File.OpenRead(path);
        var buffer = new byte[(int)Math.Min(count, file.Length)];
        file.ReadExactly(buffer);
        return buffer;
    }
}
