using System.Globalization;
using System.Text.RegularExpressions;
using Sharplight.Cpu;
using Sharplight.Import.Gguf;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Studio.Services.Hub;

namespace Sharplight.Studio.Services.Compatibility;

/// <summary>
/// Says which compatibility reason explains a failed import or load (unsupported architecture, unsupported weight format), so the
/// library's error names it instead of only the importer's message.
/// </summary>
internal static partial class ImportFailure
{
    /// <summary>Bytes of a local GGUF file read for its metadata and tensor table.</summary>
    private const int GgufPrefixBytes = VariantProbe.MaxGgufPrefixBytes;

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
                // Image models in GGUF (stable-diffusion.cpp, FLUX) are not judged by a language-model architecture: their importer's message stands.
                return Library.ImageModelProbe.KindOf(path) is null ? FromLocalGguf(path) : null;
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

    /// <summary>A GGUF file on disk, with the tensor tables of all its shards when it is the first shard of a split model.</summary>
    private static ModelCompatibility FromLocalGguf(string path)
    {
        var prefix = ReadPrefix(path, GgufPrefixBytes);
        var metadata = GgufHeaderProbe.ReadMetadata(prefix);
        var tensors = GgufHeaderProbe.ReadTensorTable(prefix)?.ToList();
        var shard = ShardPattern().Match(Path.GetFileName(path));
        if (tensors is not null && shard.Success && int.Parse(shard.Groups[1].Value, CultureInfo.InvariantCulture) == 1)
        {
            int count = int.Parse(shard.Groups[2].Value, CultureInfo.InvariantCulture);
            for (int i = 2; i <= count && tensors is not null; i++)
            {
                string next = Path.Combine(Path.GetDirectoryName(path) ?? "",
                    Path.GetFileName(path)[..shard.Index] + string.Create(CultureInfo.InvariantCulture, $"-{i:D5}-of-{shard.Groups[2].Value}.gguf"));
                var more = File.Exists(next) ? GgufHeaderProbe.ReadTensorTable(ReadPrefix(next, GgufPrefixBytes)) : null;
                tensors = more is null ? null : [.. tensors, .. more];
            }
        }

        var verdict = CompatibilityChecker.FromGguf(metadata, tensors);
        return tensors is null && verdict.State is CompatibilityState.Supported or CompatibilityState.Partial
            ? ModelCompatibility.Unknown($"{verdict.Reason} The tensor table could not be read completely.")
            : verdict;
    }

    [GeneratedRegex(@"-(\d{5})-of-(\d{5})\.gguf$", RegexOptions.IgnoreCase)]
    private static partial Regex ShardPattern();

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
