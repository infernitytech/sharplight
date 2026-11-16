using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.PyTorch;

/// <summary>
/// Importer for RRDBNet super-resolution checkpoints in the PyTorch layout of ESRGAN and Real-ESRGAN (<c>RealESRGAN_x4plus.pth</c>, <c>RealESRGAN_x2plus.pth</c>,
/// ESRGAN <c>.pth</c> files with the BasicSR names), read with the restricted unpickler (nothing is executed). The weights keep their names and become the
/// canonical <c>rrdbnet</c> model source documented on <c>RrdbNetAdapter</c>; the net scale follows from the input channels of the first convolution
/// (3: ×4, 12: ×2, 48: ×1, the networks that pixel-unshuffle their input).
/// </summary>
public sealed partial class SuperResolutionImporter : IModelImporter
{
    /// <summary>The format id.</summary>
    public const string FormatName = "esrgan";

    private static readonly string[] s_wrappers = ["params_ema.", "params."];

    private static readonly TorchCheckpointOptions s_options = new() { StateDictKey = null };

    /// <inheritdoc/>
    public string Format => FormatName;

    /// <inheritdoc/>
    public int Priority => 10;

    /// <inheritdoc/>
    public bool CanOpen(string path)
    {
        string extension = Path.GetExtension(path);
        if (!File.Exists(path) || !(extension.Equals(".pth", StringComparison.OrdinalIgnoreCase) || extension.Equals(".pt", StringComparison.OrdinalIgnoreCase)))
        {
            return false;
        }

        try
        {
            using var file = TorchCheckpointFile.Open(path, s_options);
            return file.Tensors.Keys.Any(k => k.EndsWith("conv_first.weight", StringComparison.Ordinal)) && file.Tensors.Keys.Any(k => k.EndsWith("body.0.rdb1.conv1.weight", StringComparison.Ordinal));
        }
        catch (Exception e) when (e is InvalidDataException or UnsafePickleException or NotSupportedException or IOException)
        {
            return false;
        }
    }

    /// <summary>Opens a super-resolution checkpoint.</summary>
    /// <param name="path">The <c>.pth</c> file.</param>
    /// <returns>The canonical source; it owns the memory-mapped file.</returns>
    /// <exception cref="NotSupportedException">The file is not an RRDBNet checkpoint.</exception>
    public ModelSource Open(string path)
    {
        var file = TorchCheckpointFile.Open(path, s_options);
        try
        {
            string prefix = s_wrappers.FirstOrDefault(w => file.Tensors.Keys.Any(k => k.StartsWith(w, StringComparison.Ordinal))) ?? "";
            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            foreach (var (key, entry) in file.Tensors)
            {
                if (!key.StartsWith(prefix, StringComparison.Ordinal))
                {
                    continue;
                }

                string name = key[prefix.Length..];
                if (entry.DType != TorchDType.Float32)
                {
                    throw new NotSupportedException($"{Path.GetFileName(path)}: tensor '{key}' is {entry.DType}, only float32 super-resolution checkpoints are supported.");
                }

                weights[name] = new WeightInfo
                {
                    Name = name,
                    Type = DataType.F32,
                    Shape = new Shape(entry.Shape.Select(d => (Dim)d)),
                    Source = new TorchTensorSource(file, entry.FileOffset, entry.ByteLength),
                };
            }

            if (!weights.TryGetValue("conv_first.weight", out var first) || !weights.ContainsKey("conv_last.weight"))
            {
                throw new NotSupportedException($"{Path.GetFileName(path)}: not an RRDBNet checkpoint (no conv_first / conv_last).");
            }

            int blocks = weights.Keys.Select(k => BlockName().Match(k)).Where(m => m.Success).Select(m => int.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture) + 1).DefaultIfEmpty(0).Max();
            int inChannels = (int)first.Shape[1].Fixed;
            int scale = inChannels switch
            {
                3 => 4,
                12 => 2,
                48 => 1,
                _ => throw new NotSupportedException($"{Path.GetFileName(path)}: the first convolution takes {inChannels} channels (3, 12 or 48 are supported)."),
            };
            string a = "rrdbnet.";
            return new ModelSource
            {
                Format = FormatName,
                Architecture = "rrdbnet",
                Metadata = new Dictionary<string, object>(StringComparer.Ordinal)
                {
                    [CanonicalKeys.Architecture] = "rrdbnet",
                    [CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(path),
                    [a + "block_count"] = blocks,
                    [a + "scale"] = scale,
                    [a + "default_size"] = 64,
                },
                Weights = weights,
                OwnedResources = [file],
                Path = path,
            };
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    [GeneratedRegex(@"^body\.(\d+)\.rdb1\.conv1\.weight$")]
    private static partial Regex BlockName();
}
