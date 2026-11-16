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
        path = CheckpointOf(path);
        string extension = Path.GetExtension(path);
        if (!File.Exists(path) || !(extension.Equals(".pth", StringComparison.OrdinalIgnoreCase) || extension.Equals(".pt", StringComparison.OrdinalIgnoreCase)))
        {
            return false;
        }

        try
        {
            using var file = TorchCheckpointFile.Open(path, s_options);
            return file.Tensors.Keys.Any(k => k.EndsWith("conv_first.weight", StringComparison.Ordinal))
                && file.Tensors.Keys.Any(k => k.EndsWith("body.0.rdb1.conv1.weight", StringComparison.Ordinal) || k.EndsWith("layers.0.residual_group.blocks.0.attn.qkv.weight", StringComparison.Ordinal));
        }
        catch (Exception e) when (e is InvalidDataException or UnsafePickleException or NotSupportedException or IOException)
        {
            return false;
        }
    }

    // A folder (as downloaded from a model repository) holds its checkpoint as the only .pth / .pt file.
    private static string CheckpointOf(string path)
    {
        if (!Directory.Exists(path))
        {
            return path;
        }

        var files = Directory.EnumerateFiles(path, "*.pth").Concat(Directory.EnumerateFiles(path, "*.pt")).ToList();
        return files.Count == 1 ? files[0] : path;
    }

    /// <summary>Opens a super-resolution checkpoint.</summary>
    /// <param name="path">The <c>.pth</c> file, or a folder holding only one.</param>
    /// <returns>The canonical source; it owns the memory-mapped file.</returns>
    /// <exception cref="NotSupportedException">The file is not an RRDBNet checkpoint.</exception>
    public ModelSource Open(string path)
    {
        path = CheckpointOf(path);
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
                if (IsRebuiltBuffer(name))
                {
                    continue;
                }

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
                throw new NotSupportedException($"{Path.GetFileName(path)}: not an RRDBNet or SwinIR checkpoint (no conv_first / conv_last).");
            }

            if (weights.ContainsKey("layers.0.residual_group.blocks.0.attn.qkv.weight"))
            {
                return OpenSwinIr(path, weights, file);
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

    // SwinIR stores its relative position indices (int64) and shift masks as buffers; the adapter rebuilds both for the graph's size.
    private static bool IsRebuiltBuffer(string name) => name.EndsWith(".relative_position_index", StringComparison.Ordinal) || name.EndsWith(".attn_mask", StringComparison.Ordinal);

    // SwinIR checkpoints of the classical (sub-pixel convolution) and real-world x4 (nearest + convolution) variants with the 1conv residual connection; the
    // window size follows from the relative position table, the upscale factor from the upsampler's convolutions.
    private static ModelSource OpenSwinIr(string path, Dictionary<string, WeightInfo> weights, TorchCheckpointFile file)
    {
        string name = Path.GetFileName(path);
        if (!weights.TryGetValue("layers.0.conv.weight", out var residual) || residual.Shape.Rank != 4)
        {
            throw new NotSupportedException($"{name}: only SwinIR checkpoints with the 1conv residual connection are supported.");
        }

        long rows = weights["layers.0.residual_group.blocks.0.attn.relative_position_bias_table"].Shape[0].Fixed;
        int window = ((int)Math.Round(Math.Sqrt(rows)) + 1) / 2;
        string upsampler;
        int scale;
        if (weights.ContainsKey("conv_up1.weight") && weights.ContainsKey("conv_up2.weight") && weights.ContainsKey("conv_hr.weight"))
        {
            upsampler = "nearest+conv";
            scale = 4;
        }
        else if (weights.TryGetValue("upsample.0.weight", out var up) && weights.TryGetValue("conv_before_upsample.0.weight", out var before))
        {
            upsampler = "pixelshuffle";
            long ratio = up.Shape[0].Fixed / before.Shape[0].Fixed;
            int stages = 0;
            while (weights.ContainsKey($"upsample.{2 * stages}.weight"))
            {
                stages++;
            }

            scale = ratio == 9 ? 3 : 1 << stages;
        }
        else
        {
            throw new NotSupportedException($"{name}: SwinIR checkpoints without an upsampler (denoising, JPEG artifact removal) are not supported.");
        }

        string a = "swinir.";
        return new ModelSource
        {
            Format = FormatName,
            Architecture = "swinir",
            Metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Architecture] = "swinir",
                [CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(path),
                [a + "window_size"] = window,
                [a + "scale"] = scale,
                [a + "upsampler"] = upsampler,
                [a + "img_range"] = 1f,
                [a + "default_size"] = 64,
            },
            Weights = weights,
            OwnedResources = [file],
            Path = path,
        };
    }

    [GeneratedRegex(@"^body\.(\d+)\.rdb1\.conv1\.weight$")]
    private static partial Regex BlockName();
}
