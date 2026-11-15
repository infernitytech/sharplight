using Sharplight.Import.PyTorch;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// PyTorch checkpoints (<c>.ckpt</c>, <c>.pt</c>, <c>.pth</c>) as weight views, read with the restricted unpickler of
/// Sharplight.Import.PyTorch, which never executes code. Training checkpoints keep PyTorch Lightning callbacks and numpy scalars
/// (the global step) beside the weights; those globals load as inert placeholders.
/// </summary>
internal static class TorchCheckpoints
{
    /// <summary>The reader options: the state dict under <c>state_dict</c>, Lightning and numpy objects inert.</summary>
    public static TorchCheckpointOptions Options { get; } = new() { InertModulePrefixes = ["pytorch_lightning", "lightning", "numpy"] };

    /// <summary>Whether <paramref name="path"/> has a PyTorch checkpoint extension.</summary>
    /// <param name="path">A file.</param>
    public static bool HasExtension(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Equals(".ckpt", StringComparison.OrdinalIgnoreCase) || extension.Equals(".pt", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".pth", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Opens a checkpoint with <see cref="Options"/>.</summary>
    /// <param name="path">The file.</param>
    public static TorchCheckpointFile Open(string path) => TorchCheckpointFile.Open(path, Options);

    /// <summary>A zero-copy weight view of <paramref name="entry"/>, or <see langword="null"/> for dtypes without a Sharplight equivalent.</summary>
    /// <param name="file">The open checkpoint.</param>
    /// <param name="entry">One of its tensors.</param>
    /// <param name="name">The weight name.</param>
    public static WeightInfo? ToWeightInfo(TorchCheckpointFile file, TorchTensorEntry entry, string name)
    {
        DataType? type = entry.DType switch
        {
            TorchDType.Float32 => DataType.F32,
            TorchDType.Float16 => DataType.F16,
            TorchDType.BFloat16 => DataType.BF16,
            TorchDType.Float64 => DataType.F64,
            TorchDType.Int64 => DataType.I64,
            TorchDType.Int32 => DataType.I32,
            TorchDType.Int16 => DataType.I16,
            TorchDType.Int8 => DataType.I8,
            TorchDType.UInt8 => DataType.U8,
            TorchDType.Bool => DataType.Bool,
            _ => null,
        };
        if (type is null)
        {
            return null;
        }

        return new WeightInfo
        {
            Name = name,
            Type = type,
            Shape = new Shape(entry.Shape.Select(d => (Dim)d)),
            Source = new TorchWeightSource(file, entry.FileOffset, entry.ByteLength),
        };
    }
}
