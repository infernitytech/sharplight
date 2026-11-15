using Sharplight.Import.Gguf;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The tensors of one or more checkpoint files under their stored names (safetensors, GGUF or PyTorch), as lazy weight views over the
/// memory-mapped files. The files stay open until <see cref="Resources"/> are disposed (by the model source that takes them over).
/// </summary>
internal sealed class CheckpointTensors
{
    private readonly Dictionary<string, WeightInfo> _tensors = new(StringComparer.Ordinal);
    private readonly List<IDisposable> _resources = [];

    /// <summary>Tensors by stored name.</summary>
    public IReadOnlyDictionary<string, WeightInfo> Tensors => _tensors;

    /// <summary>The open files backing <see cref="Tensors"/>.</summary>
    public IReadOnlyList<IDisposable> Resources => _resources;

    /// <summary>Whether the file is a GGUF file (by its magic bytes).</summary>
    /// <param name="path">The file.</param>
    public static bool IsGguf(string path)
    {
        using var file = File.OpenRead(path);
        Span<byte> magic = stackalloc byte[4];
        return file.Read(magic) == 4 && magic.SequenceEqual("GGUF"u8);
    }

    /// <summary>
    /// Adds the tensors of a safetensors, GGUF or PyTorch checkpoint file; names already present are replaced. Tensors whose stored type has no
    /// Sharplight equivalent are left out (they are never weights of the supported networks, e.g. i64 position ids).
    /// </summary>
    /// <param name="path">The file.</param>
    /// <param name="prefix">Prepended to every stored name (e.g. "unet." for a diffusers component file).</param>
    public void Add(string path, string prefix = "")
    {
        if (IsGguf(path))
        {
            var gguf = GgufFile.Open(path);
            _resources.Add(gguf);
            foreach (var tensor in gguf.Tensors)
            {
                _tensors[prefix + tensor.Name] = Rename(tensor.ToWeightInfo(), prefix + tensor.Name);
            }

            return;
        }

        if (TorchCheckpoints.HasExtension(path))
        {
            var checkpoint = TorchCheckpoints.Open(path);
            _resources.Add(checkpoint);
            foreach (var (name, entry) in checkpoint.Tensors)
            {
                if (TorchCheckpoints.ToWeightInfo(checkpoint, entry, prefix + name) is { } weight)
                {
                    _tensors[prefix + name] = weight;
                }
            }

            return;
        }

        var file = SafetensorsFile.Open(path);
        _resources.Add(file);
        foreach (var (name, tensor) in file.Tensors)
        {
            if (tensor.Type is null)
            {
                continue;
            }

            _tensors[prefix + name] = Rename(tensor.ToWeightInfo(), prefix + name);
        }
    }

    /// <summary>Releases the files (after a failed import).</summary>
    public void DisposeAll()
    {
        foreach (var resource in _resources)
        {
            resource.Dispose();
        }
    }

    private static WeightInfo Rename(WeightInfo weight, string name) => weight.Name == name ? weight : weight.As(name);
}
