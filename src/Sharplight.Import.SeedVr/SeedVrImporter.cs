using Sharplight.Import.HuggingFace;
using Sharplight.Import.PyTorch;
using Sharplight.Models;

namespace Sharplight.Import.SeedVr;

/// <summary>
/// Reads SeedVR2 (ByteDance-Seed's one-step video restoration and super-resolution DiT, 3B and 7B) into the canonical <c>seedvr</c> source:
/// a folder (or its DiT file) with the DiT checkpoint (<c>seedvr2_ema_3b.pth</c>, <c>seedvr2_ema_7b.pth</c>, <c>seedvr2_ema_7b_sharp.pth</c>
/// or their <c>_fp16</c> / <c>_fp8_e4m3fn</c> safetensors conversions), the causal video VAE (<c>ema_vae.pth</c> or <c>ema_vae_fp16.safetensors</c>)
/// and the precomputed text embeddings (<c>pos_emb.pt</c>, optionally <c>neg_emb.pt</c>). PyTorch files are read with the restricted unpickler,
/// which never executes code. Names and metadata as documented on <see cref="SeedVrWeights"/> and <see cref="SeedVrHyperParameters"/>.
/// </summary>
public sealed class SeedVrImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "seedvr";

    private static readonly TorchCheckpointOptions s_torchOptions = new() { StateDictKey = null };

    /// <inheritdoc/>
    public string Format => "seedvr";

    /// <inheritdoc/>
    public int Priority => 100;

    /// <inheritdoc/>
    public bool CanOpen(string path) => SeedVrFiles.TryLocate(path) is not null;

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens a SeedVR2 model folder or DiT file.</summary>
    /// <param name="path">The folder, or the DiT file with the VAE and the text embeddings beside it.</param>
    /// <exception cref="InvalidDataException">The files are missing or are not SeedVR2 checkpoints.</exception>
    public static ModelSource Open(string path)
    {
        var files = SeedVrFiles.TryLocate(path) ?? throw new InvalidDataException(
            $"'{path}' is not a SeedVR2 model: expected a seedvr2_*.pth / .safetensors DiT, an ema_vae*.pth / .safetensors VAE and pos_emb.pt in one folder.");
        var owned = new List<IDisposable>();
        try
        {
            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            SeedVrWeights.MapTransformer(OpenFile(files.Transformer, owned).Tensors, weights);
            SeedVrWeights.MapVae(OpenFile(files.Vae, owned).Tensors, weights);
            SeedVrWeights.MapEmbedding(OpenFile(files.PositiveEmbedding, owned).Tensors, SeedVrWeights.PositiveEmbedding, weights);
            if (files.NegativeEmbedding is not null)
            {
                SeedVrWeights.MapEmbedding(OpenFile(files.NegativeEmbedding, owned).Tensors, SeedVrWeights.NegativeEmbedding, weights);
            }

            var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Architecture] = Architecture,
                [CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(files.Transformer),
            };
            SeedVrHyperParameters.Add(weights, metadata);
            return new ModelSource
            {
                Format = "seedvr",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = weights,
                OwnedResources = owned,
                Path = path,
            };
        }
        catch
        {
            foreach (var resource in owned)
            {
                resource.Dispose();
            }

            throw;
        }
    }

    private static SafetensorsFile OpenFile(string path, List<IDisposable> owned)
    {
        var file = SeedVrFiles.IsSafetensors(path) ? SafetensorsFile.Open(path) : SafetensorsFile.OpenTorch(path, s_torchOptions);
        owned.Add(file);
        return file;
    }
}
