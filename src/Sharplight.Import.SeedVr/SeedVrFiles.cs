namespace Sharplight.Import.SeedVr;

/// <summary>
/// The files of one SeedVR2 model: the DiT checkpoint, the VAE and the precomputed text embeddings, found from a folder or from the DiT
/// file (the others lie beside it). File names follow the official release (<c>seedvr2_ema_3b.pth</c>, <c>ema_vae.pth</c>,
/// <c>pos_emb.pt</c>, <c>neg_emb.pt</c>) and the fp16 / fp8 safetensors conversions (<c>seedvr2_ema_3b_fp16.safetensors</c>,
/// <c>ema_vae_fp16.safetensors</c>).
/// </summary>
/// <param name="Transformer">The DiT checkpoint.</param>
/// <param name="Vae">The VAE checkpoint.</param>
/// <param name="PositiveEmbedding">The precomputed positive text embedding (<c>pos_emb.pt</c>).</param>
/// <param name="NegativeEmbedding">The precomputed negative text embedding (<c>neg_emb.pt</c>), or <see langword="null"/> when absent.</param>
internal sealed record SeedVrFiles(string Transformer, string Vae, string PositiveEmbedding, string? NegativeEmbedding)
{
    private const string TransformerPrefix = "seedvr2_";
    private const string VaePrefix = "ema_vae";
    private const string PositiveFile = "pos_emb.pt";
    private const string NegativeFile = "neg_emb.pt";

    /// <summary>The model files at <paramref name="path"/> (a folder holding one DiT checkpoint, or the DiT file itself), or <see langword="null"/>.</summary>
    /// <param name="path">A folder or file.</param>
    public static SeedVrFiles? TryLocate(string path)
    {
        string? folder;
        string? transformer;
        if (Directory.Exists(path))
        {
            folder = path;
            var candidates = Directory.EnumerateFiles(path).Where(IsTransformerFile).Order(StringComparer.Ordinal).ToList();
            if (candidates.Count != 1)
            {
                return null;
            }

            transformer = candidates[0];
        }
        else if (File.Exists(path) && IsTransformerFile(path))
        {
            folder = Path.GetDirectoryName(Path.GetFullPath(path));
            transformer = path;
        }
        else
        {
            return null;
        }

        if (folder is null)
        {
            return null;
        }

        string? vae = Directory.EnumerateFiles(folder).Where(IsVaeFile).Order(StringComparer.Ordinal).FirstOrDefault();
        string positive = Path.Combine(folder, PositiveFile);
        if (vae is null || !File.Exists(positive))
        {
            return null;
        }

        string negative = Path.Combine(folder, NegativeFile);
        return new SeedVrFiles(transformer, vae, positive, File.Exists(negative) ? negative : null);
    }

    private static bool IsTransformerFile(string path)
    {
        string name = Path.GetFileName(path);
        return name.StartsWith(TransformerPrefix, StringComparison.OrdinalIgnoreCase) && HasWeightExtension(name);
    }

    private static bool IsVaeFile(string path)
    {
        string name = Path.GetFileName(path);
        return name.StartsWith(VaePrefix, StringComparison.OrdinalIgnoreCase) && HasWeightExtension(name);
    }

    /// <summary>Whether <paramref name="name"/> is a PyTorch or safetensors file.</summary>
    /// <param name="name">A file name.</param>
    public static bool IsSafetensors(string name) => name.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase);

    private static bool HasWeightExtension(string name)
        => IsSafetensors(name) || name.EndsWith(".pth", StringComparison.OrdinalIgnoreCase) || name.EndsWith(".pt", StringComparison.OrdinalIgnoreCase);
}
