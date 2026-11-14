using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Settings of <see cref="StreamHyperConnections"/>, read from the <c>{arch}.hyper_connections.*</c> keys shared by every adapter that
/// supports them: <c>streams</c> (N, 0 = off), <c>eps</c> (added to the collapse and mixing weights), <c>magnitude</c> (upper bound of the
/// expansion weights, default 2), <c>post_offset</c> (added to the expansion weights, default 0; Hunyuan v4 adds eps),
/// <c>sinkhorn_iterations</c> (0 = no stream mixing matrix; DeepSeek-V4 uses 20), the RMS norm epsilon of the stream norm, and the
/// variant keys <c>pre_eps</c> (added to the collapse weights, default eps), <c>mixing_eps</c> (added to the softmax of the mixing logits,
/// default eps), <c>mixing_clamp</c> (bound of the mixing logits before the softmax, 0 = none) and <c>mixing_transposed</c> (stream k reads
/// row k of the mixing matrix instead of column k).
/// </summary>
/// <param name="Streams">Residual streams N.</param>
/// <param name="NormEps">Epsilon of the RMS norm over all streams.</param>
/// <param name="Eps">Added to the Sinkhorn denominators.</param>
/// <param name="Magnitude">Factor of the sigmoid expansion weights.</param>
/// <param name="PostOffset">Added to the expansion weights.</param>
/// <param name="SinkhornIterations">Sinkhorn normalizations of the stream mixing matrix; 0 when the module has none.</param>
/// <param name="PreEps">Added to the collapse weights.</param>
/// <param name="MixingEps">Added to the row softmax of the mixing logits.</param>
/// <param name="MixingClamp">The mixing logits are clamped to ±this before the softmax; 0 = no clamp.</param>
/// <param name="MixingTransposed">Stream k after a block mixes the old streams with row k of the mixing matrix (Σ_j C[k, j]·X_j).</param>
internal sealed record StreamHyperConnectionOptions(int Streams, float NormEps, float Eps, float Magnitude, float PostOffset, int SinkhornIterations,
    float PreEps, float MixingEps, float MixingClamp, bool MixingTransposed)
{
    /// <summary>Reads the options of <paramref name="src"/>; null when the model has no hyper-connections.</summary>
    /// <param name="src">The model source.</param>
    /// <param name="normEps">The model's RMS norm epsilon (the default of <c>hyper_connections.norm_eps</c>).</param>
    public static StreamHyperConnectionOptions? Read(ModelSource src, float normEps)
    {
        int streams = src.Arch("hyper_connections.streams", 0);
        float eps = src.Arch("hyper_connections.eps", 1e-6f);
        return streams <= 0 ? null : new StreamHyperConnectionOptions(
            streams,
            src.Arch("hyper_connections.norm_eps", normEps),
            eps,
            src.Arch("hyper_connections.magnitude", 2f),
            src.Arch("hyper_connections.post_offset", 0f),
            src.Arch("hyper_connections.sinkhorn_iterations", 0),
            src.Arch("hyper_connections.pre_eps", eps),
            src.Arch("hyper_connections.mixing_eps", eps),
            src.Arch("hyper_connections.mixing_clamp", 0f),
            src.Arch("hyper_connections.mixing_transposed", false));
    }
}
