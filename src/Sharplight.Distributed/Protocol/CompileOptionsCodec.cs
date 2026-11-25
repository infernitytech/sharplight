using Sharplight.Runtime;

namespace Sharplight.Distributed.Protocol;

/// <summary>The device-independent compile options a coordinator sends with a segment (bounds, precision, feature switches).</summary>
internal static class CompileOptionsCodec
{
    public static void Write(PayloadWriter w, CompileOptions options)
    {
        w.Int64(options.MaxTokens).Int64(options.ContextLength).Map(options.SymbolBounds).Strings([.. options.DisabledFeatures]);
        w.Bool(options.AllowReducedPrecision).Bool(options.SplitHalfMatMul).Int32((int)options.MatrixPrecision).Int64(options.RowInvariantRows);
        w.Int32((int)options.ActivationPrecision);
    }

    /// <summary>The options as sent; the worker adds its own weight sharing, expert store and cancellation.</summary>
    public static CompileOptions Read(PayloadReader r, SharedWeights? sharedWeights, Runtime.Experts.ExpertStore? diskExperts, CancellationToken cancellationToken)
    {
        long maxTokens = r.Int64();
        long context = r.Int64();
        var bounds = new Dictionary<string, long>(StringComparer.Ordinal);
        r.Map(bounds);
        var disabled = r.Strings().ToHashSet(StringComparer.Ordinal);
        return new CompileOptions
        {
            MaxTokens = maxTokens,
            ContextLength = context,
            SymbolBounds = bounds,
            DisabledFeatures = disabled,
            AllowReducedPrecision = r.Bool(),
            SplitHalfMatMul = r.Bool(),
            MatrixPrecision = (MatrixPrecision)r.Int32(),
            RowInvariantRows = r.Int64(),
            ActivationPrecision = (ActivationPrecision)r.Int32(),
            SharedWeights = sharedWeights,
            DiskExperts = diskExperts,
            CancellationToken = cancellationToken,
        };
    }
}
