namespace Sharplight.Studio.Services.Runtime;

/// <summary>Where a model load is, for progress bars.</summary>
/// <param name="Fraction">Overall progress, 0 to 1.</param>
/// <param name="Text">The running stages, e.g. <c>Compiling kernels 40 / 107 (35 cached) · Uploading 0.31 / 0.63 GB</c>.</param>
/// <param name="IsDone">The model is ready.</param>
internal sealed record LoadStatus(double Fraction, string Text, bool IsDone = false)
{
    /// <summary>A plain message without a known fraction.</summary>
    public static LoadStatus Message(string text) => new(0, text);
}
