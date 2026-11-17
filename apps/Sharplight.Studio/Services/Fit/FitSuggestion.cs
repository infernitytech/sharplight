using Sharplight.Hosting.Fit;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;

namespace Sharplight.Studio.Services.Fit;

/// <summary>A change that makes a model fit better, with the verdict it leads to.</summary>
/// <param name="Text">What to change, e.g. <c>Context of 16,384 tokens · needs about 9.1 GB</c>.</param>
/// <param name="Verdict">The verdict with the change.</param>
/// <param name="Settings">The load settings with the change applied, or null.</param>
/// <param name="Variant">Another library model to load instead (a smaller quantization), or null.</param>
internal sealed record FitSuggestion(string Text, FitVerdict Verdict, LoadSettings? Settings = null, LibraryEntry? Variant = null);
