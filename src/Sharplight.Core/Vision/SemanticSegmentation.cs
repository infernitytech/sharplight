namespace Sharplight.Vision;

/// <summary>The result of <see cref="ISemanticSegmenter.SegmentAsync"/>: a class index per pixel, row-major.</summary>
/// <param name="Width">Width of the label map in pixels.</param>
/// <param name="Height">Height of the label map in pixels.</param>
/// <param name="Labels">The class index of every pixel, <c>Width · Height</c> entries.</param>
/// <param name="ClassNames">The class names by index (may be empty).</param>
public sealed record SemanticSegmentation(int Width, int Height, int[] Labels, IReadOnlyList<string> ClassNames)
{
    /// <summary>The class scores at the model's output resolution, class-major (<c>ClassCount · LogitsHeight · LogitsWidth</c>), when requested.</summary>
    public float[]? Logits { get; init; }

    /// <summary>Width of <see cref="Logits"/>.</summary>
    public int LogitsWidth { get; init; }

    /// <summary>Height of <see cref="Logits"/>.</summary>
    public int LogitsHeight { get; init; }

    /// <summary>The number of classes the model scores.</summary>
    public int ClassCount { get; init; }

    /// <summary>For matting models (one output channel): the foreground probability per pixel, <c>Width · Height</c> entries; otherwise <see langword="null"/>.</summary>
    public float[]? Alpha { get; init; }

    /// <summary>The name of a class, or <c>class N</c> when the model has no names.</summary>
    /// <param name="classId">The class index.</param>
    public string NameOf(int classId) => classId >= 0 && classId < ClassNames.Count ? ClassNames[classId] : $"class {classId}";

    /// <summary>The classes that occur in the label map with their pixel counts, largest first.</summary>
    public IReadOnlyList<(int ClassId, string Name, int Pixels)> Classes
        => [.. Labels.GroupBy(l => l).Select(g => (g.Key, NameOf(g.Key), g.Count())).OrderByDescending(c => c.Item3)];

    /// <summary>The mask of one class: 255 where the pixel has the class, 0 elsewhere (one byte per pixel).</summary>
    /// <param name="classId">The class index.</param>
    public byte[] Mask(int classId)
    {
        var mask = new byte[Labels.Length];
        for (int i = 0; i < mask.Length; i++)
        {
            mask[i] = Labels[i] == classId ? (byte)255 : (byte)0;
        }

        return mask;
    }
}
