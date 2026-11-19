namespace Sharplight.Architectures;

/// <summary>Values that replace what VoxCPM2 generation would produce, for teacher-forced verification against a reference run.</summary>
internal sealed class VoxCpm2Forcing
{
    /// <summary>The start noise of each patch's diffusion ([feature, patch] row-major, the reference's layout), or <see langword="null"/> for seeded noise.</summary>
    public IReadOnlyList<float[]>? Noise { get; init; }

    /// <summary>The patches ([patch, feature] row-major) that feed the language models and the decoder in place of the diffusion's, or <see langword="null"/>.</summary>
    public IReadOnlyList<float[]>? Patches { get; init; }

    /// <summary>The number of patches to generate regardless of the stop head, or <see langword="null"/> to stop where the stop head says.</summary>
    public int? Steps { get; init; }

    /// <summary>The most patches to generate (the reference's <c>max_len</c>), or <see langword="null"/> for the length bound of the text.</summary>
    public int? MaxSteps { get; init; }
}
