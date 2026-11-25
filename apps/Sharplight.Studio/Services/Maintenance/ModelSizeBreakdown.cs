namespace Sharplight.Studio.Services.Maintenance;

/// <summary>Where a model's bytes are: weights, attachments, precompiled artifacts per provider and device, and the rest.</summary>
/// <param name="TotalBytes">The size on disk.</param>
/// <param name="WeightBytes">The weights.</param>
/// <param name="AttachmentBytes">Attachments (tokenizer files, scheduler configuration, filter banks).</param>
/// <param name="Artifacts">The precompiled artifacts, per provider and device.</param>
/// <param name="ReclaimableBytes">Dead space a compaction removes (superseded artifact sets, old section tables).</param>
/// <param name="IsPackage">The model is a <c>.slm</c> package (otherwise only the total is known).</param>
internal sealed record ModelSizeBreakdown(long TotalBytes, long WeightBytes, long AttachmentBytes, IReadOnlyList<ArtifactUsage> Artifacts, long ReclaimableBytes, bool IsPackage)
{
    /// <summary>The precompiled artifacts in total.</summary>
    public long ArtifactBytes => Artifacts.Sum(item => item.Bytes);

    /// <summary>Definition, weight index, description and padding: what is left of <see cref="TotalBytes"/>.</summary>
    public long OtherBytes => Math.Max(0, TotalBytes - WeightBytes - AttachmentBytes - ArtifactBytes - ReclaimableBytes);
}
