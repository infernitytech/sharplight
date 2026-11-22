namespace Sharplight.Distributed;

/// <summary>How a segment worker checks that its copy of the model matches the coordinator's before it compiles a segment.</summary>
public enum WeightVerification
{
    /// <summary>
    /// Names, types and shapes of the segment's weights plus three 4 KiB samples (start, middle, end) of every file-backed weight: cheap,
    /// catches another model, quantization or revision (the default).
    /// </summary>
    Sampled,
    /// <summary>Every byte of the segment's weights on both sides (reads the segment's weights once on the coordinator and the worker).</summary>
    Full,
    /// <summary>Only names, types and shapes.</summary>
    Metadata,
}
