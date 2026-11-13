namespace Sharplight.Runtime;

/// <summary>Estimated device memory of a compiled model, in bytes.</summary>
public readonly record struct ModelFootprint(long WeightBytes, long StateBytes, long ActivationBytes)
{
    public long Total => WeightBytes + StateBytes + ActivationBytes;
    public override string ToString() => $"{Total / 1048576.0:F0} MiB (weights {WeightBytes / 1048576.0:F0}, state {StateBytes / 1048576.0:F0}, activations ~{ActivationBytes / 1048576.0:F0})";
}
