namespace Sharplight.Onnx.Conformance;

/// <summary>Outcome of one node test on one provider.</summary>
/// <param name="Status">Result class.</param>
/// <param name="Message">Failure detail; empty on success.</param>
internal sealed record NodeTestOutcome(NodeTestStatus Status, string Message)
{
    public static readonly NodeTestOutcome Passed = new(NodeTestStatus.Pass, "");

    /// <summary>Inputs that had to be pinned to their test values at import (shape-defining inputs; see NodeTestRunner).</summary>
    public string[] Pinned { get; init; } = [];

    /// <summary>IR operators that were placed on the fallback device (empty when everything ran on the tested device).</summary>
    public string[] FallbackOps { get; init; } = [];
}
