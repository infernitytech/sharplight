namespace Sharplight.Onnx.Conformance;

/// <summary>Outcome of one node test on one provider.</summary>
/// <param name="Status">Result class.</param>
/// <param name="Message">Failure detail; empty on success.</param>
internal sealed record NodeTestOutcome(NodeTestStatus Status, string Message)
{
    public static readonly NodeTestOutcome Passed = new(NodeTestStatus.Pass, "");
}
