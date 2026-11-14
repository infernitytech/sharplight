namespace Sharplight.Onnx.Conformance;

/// <summary>Result class of one node test on one provider.</summary>
internal enum NodeTestStatus
{
    /// <summary>All outputs of all data sets match.</summary>
    Pass,

    /// <summary>The model ran but an output differs (shape or values).</summary>
    Wrong,

    /// <summary>The importer rejected the model (unsupported op, type or shape).</summary>
    ImportFail,

    /// <summary>The provider could not compile the graph (no kernel for a node).</summary>
    CompileFail,

    /// <summary>Running the compiled model threw.</summary>
    RunFail,

    /// <summary>The test data uses a value kind the harness cannot feed (sequence, map, optional or string inputs).</summary>
    HarnessUnsupported,
}
