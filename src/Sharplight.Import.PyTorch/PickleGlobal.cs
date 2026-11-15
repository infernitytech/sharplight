namespace Sharplight.Import.PyTorch;

/// <summary>The only callables a checkpoint pickle may name. Each is implemented by the unpickler itself; none is looked up by reflection.</summary>
internal enum PickleGlobal
{
    /// <summary><c>torch._utils._rebuild_tensor_v2</c> (and the older <c>_rebuild_tensor</c>).</summary>
    RebuildTensor,

    /// <summary><c>torch._utils._rebuild_parameter</c>: returns its first argument.</summary>
    RebuildParameter,

    /// <summary><c>collections.OrderedDict</c>.</summary>
    OrderedDict,

    /// <summary><c>torch.Size</c>: a tuple of ints.</summary>
    TorchSize,

    /// <summary><c>_codecs.encode</c> restricted to latin1 (how protocol 2 spells bytes).</summary>
    CodecsEncode,
}
