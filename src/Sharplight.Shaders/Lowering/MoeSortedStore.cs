using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// The sorted f16 store of a dual expert GEMM (gate / up on the hand-written grouped MXFP4 GEMM, <c>gemm_moe_native.cu</c>
/// NM_Y16): its group's output written as the expert-sorted f16 rows the next expert product (the down projection) reads, instead
/// of f32 rows per item that the down projection's gather converts. Created with the gate / up launches; the down projection
/// sets <see cref="Consumer"/> when it reads those rows on the same routing, which switches the store on for those batches.
/// </summary>
internal sealed class MoeSortedStore
{
    /// <summary>The group output the f32 form writes (the down projection's activations).</summary>
    public required Value Output { get; init; }

    /// <summary>The f32 tensor of <see cref="Output"/>; the sorted store spills the values a row-ranged row must scale there.</summary>
    public required TensorRef Spill { get; init; }

    /// <summary>The expert-sorted f16 rows ([items, K], row-range scale words after them when <see cref="RangeScales"/> is set).</summary>
    public required TensorRef Rows { get; init; }

    /// <summary>The byte offset of the rows' scale words in <see cref="Rows"/>'s buffer, or null for plain f16 rows.</summary>
    public required long? RangeScales { get; init; }

    /// <summary>The routing the gate / up launches read (see the lowerer's routing serial): the down projection must read the same.</summary>
    public required int RouteSerial { get; init; }

    /// <summary>Outputs per row (the down projection's K).</summary>
    public required long Width { get; init; }

    /// <summary>The largest item count (tokens times slots).</summary>
    public required long MaxItems { get; init; }

    /// <summary>The batches the gate / up products run on the hand-written grouped GEMM.</summary>
    public required Func<Bindings, bool> Native { get; init; }

    /// <summary>The batches the down projection reads expert-sorted f16 rows in (null until it is lowered, or when it reads f32 rows).</summary>
    public Func<Bindings, bool>? Consumer { get; set; }

    /// <summary>True for the batches the sorted store serves: native gate / up products whose rows the down projection reads sorted.</summary>
    public bool Serves(Bindings b) => Native(b) && (Consumer?.Invoke(b) ?? false);
}
