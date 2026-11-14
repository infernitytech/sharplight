namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>A navigable reference from the selected IR line: the producer of a value it reads, a consumer of a value it defines, or a dataflow neighbor.</summary>
/// <param name="Label">What the reference is (e.g. <c>%12 from</c>, <c>%12 used by</c>, or a level marker).</param>
/// <param name="Target">The line to jump to.</param>
internal sealed record IrReferenceRow(string Label, IrLineRow Target);
