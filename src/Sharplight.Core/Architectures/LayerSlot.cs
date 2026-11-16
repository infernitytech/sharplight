namespace Sharplight.Architectures;

/// <summary>
/// Position of one decoder layer pass: the physical layer whose weights and hyper-parameters it uses and the logical slot that owns
/// its KV cache (they differ when layer loops run the physical layers several times, and for the draft layer after the trunk).
/// </summary>
/// <param name="Layer">Physical layer index (weights <c>layers.{Layer}.</c>, per-layer hyper-parameters).</param>
/// <param name="Slot">KV-cache slot of this pass.</param>
internal readonly record struct LayerSlot(int Layer, int Slot);
