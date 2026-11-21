using System.Runtime.InteropServices;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// The experts the rows of a sparse segment's pass select (<see cref="Runtime.CompileOptions.SparseExpertSlots"/>), in slot order: the ids
/// input is rewritten to the slots their experts are copied to.
/// </summary>
/// <param name="experts">The experts of the banks.</param>
/// <param name="slots">The experts a bank holds.</param>
internal sealed class SparseExpertSelection(int experts, int slots)
{
    private readonly List<int> _selected = [];

    /// <summary>The selected experts of the last <see cref="Remap"/>, in slot order.</summary>
    public IReadOnlyList<int> Selected => _selected;

    /// <summary>
    /// Rewrites expert ids to slots (ids outside the bank, such as -1, select no expert and stay -1), remembering the experts in slot order.
    /// </summary>
    /// <exception cref="InvalidOperationException">The rows select more experts than the banks hold.</exception>
    public byte[] Remap(ReadOnlySpan<byte> data)
    {
        var ids = MemoryMarshal.Cast<byte, int>(data);
        var remapped = new byte[data.Length];
        var slotIds = MemoryMarshal.Cast<byte, int>(remapped.AsSpan());
        _selected.Clear();
        for (int i = 0; i < ids.Length; i++)
        {
            int e = ids[i];
            if ((uint)e >= (uint)experts)
            {
                slotIds[i] = -1;
                continue;
            }

            int slot = _selected.IndexOf(e);
            if (slot < 0)
            {
                if (_selected.Count == slots)
                {
                    throw new InvalidOperationException($"The rows select more than the {slots} experts a sparse segment holds.");
                }

                slot = _selected.Count;
                _selected.Add(e);
            }

            slotIds[i] = slot;
        }

        return remapped;
    }
}
