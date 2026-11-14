using System.Text;

namespace Sharplight.Onnx;

/// <summary>
/// Finds the operator sets an ONNX model imports without reading its graph, e.g. from ranges of a file that is not downloaded yet.
/// It walks only the top-level fields of <c>ModelProto</c>: the graph (and every other length-delimited field but
/// <c>opset_import</c>) is skipped by its length, so after the first bytes the caller is asked for the bytes behind the graph, where
/// writers put the opset imports.
/// </summary>
/// <remarks>
/// Feed the bytes that start at <see cref="Position"/> until <see cref="IsComplete"/> (or give up). Operators are not checked: whether
/// each one is supported is known only when the model is imported.
/// </remarks>
public sealed class OnnxOpsetScanner
{
    private const int OpsetImportField = 8;
    private const int DomainField = 1;
    private const int VersionField = 2;

    private readonly List<OnnxOpset> _opsets = [];

    /// <summary>Creates a scanner for a file of <paramref name="fileLength"/> bytes.</summary>
    /// <param name="fileLength">The size of the <c>.onnx</c> file.</param>
    public OnnxOpsetScanner(long fileLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(fileLength);
        FileLength = fileLength;
    }

    /// <summary>The size of the file.</summary>
    public long FileLength { get; }

    /// <summary>The file offset of the next bytes the scanner needs.</summary>
    public long Position { get; private set; }

    /// <summary>Every top-level field was read (or skipped).</summary>
    public bool IsComplete => !IsMalformed && Position >= FileLength;

    /// <summary>The bytes do not form a <c>ModelProto</c>; feeding more does nothing.</summary>
    public bool IsMalformed { get; private set; }

    /// <summary>The operator sets found so far, in file order.</summary>
    public IReadOnlyList<OnnxOpset> Opsets => _opsets;

    /// <summary>Reads the bytes that start at <see cref="Position"/> and advances past every field they hold completely.</summary>
    /// <param name="bytes">Bytes of the file starting at <see cref="Position"/> (any length).</param>
    public void Feed(ReadOnlySpan<byte> bytes)
    {
        int position = 0;
        while (!IsMalformed && Position + position < FileLength)
        {
            int start = position;
            if (!TryReadVarint(bytes, ref position, out ulong key))
            {
                break;
            }

            int field = (int)(key >> 3);
            long end;
            switch ((int)(key & 7))
            {
                case 0:
                    if (!TryReadVarint(bytes, ref position, out _))
                    {
                        Position += start;
                        return;
                    }

                    end = position;
                    break;
                case 1:
                    end = (long)position + 8;
                    break;
                case 5:
                    end = (long)position + 4;
                    break;
                case 2:
                    if (!TryReadVarint(bytes, ref position, out ulong length) || length > (ulong)FileLength)
                    {
                        IsMalformed = length > (ulong)FileLength;
                        Position += start;
                        return;
                    }

                    end = position + (long)length;
                    if (field == OpsetImportField)
                    {
                        if (end > bytes.Length)
                        {
                            // Read the whole entry with the next bytes.
                            Position += start;
                            return;
                        }

                        ReadOpset(bytes[position..(int)end]);
                    }

                    break;
                default:
                    IsMalformed = true;
                    return;
            }

            if (field == 0 || Position + end > FileLength)
            {
                IsMalformed = true;
                return;
            }

            if (end > bytes.Length)
            {
                // A skipped field reaches past these bytes: continue behind it.
                Position += end;
                return;
            }

            position = (int)end;
        }

        Position += position;
    }

    private void ReadOpset(ReadOnlySpan<byte> entry)
    {
        string domain = "";
        long version = 0;
        int position = 0;
        while (position < entry.Length)
        {
            if (!TryReadVarint(entry, ref position, out ulong key))
            {
                IsMalformed = true;
                return;
            }

            int field = (int)(key >> 3);
            int wire = (int)(key & 7);
            if (wire == 0)
            {
                if (!TryReadVarint(entry, ref position, out ulong value))
                {
                    IsMalformed = true;
                    return;
                }

                if (field == VersionField)
                {
                    version = (long)value;
                }
            }
            else if (wire == 2 && TryReadVarint(entry, ref position, out ulong length) && length <= (ulong)(entry.Length - position))
            {
                if (field == DomainField)
                {
                    domain = Encoding.UTF8.GetString(entry.Slice(position, (int)length));
                }

                position += (int)length;
            }
            else
            {
                IsMalformed = true;
                return;
            }
        }

        _opsets.Add(new OnnxOpset(domain.Length == 0 ? "ai.onnx" : domain, version));
    }

    private static bool TryReadVarint(ReadOnlySpan<byte> bytes, ref int position, out ulong value)
    {
        value = 0;
        for (int shift = 0, at = position; shift < 64; shift += 7, at++)
        {
            if (at >= bytes.Length)
            {
                return false;
            }

            byte b = bytes[at];
            value |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
            {
                position = at + 1;
                return true;
            }
        }

        return false;
    }
}
