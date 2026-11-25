namespace Sharplight.Shaders.Lowering;

/// <summary>
/// The byte ranges a provider copies for <see cref="Runtime.ICompiledSegment.CopyStateRows"/>: rows of the outermost dimension of the
/// named state buffers a segment allocated (a state shared by several segments of a graph is copied by the one that allocated it).
/// </summary>
internal static class StateRowCopies
{
    /// <summary>One copy within a state buffer: byte offsets of the source and destination rows and the bytes to copy.</summary>
    /// <param name="program">The segment's program.</param>
    /// <param name="borrowed">State buffers the segment borrows from another segment (skipped).</param>
    /// <param name="states">State names to copy.</param>
    /// <param name="from">First source row.</param>
    /// <param name="to">First destination row.</param>
    /// <param name="count">Rows to copy.</param>
    /// <param name="allocatedBytes">Bytes allocated for a buffer (a KV cache's rows are the compiled context: its size over the row size).</param>
    /// <exception cref="ArgumentOutOfRangeException">A range exceeds a state's rows.</exception>
    public static List<(BufferDecl Buffer, long Source, long Destination, long Bytes)> Regions(KernelProgram program, IReadOnlySet<BufferDecl> borrowed,
        IReadOnlyCollection<string> states, int from, int to, int count, Func<BufferDecl, long> allocatedBytes)
    {
        var regions = new List<(BufferDecl, long, long, long)>();
        foreach (var b in program.Buffers)
        {
            if (b.Kind != BufferKind.State || b.State is not { } st || !states.Contains(st.Name) || borrowed.Contains(b))
            {
                continue;
            }

            // Rows of the outermost dimension: fixed for recurrent states, the compiled context for KV caches ([Ctx, ...]).
            long row = st.Type.ByteSize(st.Shape.Dims.Skip(1).Aggregate(1L, (n, d) => n * d.Fixed));
            long rows = st.Shape[0].IsSymbolic ? allocatedBytes(b) / row : st.Shape[0].Fixed;
            if (from < 0 || to < 0 || (long)from + count > rows || (long)to + count > rows)
            {
                throw new ArgumentOutOfRangeException(nameof(to), $"State '{st.Name}' has {rows} rows.");
            }

            regions.Add((b, from * row, to * row, row * count));
        }

        return regions;
    }
}
