using Sharplight.Shaders.Lowering;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The barrier placement of <see cref="HazardTracker.Order(IReadOnlyList{KernelLaunch}, Bindings, bool)"/> against a reference of the
/// rule it implements (barrier batches filled in program order, launches deferred past a conflict within the lookahead), on random
/// launch lists with shared buffers, overlapping ranges and write groups. No GPU needed.
/// </summary>
public sealed class HazardOrderTests
{
    private const int Lookahead = 32;

    /// <summary>Random programs from chains to wide independent fans: the same order and barriers as the reference.</summary>
    [Theory]
    [InlineData(1, 40, 3)]
    [InlineData(2, 300, 6)]
    [InlineData(3, 600, 24)]
    [InlineData(4, 900, 64)]
    public void MatchesReference(int seed, int count, int bufferCount)
    {
        var random = new Random(seed);
        var buffers = Enumerable.Range(0, bufferCount)
            .Select(i => new BufferDecl { Id = i, Kind = BufferKind.Activations, Name = $"b{i}", Bytes = 4096 })
            .ToArray();
        var groups = new[] { new object(), new object() };
        var variant = new KernelVariant(_ => true, new Sharplight.Shaders.CompiledKernel("k", "main", Sharplight.Shaders.ShaderTarget.Ptx, [], "k"),
            _ => new Grid(1, 1, 1));
        BufferRange Range()
        {
            long start = random.Next(0, 64) * 64;
            return new BufferRange(buffers[random.Next(bufferCount)], start, start + random.Next(1, 16) * 64);
        }

        var launches = Enumerable.Range(0, count).Select(i => new KernelLaunch
        {
            Name = $"l{i}",
            Variants = [variant],
            Args = [],
            Reads = [.. Enumerable.Range(0, random.Next(0, 6)).Select(_ => Range())],
            Writes = [.. Enumerable.Range(0, random.Next(1, 3)).Select(_ => Range())],
            WriteGroup = random.Next(4) == 0 ? groups[random.Next(2)] : null,
        }).ToList();

        var bindings = new Bindings(new Dictionary<string, long>());
        var expected = Reference(launches);
        var actual = HazardTracker.Order(launches, bindings, reorder: true).Select(o => (o.Index, o.BarrierBefore)).ToList();
        Assert.Equal(expected, actual);
        var cached = HazardTracker.Order(launches, bindings, reorder: true, [], 0).Select(o => (o.Index, o.BarrierBefore)).ToList();
        Assert.Equal(expected, cached);
    }

    /// <summary>The original placement: pending accesses as lists, every check against the union of a batch's or the deferred launches' accesses.</summary>
    private static List<(int Index, bool BarrierBefore)> Reference(List<KernelLaunch> launches)
    {
        bool Conflicts(KernelLaunch launch, List<KernelLaunch> set)
            => set.Any(m => launch.Reads.Any(r => m.Writes.Any(w => w.Overlaps(r)))
                            || launch.Writes.Any(w => m.Writes.Any(o => o.Overlaps(w) && (m.WriteGroup is null || !ReferenceEquals(m.WriteGroup, launch.WriteGroup)))
                                                      || m.Reads.Any(r => r.Overlaps(w))));

        var order = new List<(int, bool)>();
        var pending = Enumerable.Range(0, launches.Count).ToList();
        bool barrierBefore = false;
        while (pending.Count > 0)
        {
            var batch = new List<KernelLaunch>();
            var deferred = new List<KernelLaunch>();
            var next = new List<int>();
            for (int i = 0; i < pending.Count; i++)
            {
                if (deferred.Count >= Lookahead)
                {
                    next.AddRange(pending.Skip(i));
                    break;
                }

                var launch = launches[pending[i]];
                if (batch.Count == 0 || (!Conflicts(launch, batch) && !Conflicts(launch, deferred)))
                {
                    order.Add((pending[i], barrierBefore && batch.Count == 0));
                    batch.Add(launch);
                }
                else
                {
                    deferred.Add(launch);
                    next.Add(pending[i]);
                }
            }

            pending = next;
            barrierBefore = true;
        }

        return order;
    }
}
