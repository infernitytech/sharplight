/// <summary>The members of a provider's compiled segment the benchmark uses (Vulkan and CUDA segments).</summary>
internal sealed record SegmentView(Func<double> Gpu, Action<bool> SetProfile, Action Reset, Func<int, string> Report, Func<long, (int Dispatches, int Barriers)> Stats);
