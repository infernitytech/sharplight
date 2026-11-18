using Sharplight.ModelFeed.Index;

namespace Sharplight.Studio.Services.FirstRun;

/// <summary>One recommended package.</summary>
/// <param name="Role">What it is for.</param>
/// <param name="Package">The feed entry.</param>
/// <param name="DownloadBytes">Bytes to download: the package and the required dependencies not already counted by an earlier item.</param>
/// <param name="MemoryBytes">Memory the package needs to run.</param>
/// <param name="OnGpu">It runs on the GPU (else on the CPU).</param>
/// <param name="Preselected">It is ticked by default.</param>
/// <param name="DownloadTime">The estimated download time at the assumed connection speed.</param>
internal sealed record StarterItem(StarterRole Role, FeedPackageSummary Package, long DownloadBytes, long MemoryBytes, bool OnGpu, bool Preselected, TimeSpan DownloadTime);
