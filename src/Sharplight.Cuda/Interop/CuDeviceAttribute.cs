namespace Sharplight.Cuda.Interop;

/// <summary>The <c>CUdevice_attribute</c> values the provider queries.</summary>
internal enum CuDeviceAttribute
{
    MaxThreadsPerBlock = 1,
    MaxSharedMemoryPerBlock = 8,
    WarpSize = 10,
    MultiprocessorCount = 16,
    Integrated = 18,
    PciBusId = 33,
    PciDeviceId = 34,
    TccDriver = 35,
    MemoryClockRate = 36,
    GlobalMemoryBusWidth = 37,
    L2CacheSize = 38,
    MaxThreadsPerMultiprocessor = 39,
    PciDomainId = 50,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76,
    MaxSharedMemoryPerMultiprocessor = 81,
    MaxSharedMemoryPerBlockOptin = 97,
    ReservedSharedMemoryPerBlock = 111,
    ClusterLaunch = 120,
}
