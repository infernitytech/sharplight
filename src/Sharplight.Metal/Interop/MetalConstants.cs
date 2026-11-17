namespace Sharplight.Metal.Interop;

/// <summary>Enumeration values of the Metal API the provider passes (from the Metal headers).</summary>
internal static class MetalConstants
{
    /// <summary><c>MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked</c>: CPU and GPU share the pages (unified memory),
    /// and Metal does not track hazards (the provider places its own barriers).</summary>
    internal const nuint SharedUntracked = (0u << 4) | (1u << 8);

    /// <summary><c>MTLResourceUsageRead | MTLResourceUsageWrite</c>.</summary>
    internal const nuint UsageReadWrite = 1 | 2;

    /// <summary><c>MTLDispatchTypeSerial</c>.</summary>
    internal const nuint DispatchSerial = 0;

    /// <summary><c>MTLDispatchTypeConcurrent</c>.</summary>
    internal const nuint DispatchConcurrent = 1;

    /// <summary><c>MTLBarrierScopeBuffers</c>.</summary>
    internal const nuint BarrierScopeBuffers = 1;

    /// <summary><c>MTLCommandBufferStatusCompleted</c>.</summary>
    internal const ulong CommandBufferCompleted = 4;

    /// <summary><c>MTLCommandBufferStatusError</c>.</summary>
    internal const ulong CommandBufferError = 5;

    /// <summary><c>MTLGPUFamilyApple7</c> (M1, A14).</summary>
    internal const nint FamilyApple7 = 1007;

    /// <summary><c>MTLGPUFamilyApple8</c> (M2, A15).</summary>
    internal const nint FamilyApple8 = 1008;

    /// <summary><c>MTLGPUFamilyApple9</c> (M3, M4, A17).</summary>
    internal const nint FamilyApple9 = 1009;

    /// <summary><c>MTLGPUFamilyApple10</c> (M5).</summary>
    internal const nint FamilyApple10 = 1010;

    /// <summary><c>MTLGPUFamilyMac2</c>.</summary>
    internal const nint FamilyMac2 = 2002;

    /// <summary><c>MTLArgumentBuffersTier2</c>: buffers may hold GPU addresses of other buffers (the kernels' pointer arguments).</summary>
    internal const ulong ArgumentBuffersTier2 = 1;

    /// <summary><c>MTLGPUFamilyMetal3</c>.</summary>
    internal const nint FamilyMetal3 = 5001;

    /// <summary><c>MTLGPUFamilyMetal4</c>.</summary>
    internal const nint FamilyMetal4 = 5002;
}
