namespace Sharplight.Metal.Interop;

/// <summary>Registered Objective-C selectors of the messages the provider sends (resolved on first use, macOS only).</summary>
internal static class Selectors
{
    /// <summary><c>alloc</c>.</summary>
    internal static readonly nint Alloc = ObjCRuntime.RegisterSelector("alloc");

    /// <summary><c>init</c>.</summary>
    internal static readonly nint Init = ObjCRuntime.RegisterSelector("init");

    /// <summary><c>release</c>.</summary>
    internal static readonly nint Release = ObjCRuntime.RegisterSelector("release");

    /// <summary><c>retain</c>.</summary>
    internal static readonly nint Retain = ObjCRuntime.RegisterSelector("retain");

    /// <summary><c>stringWithUTF8String:</c>.</summary>
    internal static readonly nint StringWithUtf8 = ObjCRuntime.RegisterSelector("stringWithUTF8String:");

    /// <summary><c>UTF8String</c>.</summary>
    internal static readonly nint Utf8String = ObjCRuntime.RegisterSelector("UTF8String");

    /// <summary><c>localizedDescription</c>.</summary>
    internal static readonly nint LocalizedDescription = ObjCRuntime.RegisterSelector("localizedDescription");

    /// <summary><c>count</c>.</summary>
    internal static readonly nint Count = ObjCRuntime.RegisterSelector("count");

    /// <summary><c>objectAtIndex:</c>.</summary>
    internal static readonly nint ObjectAtIndex = ObjCRuntime.RegisterSelector("objectAtIndex:");

    /// <summary><c>name</c>.</summary>
    internal static readonly nint Name = ObjCRuntime.RegisterSelector("name");

    /// <summary><c>registryID</c>.</summary>
    internal static readonly nint RegistryId = ObjCRuntime.RegisterSelector("registryID");

    /// <summary><c>hasUnifiedMemory</c>.</summary>
    internal static readonly nint HasUnifiedMemory = ObjCRuntime.RegisterSelector("hasUnifiedMemory");

    /// <summary><c>isLowPower</c>.</summary>
    internal static readonly nint IsLowPower = ObjCRuntime.RegisterSelector("isLowPower");

    /// <summary><c>isRemovable</c>.</summary>
    internal static readonly nint IsRemovable = ObjCRuntime.RegisterSelector("isRemovable");

    /// <summary><c>recommendedMaxWorkingSetSize</c>.</summary>
    internal static readonly nint RecommendedMaxWorkingSetSize = ObjCRuntime.RegisterSelector("recommendedMaxWorkingSetSize");

    /// <summary><c>currentAllocatedSize</c>.</summary>
    internal static readonly nint CurrentAllocatedSize = ObjCRuntime.RegisterSelector("currentAllocatedSize");

    /// <summary><c>maxThreadgroupMemoryLength</c>.</summary>
    internal static readonly nint MaxThreadgroupMemoryLength = ObjCRuntime.RegisterSelector("maxThreadgroupMemoryLength");

    /// <summary><c>maxThreadsPerThreadgroup</c>.</summary>
    internal static readonly nint MaxThreadsPerThreadgroup = ObjCRuntime.RegisterSelector("maxThreadsPerThreadgroup");

    /// <summary><c>maxBufferLength</c>.</summary>
    internal static readonly nint MaxBufferLength = ObjCRuntime.RegisterSelector("maxBufferLength");

    /// <summary><c>argumentBuffersSupport</c>.</summary>
    internal static readonly nint ArgumentBuffersSupport = ObjCRuntime.RegisterSelector("argumentBuffersSupport");

    /// <summary><c>supportsFamily:</c>.</summary>
    internal static readonly nint SupportsFamily = ObjCRuntime.RegisterSelector("supportsFamily:");

    /// <summary><c>newCommandQueue</c>.</summary>
    internal static readonly nint NewCommandQueue = ObjCRuntime.RegisterSelector("newCommandQueue");

    /// <summary><c>newBufferWithLength:options:</c>.</summary>
    internal static readonly nint NewBufferWithLength = ObjCRuntime.RegisterSelector("newBufferWithLength:options:");

    /// <summary><c>newBufferWithBytesNoCopy:length:options:deallocator:</c>.</summary>
    internal static readonly nint NewBufferWithBytesNoCopy = ObjCRuntime.RegisterSelector("newBufferWithBytesNoCopy:length:options:deallocator:");

    /// <summary><c>newLibraryWithSource:options:error:</c>.</summary>
    internal static readonly nint NewLibraryWithSource = ObjCRuntime.RegisterSelector("newLibraryWithSource:options:error:");

    /// <summary><c>newFunctionWithName:</c>.</summary>
    internal static readonly nint NewFunctionWithName = ObjCRuntime.RegisterSelector("newFunctionWithName:");

    /// <summary><c>newComputePipelineStateWithFunction:error:</c>.</summary>
    internal static readonly nint NewComputePipelineStateWithFunction = ObjCRuntime.RegisterSelector("newComputePipelineStateWithFunction:error:");

    /// <summary><c>maxTotalThreadsPerThreadgroup</c>.</summary>
    internal static readonly nint MaxTotalThreadsPerThreadgroup = ObjCRuntime.RegisterSelector("maxTotalThreadsPerThreadgroup");

    /// <summary><c>threadExecutionWidth</c>.</summary>
    internal static readonly nint ThreadExecutionWidth = ObjCRuntime.RegisterSelector("threadExecutionWidth");

    /// <summary><c>setFastMathEnabled:</c>.</summary>
    internal static readonly nint SetFastMathEnabled = ObjCRuntime.RegisterSelector("setFastMathEnabled:");

    /// <summary><c>setLanguageVersion:</c>.</summary>
    internal static readonly nint SetLanguageVersion = ObjCRuntime.RegisterSelector("setLanguageVersion:");

    /// <summary><c>contents</c>.</summary>
    internal static readonly nint Contents = ObjCRuntime.RegisterSelector("contents");

    /// <summary><c>length</c>.</summary>
    internal static readonly nint Length = ObjCRuntime.RegisterSelector("length");

    /// <summary><c>gpuAddress</c>.</summary>
    internal static readonly nint GpuAddress = ObjCRuntime.RegisterSelector("gpuAddress");

    /// <summary><c>commandBufferWithUnretainedReferences</c>.</summary>
    internal static readonly nint CommandBufferWithUnretainedReferences = ObjCRuntime.RegisterSelector("commandBufferWithUnretainedReferences");

    /// <summary><c>computeCommandEncoderWithDispatchType:</c>.</summary>
    internal static readonly nint ComputeCommandEncoderWithDispatchType = ObjCRuntime.RegisterSelector("computeCommandEncoderWithDispatchType:");

    /// <summary><c>setComputePipelineState:</c>.</summary>
    internal static readonly nint SetComputePipelineState = ObjCRuntime.RegisterSelector("setComputePipelineState:");

    /// <summary><c>setBytes:length:atIndex:</c>.</summary>
    internal static readonly nint SetBytes = ObjCRuntime.RegisterSelector("setBytes:length:atIndex:");

    /// <summary><c>useResources:count:usage:</c>.</summary>
    internal static readonly nint UseResources = ObjCRuntime.RegisterSelector("useResources:count:usage:");

    /// <summary><c>memoryBarrierWithScope:</c>.</summary>
    internal static readonly nint MemoryBarrierWithScope = ObjCRuntime.RegisterSelector("memoryBarrierWithScope:");

    /// <summary><c>dispatchThreadgroups:threadsPerThreadgroup:</c>.</summary>
    internal static readonly nint DispatchThreadgroups = ObjCRuntime.RegisterSelector("dispatchThreadgroups:threadsPerThreadgroup:");

    /// <summary><c>endEncoding</c>.</summary>
    internal static readonly nint EndEncoding = ObjCRuntime.RegisterSelector("endEncoding");

    /// <summary><c>commit</c>.</summary>
    internal static readonly nint Commit = ObjCRuntime.RegisterSelector("commit");

    /// <summary><c>waitUntilCompleted</c>.</summary>
    internal static readonly nint WaitUntilCompleted = ObjCRuntime.RegisterSelector("waitUntilCompleted");

    /// <summary><c>status</c>.</summary>
    internal static readonly nint Status = ObjCRuntime.RegisterSelector("status");

    /// <summary><c>error</c>.</summary>
    internal static readonly nint Error = ObjCRuntime.RegisterSelector("error");

    /// <summary><c>GPUStartTime</c>.</summary>
    internal static readonly nint GpuStartTime = ObjCRuntime.RegisterSelector("GPUStartTime");

    /// <summary><c>GPUEndTime</c>.</summary>
    internal static readonly nint GpuEndTime = ObjCRuntime.RegisterSelector("GPUEndTime");

    /// <summary><c>setUrl:</c>.</summary>
    internal static readonly nint SetUrl = ObjCRuntime.RegisterSelector("setUrl:");

    /// <summary><c>newBinaryArchiveWithDescriptor:error:</c>.</summary>
    internal static readonly nint NewBinaryArchiveWithDescriptor = ObjCRuntime.RegisterSelector("newBinaryArchiveWithDescriptor:error:");

    /// <summary><c>setComputeFunction:</c>.</summary>
    internal static readonly nint SetComputeFunction = ObjCRuntime.RegisterSelector("setComputeFunction:");

    /// <summary><c>setBinaryArchives:</c>.</summary>
    internal static readonly nint SetBinaryArchives = ObjCRuntime.RegisterSelector("setBinaryArchives:");

    /// <summary><c>addComputePipelineFunctionsWithDescriptor:error:</c>.</summary>
    internal static readonly nint AddComputePipelineFunctionsWithDescriptor = ObjCRuntime.RegisterSelector("addComputePipelineFunctionsWithDescriptor:error:");

    /// <summary><c>serializeToURL:error:</c>.</summary>
    internal static readonly nint SerializeToUrl = ObjCRuntime.RegisterSelector("serializeToURL:error:");

    /// <summary><c>newComputePipelineStateWithDescriptor:options:reflection:error:</c>.</summary>
    internal static readonly nint NewComputePipelineStateWithDescriptor = ObjCRuntime.RegisterSelector("newComputePipelineStateWithDescriptor:options:reflection:error:");

    /// <summary><c>fileURLWithPath:</c>.</summary>
    internal static readonly nint FileUrlWithPath = ObjCRuntime.RegisterSelector("fileURLWithPath:");

    /// <summary><c>arrayWithObject:</c>.</summary>
    internal static readonly nint ArrayWithObject = ObjCRuntime.RegisterSelector("arrayWithObject:");
}
