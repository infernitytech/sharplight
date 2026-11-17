using Sharplight.Metal.Interop;
using Sharplight.Storage;

namespace Sharplight.Metal;

/// <summary>
/// An <c>MTLBinaryArchive</c>: the GPU machine code of compute pipelines for one GPU family and OS build, the device stage of the Metal
/// provider. Archives are created from and serialized to files, so stored bytes pass through a temporary file that lives as long as the
/// archive (Metal may read it lazily), in the cache folder of the active storage mode (<see cref="StorageLayout"/>, <c>metal</c>). Not thread-safe for adding pipelines; pipeline creation may use it from any thread.
/// </summary>
internal sealed unsafe class MetalBinaryArchive : IDisposable
{
    private readonly string? _file;

    private MetalBinaryArchive(nint handle, string? file)
    {
        Handle = handle;
        _file = file;
    }

    /// <summary>The retained <c>MTLBinaryArchive</c>.</summary>
    public nint Handle { get; private set; }

    /// <summary>A new, empty archive.</summary>
    /// <exception cref="InvalidOperationException">Metal could not create the archive.</exception>
    public static MetalBinaryArchive CreateEmpty(nint device) => Create(device, null) ?? throw new InvalidOperationException("Metal could not create a binary archive.");

    /// <summary>The archive serialized in <paramref name="data"/>, or null when Metal rejects it (damaged, other GPU or OS build).</summary>
    public static MetalBinaryArchive? Load(nint device, ReadOnlySpan<byte> data)
    {
        string file = TemporaryFile();
        File.WriteAllBytes(file, data.ToArray());
        var archive = Create(device, file);
        if (archive is null)
        {
            File.Delete(file);
        }

        return archive;
    }

    private static MetalBinaryArchive? Create(nint device, string? file)
    {
        using var pool = AutoreleasePool.Push();
        nint descriptor = Foundation.New("MTLBinaryArchiveDescriptor");
        try
        {
            if (file is not null)
            {
                ObjCRuntime.SendVoid(descriptor, Selectors.SetUrl, Foundation.FileUrl(file));
            }

            nint error = 0;
            nint handle = ObjCRuntime.Send(device, Selectors.NewBinaryArchiveWithDescriptor, descriptor, &error);
            return handle == 0 ? null : new MetalBinaryArchive(handle, file);
        }
        finally
        {
            Foundation.Release(descriptor);
        }
    }

    /// <summary>Compiles the pipeline of <paramref name="function"/> into the archive.</summary>
    /// <exception cref="InvalidOperationException">Metal could not add the pipeline.</exception>
    public void Add(nint function)
    {
        using var pool = AutoreleasePool.Push();
        nint descriptor = Foundation.New("MTLComputePipelineDescriptor");
        try
        {
            ObjCRuntime.SendVoid(descriptor, Selectors.SetComputeFunction, function);
            nint error = 0;
            if (ObjCRuntime.SendBool(Handle, Selectors.AddComputePipelineFunctionsWithDescriptor, descriptor, &error) == 0)
            {
                throw new InvalidOperationException("Metal could not add a pipeline to the binary archive: " + Foundation.ErrorText(error, "unknown error"));
            }
        }
        finally
        {
            Foundation.Release(descriptor);
        }
    }

    /// <summary>The archive's serialized bytes.</summary>
    /// <exception cref="InvalidOperationException">Metal could not serialize the archive.</exception>
    public byte[] Serialize()
    {
        string file = TemporaryFile();
        try
        {
            using var pool = AutoreleasePool.Push();
            nint error = 0;
            if (ObjCRuntime.SendBool(Handle, Selectors.SerializeToUrl, Foundation.FileUrl(file), &error) == 0)
            {
                throw new InvalidOperationException("Metal could not serialize the binary archive: " + Foundation.ErrorText(error, "unknown error"));
            }

            return File.ReadAllBytes(file);
        }
        finally
        {
            File.Delete(file);
        }
    }

    private static string TemporaryFile()
    {
        string directory = Path.Combine(StorageLayout.Resolve(StorageEnvironment.Process).CacheDirectory, "metal");
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, $"archive-{Guid.NewGuid():N}.metallib");
    }

    public void Dispose()
    {
        Foundation.Release(Handle);
        Handle = 0;
        if (_file is not null)
        {
            File.Delete(_file);
        }
    }
}
