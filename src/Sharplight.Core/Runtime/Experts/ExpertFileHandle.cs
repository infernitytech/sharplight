using Microsoft.Win32.SafeHandles;

namespace Sharplight.Runtime.Experts;

/// <summary>
/// A file opened for the expert readers: on Linux with <c>O_DIRECT</c> where the file system takes it (reads then bypass the page cache, which
/// keeps expert reads from evicting other memory, and must be aligned to <see cref="DirectAlignment"/>), else buffered.
/// </summary>
internal sealed class ExpertFileHandle : IExpertFile
{
    /// <summary>The alignment of direct reads: the logical block size of every common device and the page size.</summary>
    public const int DirectAlignment = 4096;

    private ExpertFileHandle(string path, SafeFileHandle handle, int descriptor, int alignment)
    {
        Path = path;
        Handle = handle;
        Descriptor = descriptor;
        Alignment = alignment;
        Length = RandomAccess.GetLength(handle);
    }

    /// <inheritdoc/>
    public string Path { get; }

    /// <inheritdoc/>
    public long Length { get; }

    /// <inheritdoc/>
    public int Alignment { get; }

    /// <summary>The handle (positional reads go through <see cref="RandomAccess"/>).</summary>
    public SafeFileHandle Handle { get; }

    /// <summary>The Linux file descriptor, or -1 elsewhere.</summary>
    public int Descriptor { get; }

    /// <summary>Opens <paramref name="path"/>: direct where possible (<paramref name="direct"/> and Linux), else buffered.</summary>
    /// <param name="path">The file.</param>
    /// <param name="direct">Whether to try <c>O_DIRECT</c>.</param>
    public static ExpertFileHandle Open(string path, bool direct)
    {
        if (OperatingSystem.IsLinux())
        {
            int fd = direct ? LinuxFileNative.Open(path, LinuxFileNative.OpenReadOnly | LinuxFileNative.OpenDirect) : -1;
            int alignment = DirectAlignment;
            if (fd < 0)
            {
                // tmpfs, some FUSE and network file systems refuse O_DIRECT: read through the page cache there.
                fd = LinuxFileNative.Open(path, LinuxFileNative.OpenReadOnly);
                alignment = 1;
            }

            if (fd < 0)
            {
                throw new IOException($"Cannot open '{path}' for reading routed experts (errno {System.Runtime.InteropServices.Marshal.GetLastPInvokeError()}).");
            }

            return new ExpertFileHandle(path, new SafeFileHandle(fd, ownsHandle: true), fd, alignment);
        }

        var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete, FileOptions.RandomAccess);
        return new ExpertFileHandle(path, handle, -1, 1);
    }

    /// <inheritdoc/>
    public void Dispose() => Handle.Dispose();
}
