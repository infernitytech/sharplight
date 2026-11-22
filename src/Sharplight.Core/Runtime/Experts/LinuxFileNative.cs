using System.Runtime.InteropServices;

namespace Sharplight.Runtime.Experts;

/// <summary>The few libc calls the Linux expert readers make (no native library of Sharplight's own).</summary>
internal static partial class LinuxFileNative
{
    private const string Libc = "libc";

    /// <summary><c>O_RDONLY | O_CLOEXEC</c>.</summary>
    public const int OpenReadOnly = 0x80000;

    /// <summary><c>O_DIRECT</c>: 040000 on x86-64, 0200000 on arm64 (and the other asm-generic architectures).</summary>
    public static int OpenDirect => RuntimeInformation.ProcessArchitecture is Architecture.X64 or Architecture.X86 ? 0x4000 : 0x10000;

    [LibraryImport(Libc, EntryPoint = "open", SetLastError = true, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Open(string path, int flags);

    [LibraryImport(Libc, EntryPoint = "close", SetLastError = true)]
    public static partial int Close(int fd);

    [LibraryImport(Libc, EntryPoint = "syscall", SetLastError = true)]
    public static partial long Syscall(long number, long a1, long a2, long a3, long a4, long a5, long a6);

    [LibraryImport(Libc, EntryPoint = "mmap", SetLastError = true)]
    public static partial nint Mmap(nint address, nuint length, int protection, int flags, int fd, long offset);

    [LibraryImport(Libc, EntryPoint = "munmap", SetLastError = true)]
    public static partial int Munmap(nint address, nuint length);
}
