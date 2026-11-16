using Sharplight.Storage;

namespace Sharplight.Samples.Hub;

/// <summary>Prints the progress of a storage change on one line of the error output, synchronously (so it never trails the result).</summary>
internal sealed class StorageProgressPrinter : IProgress<StorageMigrationProgress>
{
    private readonly Lock _lock = new();

    public void Report(StorageMigrationProgress value)
    {
        lock (_lock)
        {
            Console.Error.Write($"\r{value.FilesDone}/{value.FilesTotal} files, {ByteSize.Format(value.BytesDone)} of {ByteSize.Format(value.BytesTotal)}   ");
        }
    }
}
