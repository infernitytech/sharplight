using Xunit;

namespace Sharplight.Import.PyTorch.Tests;

/// <summary>Random corruption of valid checkpoints must end in a clean, typed refusal: no other exception, no hang.</summary>
public sealed class FuzzTests : IDisposable
{
    private readonly string _dir = CheckpointFiles.NewDirectory();

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static byte[] Valid() => new PickleBuilder().Proto(2).EmptyOrderedDict()
        .Tensor("w", "FloatStorage", "0", 6, 0, [2, 3], [3, 1])
        .Tensor("b", "HalfStorage", "1", 4, 0, [4], [1])
        .Stop().ToArray();

    [Fact]
    public void MutatedPicklesOnlyFailWithUnsafePickleException()
    {
        var random = new Random(12345);
        var valid = Valid();
        int refused = 0;
        for (int i = 0; i < 5_000; i++)
        {
            var mutated = (byte[])valid.Clone();
            int edits = 1 + random.Next(4);
            for (int e = 0; e < edits; e++)
            {
                switch (random.Next(3))
                {
                    case 0:
                        mutated[random.Next(mutated.Length)] = (byte)random.Next(256);
                        break;
                    case 1:
                        mutated[random.Next(mutated.Length)] ^= (byte)(1 << random.Next(8));
                        break;
                    default:
                        mutated = mutated[..random.Next(1, mutated.Length)];
                        break;
                }

                if (mutated.Length == 0)
                {
                    break;
                }
            }

            try
            {
                new RestrictedUnpickler(mutated, 0, new TorchCheckpointOptions { MaxObjects = 10_000 }).Load();
            }
            catch (UnsafePickleException)
            {
                refused++;
            }
        }

        Assert.True(refused > 250);
    }

    [Fact]
    public void RandomBytesNeverCrashTheUnpickler()
    {
        var random = new Random(777);
        for (int i = 0; i < 5_000; i++)
        {
            var data = new byte[random.Next(1, 200)];
            random.NextBytes(data);
            if (random.Next(2) == 0)
            {
                data[0] = 0x80;
            }

            try
            {
                new RestrictedUnpickler(data, 0, new TorchCheckpointOptions { MaxObjects = 10_000 }).Load();
            }
            catch (UnsafePickleException)
            {
            }
        }
    }

    [Fact]
    public void MutatedCheckpointFilesFailCleanly()
    {
        var random = new Random(99);
        string path = CheckpointFiles.WriteZip(_dir, Valid(), ("0", CheckpointFiles.Floats(1, 2, 3, 4, 5, 6)), ("1", new byte[8]));
        var valid = File.ReadAllBytes(path);
        for (int i = 0; i < 400; i++)
        {
            var mutated = (byte[])valid.Clone();
            for (int e = 0; e < 1 + random.Next(3); e++)
            {
                mutated[random.Next(mutated.Length)] = (byte)random.Next(256);
            }

            string file = CheckpointFiles.WriteBytes(_dir, mutated);
            try
            {
                using var opened = TorchCheckpointFile.Open(file);
                foreach (var tensor in opened.Tensors.Values)
                {
                    Assert.InRange(tensor.FileOffset + tensor.ByteLength, 0, opened.Length);
                }
            }
            catch (Exception ex) when (ex is UnsafePickleException or InvalidDataException or NotSupportedException)
            {
            }
            finally
            {
                File.Delete(file);
            }
        }
    }
}
