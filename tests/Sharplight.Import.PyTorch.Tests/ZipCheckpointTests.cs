using Xunit;

namespace Sharplight.Import.PyTorch.Tests;

public sealed unsafe class ZipCheckpointTests : IDisposable
{
    private readonly string _dir = CheckpointFiles.NewDirectory();

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static byte[] StateDict() => new PickleBuilder().Proto(2).EmptyOrderedDict()
        .Tensor("w", "FloatStorage", "0", 6, 0, [2, 3], [3, 1])
        .Tensor("tail", "FloatStorage", "0", 6, 4, [2], [1])
        .Tensor("scalar", "FloatStorage", "1", 1, 0, [], [])
        .Stop().ToArray();

    [Fact]
    public void ReadsTensorsOfAStoredZip()
    {
        string path = CheckpointFiles.WriteZip(_dir, StateDict(),
            ("0", CheckpointFiles.Floats(1, 2, 3, 4, 5, 6)), ("1", CheckpointFiles.Floats(42)));
        using var file = TorchCheckpointFile.Open(path);

        Assert.Equal(["w", "tail", "scalar"], file.Tensors.Keys);
        var w = file.Tensors["w"];
        Assert.Equal(TorchDType.Float32, w.DType);
        Assert.Equal([2L, 3L], w.Shape);
        Assert.Equal(24, w.ByteLength);
        Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, new ReadOnlySpan<float>(file.BasePointer + w.FileOffset, 6).ToArray());
        var tail = file.Tensors["tail"];
        Assert.Equal(new float[] { 5, 6 }, new ReadOnlySpan<float>(file.BasePointer + tail.FileOffset, 2).ToArray());
        var scalar = file.Tensors["scalar"];
        Assert.Empty(scalar.Shape);
        Assert.Equal(42f, *(float*)(file.BasePointer + scalar.FileOffset));
    }

    [Fact]
    public void RejectsNonContiguousTensors()
    {
        var pickle = new PickleBuilder().Proto(2).EmptyOrderedDict()
            .Tensor("t", "FloatStorage", "0", 6, 0, [3, 2], [1, 3]).Stop().ToArray();
        string path = CheckpointFiles.WriteZip(_dir, pickle, ("0", CheckpointFiles.Floats(1, 2, 3, 4, 5, 6)));
        var error = Assert.Throws<NotSupportedException>(() => TorchCheckpointFile.Open(path));
        Assert.Contains("not contiguous", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RejectsTensorsPastTheirStorage()
    {
        var pickle = new PickleBuilder().Proto(2).EmptyOrderedDict()
            .Tensor("t", "FloatStorage", "0", 6, 3, [2, 2], [2, 1]).Stop().ToArray();
        string path = CheckpointFiles.WriteZip(_dir, pickle, ("0", CheckpointFiles.Floats(1, 2, 3, 4, 5, 6)));
        Assert.Throws<InvalidDataException>(() => TorchCheckpointFile.Open(path));
    }

    [Fact]
    public void RejectsAStorageLargerThanItsEntry()
    {
        var pickle = new PickleBuilder().Proto(2).EmptyOrderedDict()
            .Tensor("t", "FloatStorage", "0", 1000, 0, [1000], [1]).Stop().ToArray();
        string path = CheckpointFiles.WriteZip(_dir, pickle, ("0", CheckpointFiles.Floats(1, 2)));
        Assert.Throws<InvalidDataException>(() => TorchCheckpointFile.Open(path));
    }

    [Fact]
    public void UnwrapsStateDictOfTrainingCheckpoints()
    {
        var pickle = new PickleBuilder().Proto(2).Raw((byte)'}').Text("epoch").Int(3).SetItem().Text("state_dict").EmptyOrderedDict()
            .Tensor("layer.weight", "FloatStorage", "0", 2, 0, [2], [1]).SetItem().Stop().ToArray();
        string path = CheckpointFiles.WriteZip(_dir, pickle, ("0", CheckpointFiles.Floats(7, 8)));
        using var file = TorchCheckpointFile.Open(path);
        Assert.Equal(["layer.weight"], file.Tensors.Keys);
    }

    [Fact]
    public void RejectsAFileThatIsNotACheckpoint()
    {
        string path = CheckpointFiles.WriteBytes(_dir, "hello world, this is not a checkpoint"u8.ToArray());
        Assert.Throws<InvalidDataException>(() => TorchCheckpointFile.Open(path));
    }

    [Fact]
    public void RejectsATruncatedZip()
    {
        string path = CheckpointFiles.WriteZip(_dir, StateDict(), ("0", CheckpointFiles.Floats(1, 2, 3, 4, 5, 6)), ("1", CheckpointFiles.Floats(42)));
        byte[] bytes = File.ReadAllBytes(path);
        string cut = CheckpointFiles.WriteBytes(_dir, bytes[..(bytes.Length / 2)]);
        Assert.Throws<InvalidDataException>(() => TorchCheckpointFile.Open(cut));
    }
}
