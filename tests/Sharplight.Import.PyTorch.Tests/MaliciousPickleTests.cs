using System.Text;
using Xunit;

namespace Sharplight.Import.PyTorch.Tests;

/// <summary>Pickles that would run code under <c>pickle.load</c> must be refused before anything is evaluated.</summary>
public sealed class MaliciousPickleTests
{
    private static UnsafePickleException Refuse(byte[] pickle) =>
        Assert.Throws<UnsafePickleException>(() => new RestrictedUnpickler(pickle, 0, TorchCheckpointOptions.Default).Load());

    private static byte[] Call(string module, string name, string argument) =>
        new PickleBuilder().Proto(2).Global(module, name).Text(argument).Tuple(1).Reduce().Stop().ToArray();

    [Theory]
    [InlineData("os", "system")]
    [InlineData("posix", "system")]
    [InlineData("nt", "system")]
    [InlineData("subprocess", "Popen")]
    [InlineData("subprocess", "check_output")]
    [InlineData("builtins", "eval")]
    [InlineData("builtins", "exec")]
    [InlineData("builtins", "__import__")]
    [InlineData("builtins", "getattr")]
    [InlineData("__builtin__", "eval")]
    [InlineData("os", "popen")]
    [InlineData("pickle", "loads")]
    [InlineData("importlib", "import_module")]
    [InlineData("torch", "load")]
    [InlineData("torch.storage", "_load_from_bytes")]
    [InlineData("torch.jit", "load")]
    [InlineData("torch.hub", "load")]
    [InlineData("torch._utils", "_rebuild_qtensor")]
    [InlineData("numpy.core.multiarray", "_reconstruct")]
    [InlineData("numpy", "load")]
    [InlineData("shutil", "rmtree")]
    [InlineData("webbrowser", "open")]
    [InlineData("socket", "create_connection")]
    [InlineData("collections", "OrderedDict.__init__")]
    [InlineData("torch._utils", "_rebuild_tensor_v2.__globals__")]
    public void RefusesGlobal(string module, string name)
    {
        var error = Refuse(Call(module, name, "echo pwned"));
        Assert.Contains($"{module}.{name}", error.Message, StringComparison.Ordinal);
        Assert.Contains("not allowed", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RefusesProtocol0TextPayload()
    {
        // The classic: cos\nsystem\n(S'echo pwned'\ntR.
        Refuse("cos\nsystem\n(S'echo pwned'\ntR."u8.ToArray());
    }

    [Fact]
    public void RefusesStackGlobal()
    {
        var pickle = new PickleBuilder().Proto(4).Raw(0x8c, 2).Raw("os"u8.ToArray()).Raw(0x8c, 6).Raw("system"u8.ToArray()).Raw(0x93)
            .Raw(0x8c, 4).Raw("calc"u8.ToArray()).Tuple(1).Reduce().Stop().ToArray();
        var error = Refuse(pickle);
        Assert.Contains("os.system", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RefusesInstAndObj()
    {
        Refuse("(S'echo'\nios\nsystem\n."u8.ToArray());
        Refuse("(cos\nsystem\nS'echo'\no."u8.ToArray());
    }

    [Fact]
    public void RefusesExtensionOpcodesAndNewObjEx()
    {
        Refuse([0x80, 2, 0x82, 1, (byte)'.']);
        Refuse([0x80, 4, (byte)'N', (byte)')', (byte)'}', 0x92, (byte)'.']);
    }

    [Fact]
    public void RefusesNewObjOnAllowedNonInertClass()
    {
        var pickle = new PickleBuilder().Proto(2).Global("torch", "FloatStorage").Tuple(0).Raw(0x81).Stop().ToArray();
        Refuse(pickle);
    }

    [Fact]
    public void RefusesBuildOnTensorsAndCallablesAndCallingStorageTypes()
    {
        Refuse(new PickleBuilder().Proto(2).Global("torch", "FloatStorage").Raw((byte)'}').Raw((byte)'b').Stop().ToArray());
        Refuse(new PickleBuilder().Proto(2).Global("torch", "FloatStorage").Tuple(0).Reduce().Stop().ToArray());
        Refuse(new PickleBuilder().Proto(2).Global("torch._utils", "_rebuild_parameter").Raw((byte)'}').Raw((byte)'b').Stop().ToArray());
    }

    [Fact]
    public void RefusesCodecsEncodeWithOtherEncodings()
    {
        var pickle = new PickleBuilder().Proto(2).Global("_codecs", "encode").Text("a").Text("rot13").Tuple(2).Reduce().Stop().ToArray();
        Refuse(pickle);
    }

    [Fact]
    public void RefusesPersistentIdsThatAreNotStorages()
    {
        var pickle = new PickleBuilder().Proto(2).Text("payload").Raw((byte)'Q').Stop().ToArray();
        Refuse(pickle);
    }

    [Fact]
    public void InertModulesAreOptInAndNeverExecuteAnything()
    {
        var pickle = new PickleBuilder().Proto(2).Global("pytorch_lightning.callbacks", "ModelCheckpoint").Tuple(0).Reduce().Raw((byte)'}').Raw((byte)'b').Stop().ToArray();
        Refuse(pickle);
        var options = new TorchCheckpointOptions { InertModulePrefixes = ["pytorch_lightning."] };
        Assert.NotNull(new RestrictedUnpickler(pickle, 0, options).Load());

        // An inert prefix does not open the dangerous modules.
        var evil = Call("os", "system", "x");
        Assert.Throws<UnsafePickleException>(() => new RestrictedUnpickler(evil, 0, options).Load());
    }

    [Fact]
    public void InertObjectsDiscardTupleState()
    {
        // numpy.dtype('f8') as pickled in Lightning checkpoints: BUILD with a tuple state on the reduced dtype.
        var pickle = new PickleBuilder().Proto(2).Global("numpy", "dtype").Text("f8").Tuple(1).Reduce()
            .Int(3).Text("<").Tuple(2).Raw((byte)'b').Stop().ToArray();
        Refuse(pickle);
        var options = new TorchCheckpointOptions { InertModulePrefixes = ["numpy"] };
        Assert.NotNull(new RestrictedUnpickler(pickle, 0, options).Load());

        // A dictionary still takes only dictionary state.
        Refuse(new PickleBuilder().Proto(2).Raw((byte)'}').Int(3).Tuple(1).Raw((byte)'b').Stop().ToArray());
    }

    [Fact]
    public void MalformedStreamsAreRefusedNotCrashed()
    {
        Refuse([0x80, 2]);
        Refuse([0x80, 2, (byte)'0']);
        Refuse([0x80, 9, (byte)'N', (byte)'.']);
        Refuse([0x80, 2, (byte)'h', 7, (byte)'.']);
        Refuse([0x80, 2, (byte)'X', 0xff, 0xff, 0xff, 0x7f]);
        Refuse([0x80, 2, 0x8d, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f]);
        Refuse([0x80, 2, (byte)'t', (byte)'.']);
        Refuse(Encoding.ASCII.GetBytes("cos\nsystem"));
    }

    [Fact]
    public void ObjectLimitStopsMemoryBombs()
    {
        var bomb = new byte[1_000_000];
        bomb[0] = 0x80;
        bomb[1] = 2;
        Array.Fill(bomb, (byte)']', 2, bomb.Length - 2);
        var options = new TorchCheckpointOptions { MaxObjects = 1000 };
        Assert.Throws<UnsafePickleException>(() => new RestrictedUnpickler(bomb, 0, options).Load());
    }

    [Fact]
    public void MaliciousCheckpointFileIsRefusedWhenOpened()
    {
        string dir = CheckpointFiles.NewDirectory();
        try
        {
            string path = CheckpointFiles.WriteZip(dir, Call("os", "system", "touch /tmp/pwned"));
            var error = Assert.Throws<UnsafePickleException>(() => TorchCheckpointFile.Open(path));
            Assert.Contains("os.system", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
