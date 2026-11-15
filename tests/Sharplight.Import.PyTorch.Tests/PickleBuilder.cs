using System.Text;

namespace Sharplight.Import.PyTorch.Tests;

/// <summary>Assembles pickle byte streams the way <c>torch.save</c> writes them (protocol 2), and hostile variants.</summary>
internal sealed class PickleBuilder
{
    private readonly MemoryStream _bytes = new();

    public PickleBuilder Raw(params byte[] bytes)
    {
        _bytes.Write(bytes);
        return this;
    }

    public PickleBuilder Proto(byte version) => Raw(0x80, version);

    public PickleBuilder Stop() => Raw((byte)'.');

    public PickleBuilder Mark() => Raw((byte)'(');

    public PickleBuilder Global(string module, string name) => Raw((byte)'c').Raw(Encoding.ASCII.GetBytes(module + "\n" + name + "\n"));

    public PickleBuilder Text(string value)
    {
        var utf8 = Encoding.UTF8.GetBytes(value);
        Raw((byte)'X').Raw(BitConverter.GetBytes(utf8.Length)).Raw(utf8);
        return this;
    }

    public PickleBuilder Int(long value)
    {
        if (value is >= 0 and < 256)
        {
            return Raw((byte)'K', (byte)value);
        }

        return Raw((byte)'J').Raw(BitConverter.GetBytes(checked((int)value)));
    }

    public PickleBuilder Tuple(int count) => count switch
    {
        0 => Raw((byte)')'),
        1 => Raw(0x85),
        2 => Raw(0x86),
        3 => Raw(0x87),
        _ => Raw((byte)'t'),
    };

    public PickleBuilder Reduce() => Raw((byte)'R');

    public PickleBuilder EmptyOrderedDict() => Global("collections", "OrderedDict").Tuple(0).Reduce();

    public PickleBuilder SetItem() => Raw((byte)'s');

    /// <summary>Appends <c>name: tensor</c> for a contiguous tensor stored under <paramref name="storageKey"/>.</summary>
    public PickleBuilder Tensor(string name, string storageClass, string storageKey, long storageElements, long offset, long[] shape, long[] strides)
    {
        Text(name);
        Global("torch._utils", "_rebuild_tensor_v2");
        Mark();
        Mark().Text("storage").Global("torch", storageClass).Text(storageKey).Text("cpu").Int(storageElements).Tuple(5).Raw((byte)'Q');
        Int(offset);
        foreach (long dim in shape)
        {
            Int(dim);
        }

        Tuple(shape.Length);
        foreach (long stride in strides)
        {
            Int(stride);
        }

        Tuple(strides.Length);
        Raw(0x89);
        EmptyOrderedDict();
        Tuple(6);
        Reduce();
        return SetItem();
    }

    public byte[] ToArray() => _bytes.ToArray();
}
