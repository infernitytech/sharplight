using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>Graph builder wrapper owning weights (random or explicit).</summary>
internal sealed class TG
{
    public readonly Dictionary<string, WeightInfo> Weights = [];
    public readonly GraphBuilder B;
    public static readonly Dim T = Dim.Sym("T");
    private int _seed = 1000;
    public TG(string name = "t")
    {
        B = new GraphBuilder(name, n => Weights.GetValueOrDefault(n));
    }

    public Value W(string name, DataType type, Shape shape, byte[] data)
    {
        Weights[name] = new WeightInfo { Name = name, Type = type, Shape = shape, Source = new MemSource(data) };
        return B.Weight(name);
    }

    public Value WF(string name, float[] data, params long[] dims)
        => W(name, DataType.F32, new Shape(dims.Select(d => (Dim)d)), MemoryMarshal.AsBytes(data.AsSpan()).ToArray());

    /// <summary>Random weight in any float or quantized format.</summary>
    public Value RandW(string name, DataType type, float scale, params long[] dims)
    {
        long n = dims.Aggregate(1L, (a, b) => a * b);
        var bytes = Rnd.WeightBytes(type, n, _seed++, scale);
        return W(name, type, new Shape(dims.Select(d => (Dim)d)), bytes);
    }

    public Value Const(string name, float[] data, params long[] dims)
        => B.Constant(name, DataType.F32, new Shape(dims.Select(d => (Dim)d)), MemoryMarshal.AsBytes(data.AsSpan()).ToArray());

    public Value In(string name, DataType type, params Dim[] dims) => B.Input(name, InputRole.Other, type, new Shape(dims));
    public Value InF(string name, params Dim[] dims) => In(name, DataType.F32, dims);
    public Value State(string name, DataType type, StateKind kind, params Dim[] dims)
        => B.State(new StateDefinition { Name = name, Type = type, Shape = new Shape(dims), Kind = kind });
}
