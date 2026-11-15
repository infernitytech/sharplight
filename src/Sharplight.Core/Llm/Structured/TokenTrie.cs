using System.Runtime.CompilerServices;
using System.Text;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Structured;

/// <summary>
/// The byte trie of a tokenizer's vocabulary, built once per tokenizer and shared (<see cref="For"/>). Constrained decoding
/// walks it together with the grammar state: a subtree is skipped as soon as its byte prefix is rejected, so computing a
/// token mask touches only the viable part of the vocabulary. Nodes are stored in compressed sparse rows.
/// </summary>
/// <remarks>
/// Contains every token with a non-empty byte form except special (control / unknown) and end-of-generation tokens: those
/// are only allowed through <see cref="SpecialTokens"/> when a constraint names them, and end-of-generation is the engine's
/// business.
/// </remarks>
internal sealed class TokenTrie
{
    private static readonly ConditionalWeakTable<ITokenizer, TokenTrie> Cache = [];

    private TokenTrie(ITokenizer tokenizer)
    {
        VocabSize = tokenizer.VocabSize;
        var children = new Dictionary<long, int>();
        var nodeTokens = new List<(int Node, int Token)>();
        var special = new Dictionary<string, int>(StringComparer.Ordinal);
        int nodes = 1;
        for (int id = 0; id < VocabSize; id++)
        {
            byte[] bytes = tokenizer.TokenBytes(id);
            if (bytes.Length == 0 || tokenizer.IsEndOfGeneration(id))
            {
                continue;
            }

            if (tokenizer.IsSpecial(id))
            {
                special.TryAdd(Encoding.UTF8.GetString(bytes), id);
                continue;
            }

            int node = 0;
            foreach (byte b in bytes)
            {
                long key = ((long)node << 8) | b;
                if (!children.TryGetValue(key, out int child))
                {
                    child = nodes++;
                    children.Add(key, child);
                }

                node = child;
            }

            nodeTokens.Add((node, id));
        }

        SpecialTokens = special;
        NodeCount = nodes;

        // Children sorted by (parent, byte) give a first CSR in insertion numbering ...
        var edges = children.Select(kv => (Parent: (int)(kv.Key >> 8), Byte: (byte)kv.Key, Child: kv.Value)).ToArray();
        Array.Sort(edges, static (x, y) => x.Parent != y.Parent ? x.Parent.CompareTo(y.Parent) : x.Byte.CompareTo(y.Byte));
        var firstChild = new int[nodes + 1];
        foreach (var e in edges)
        {
            firstChild[e.Parent + 1]++;
        }

        for (int n = 0; n < nodes; n++)
        {
            firstChild[n + 1] += firstChild[n];
        }

        // ... which is renumbered in depth-first preorder, so every subtree is a contiguous node range and (with tokens
        // sorted by node) a contiguous token range.
        var renumber = new int[nodes];
        var preorder = new int[nodes];
        var pending = new Stack<int>();
        pending.Push(0);
        int next = 0;
        while (pending.TryPop(out int old))
        {
            renumber[old] = next;
            preorder[next++] = old;
            for (int i = firstChild[old + 1] - 1; i >= firstChild[old]; i--)
            {
                pending.Push(edges[i].Child);
            }
        }

        ChildStart = new int[nodes + 1];
        ChildByte = new byte[edges.Length];
        ChildNode = new int[edges.Length];
        SubtreeEnd = new int[nodes];
        int edge = 0;
        for (int n = 0; n < nodes; n++)
        {
            int old = preorder[n];
            ChildStart[n] = edge;
            for (int i = firstChild[old]; i < firstChild[old + 1]; i++)
            {
                ChildByte[edge] = edges[i].Byte;
                ChildNode[edge++] = renumber[edges[i].Child];
            }
        }

        ChildStart[nodes] = edge;
        for (int n = nodes - 1; n >= 0; n--)
        {
            // In preorder the subtree of n ends where the subtree of its last child ends.
            int last = ChildStart[n + 1] - 1;
            SubtreeEnd[n] = last >= ChildStart[n] ? SubtreeEnd[ChildNode[last]] : n + 1;
        }

        var byNode = nodeTokens.Select(x => (Node: renumber[x.Node], x.Token)).ToArray();
        Array.Sort(byNode, static (x, y) => x.Node != y.Node ? x.Node.CompareTo(y.Node) : x.Token.CompareTo(y.Token));
        TokenStart = new int[nodes + 1];
        Tokens = new int[byNode.Length];
        for (int i = 0; i < byNode.Length; i++)
        {
            TokenStart[byNode[i].Node + 1]++;
            Tokens[i] = byNode[i].Token;
        }

        for (int n = 0; n < nodes; n++)
        {
            TokenStart[n + 1] += TokenStart[n];
        }
    }

    /// <summary>The (cached) trie of <paramref name="tokenizer"/>; built on first use, which takes tens of milliseconds for large vocabularies.</summary>
    public static TokenTrie For(ITokenizer tokenizer) => Cache.GetValue(tokenizer, static t => new TokenTrie(t));

    public int VocabSize { get; }
    public int NodeCount { get; }
    /// <summary>Children of node n: <c>ChildStart[n]..ChildStart[n+1]</c> in <see cref="ChildByte"/> / <see cref="ChildNode"/>, sorted by byte.</summary>
    public int[] ChildStart { get; }
    public byte[] ChildByte { get; }
    public int[] ChildNode { get; }
    /// <summary>Node ids are in depth-first preorder: the subtree of node n is the node range <c>n..SubtreeEnd[n]-1</c>.</summary>
    public int[] SubtreeEnd { get; }
    /// <summary>Tokens whose bytes end at node n: <c>TokenStart[n]..TokenStart[n+1]</c> in <see cref="Tokens"/>.</summary>
    public int[] TokenStart { get; }
    public int[] Tokens { get; }
    /// <summary>Special tokens that render as text (for example <c>&lt;tool_call&gt;</c> in some vocabularies), by text.</summary>
    public IReadOnlyDictionary<string, int> SpecialTokens { get; }
}
