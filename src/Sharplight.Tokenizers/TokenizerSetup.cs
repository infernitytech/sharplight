namespace Sharplight.Tokenizers;

/// <summary>Constructor steps shared by the tokenizer implementations.</summary>
internal static class TokenizerSetup
{
    /// <summary>The <see cref="TokenType"/> of every token; missing or undefined entries count as <see cref="TokenType.Normal"/>.</summary>
    public static TokenType[] ResolveTypes(TokenizerData data)
    {
        var types = new TokenType[data.Tokens.Length];
        for (int i = 0; i < types.Length; i++)
        {
            bool declared = data.TokenTypes is { } declaredTypes && i < declaredTypes.Length && Enum.IsDefined((TokenType)declaredTypes[i]);
            types[i] = declared ? (TokenType)data.TokenTypes![i] : TokenType.Normal;
        }

        return types;
    }

    /// <summary>The id if it lies inside the vocabulary, otherwise null.</summary>
    public static int? ValidId(int? id, int vocabSize) => id is int v && v >= 0 && v < vocabSize ? v : null;

    /// <summary>
    /// End-of-generation ids, sorted: the ids declared by the data, the EOS id, and well-known end-of-turn token names that are
    /// control or user-defined tokens of the vocabulary (llama.cpp's heuristic).
    /// </summary>
    /// <param name="data">The tokenizer description.</param>
    /// <param name="eosId">The validated EOS id.</param>
    /// <param name="types">Types of all tokens.</param>
    /// <param name="knownNames">Token strings that count as end of generation when present.</param>
    /// <param name="lookup">Vocabulary lookup of a token string.</param>
    public static int[] ResolveEndOfGeneration(TokenizerData data, int? eosId, TokenType[] types, IEnumerable<string> knownNames, Func<string, int?> lookup)
    {
        var eog = new List<int>();
        foreach (int declared in data.EosIds)
        {
            if (ValidId(declared, types.Length) is int id && !eog.Contains(id))
            {
                eog.Add(id);
            }
        }

        if (eosId is int eos && !eog.Contains(eos))
        {
            eog.Add(eos);
        }

        foreach (string name in knownNames)
        {
            if (lookup(name) is int id && types[id] is TokenType.Control or TokenType.UserDefined && !eog.Contains(id))
            {
                eog.Add(id);
            }
        }

        // Harmony (gpt-oss) and Solar Open end messages with <|end|> but turns with <|return|> / <|call|> (<|calls|> / <|flush|>):
        // like llama.cpp, <|end|> does not end the generation there, so the final channel can follow the analysis one.
        bool Has(string name) => lookup(name) is int id && eog.Contains(id);
        bool hasCall = Has("<|call|>") || Has("<|calls|>");
        if (lookup("<|end|>") is int end && Has("<|end|>") && hasCall && (Has("<|return|>") || Has("<|flush|>")))
        {
            eog.Remove(end);
        }

        eog.Sort();
        return eog.ToArray();
    }
}
