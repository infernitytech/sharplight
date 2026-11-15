namespace Sharplight.Llm.Structured;

/// <summary>
/// Collects named rules (<see cref="GrammarExpr"/>) and compiles them into the flat element form of a <see cref="Grammar"/>:
/// every rule is a list of alternatives, every alternative a run of byte-set and rule-reference elements closed by an end
/// element. Choices nested inside sequences and repetitions become generated rules (named <c>#n</c>, which no front end
/// can produce).
/// </summary>
internal sealed class GrammarBuilder
{
    private readonly Dictionary<string, GrammarExpr> _rules = new(StringComparer.Ordinal);
    private readonly List<string> _order = [];
    private int _generated;

    public bool Contains(string name) => _rules.ContainsKey(name);

    /// <summary>Defines (or replaces) rule <paramref name="name"/>.</summary>
    public void Add(string name, GrammarExpr expr)
    {
        if (!_rules.ContainsKey(name))
        {
            _order.Add(name);
        }

        _rules[name] = expr;
    }

    /// <summary>A fresh rule name for a front end's helper rules.</summary>
    public string NewName(string hint) => $"{hint}#{_generated++}";

    /// <summary>
    /// Any byte string up to and including the first occurrence of <paramref name="terminator"/> (for example a free-form
    /// reasoning block closed by <c>&lt;/think&gt;</c>): the string-matching automaton of the terminator, one rule per state.
    /// </summary>
    public GrammarExpr Until(string terminator)
    {
        byte[] t = System.Text.Encoding.UTF8.GetBytes(terminator);
        if (t.Length == 0)
        {
            return GrammarExpr.Empty;
        }

        // KMP failure function: fail[k] = length of the longest proper border of t[..k].
        var fail = new int[t.Length + 1];
        for (int k = 1, j = 0; k < t.Length; k++)
        {
            while (j > 0 && t[k] != t[j])
            {
                j = fail[j];
            }

            if (t[k] == t[j])
            {
                j++;
            }

            fail[k + 1] = j;
        }

        string prefix = NewName("until");
        for (int state = 0; state < t.Length; state++)
        {
            var targets = new ByteSet[t.Length + 1];
            for (int b = 0; b < 256; b++)
            {
                int next = state;
                while (next > 0 && t[next] != b)
                {
                    next = fail[next];
                }

                if (t[next] == b)
                {
                    next++;
                }

                targets[next] = targets[next].Union(ByteSet.Of((byte)b));
            }

            var alternatives = new List<GrammarExpr>();
            for (int next = 0; next <= t.Length; next++)
            {
                if (!targets[next].IsEmpty)
                {
                    var step = GrammarExpr.Bytes(targets[next]);
                    alternatives.Add(next == t.Length ? step : GrammarExpr.Seq(step, GrammarExpr.Ref($"{prefix}-{next}")));
                }
            }

            Add($"{prefix}-{state}", GrammarExpr.Choice(alternatives));
        }

        return GrammarExpr.Ref($"{prefix}-0");
    }

    /// <summary>Compiles the rules reachable from <paramref name="root"/>.</summary>
    /// <exception cref="ArgumentException">An undefined rule is referenced, or a rule is left-recursive.</exception>
    public Grammar Build(string root)
    {
        if (!_rules.ContainsKey(root))
        {
            throw new ArgumentException($"The grammar has no rule '{root}'.");
        }

        var compiler = new Compiler(_rules);
        compiler.RuleIndex(root);
        return compiler.Finish();
    }

    /// <summary>One compilation: assigns rule indices on first reference and appends each rule's alternatives as a unit.</summary>
    private sealed class Compiler(Dictionary<string, GrammarExpr> rules)
    {
        private readonly Dictionary<string, int> _index = new(StringComparer.Ordinal);
        private readonly List<Grammar.Element> _elements = [];
        private readonly List<int[]> _alternatives = [];
        private readonly List<string> _names = [];

        public int RuleIndex(string name)
        {
            if (_index.TryGetValue(name, out int index))
            {
                return index;
            }

            if (!rules.TryGetValue(name, out var expr))
            {
                throw new ArgumentException($"The grammar references undefined rule '{name}'.");
            }

            index = Reserve(name);
            Define(index, expr);
            return index;
        }

        private int Reserve(string name)
        {
            int index = _alternatives.Count;
            _index[name] = index;
            _alternatives.Add([]);
            _names.Add(name);
            return index;
        }

        private void Define(int index, GrammarExpr expr)
        {
            var alternatives = expr.Kind == GrammarExpr.ExprKind.Choice ? expr.Items : [expr];
            var bodies = new List<List<Grammar.Element>>(alternatives.Count);
            foreach (var alternative in alternatives)
            {
                var body = new List<Grammar.Element>();
                Emit(alternative, body);
                bodies.Add(body);
            }

            // Appended only now: emitting may have compiled other rules, and each alternative must be contiguous.
            var starts = new int[bodies.Count];
            for (int i = 0; i < bodies.Count; i++)
            {
                starts[i] = _elements.Count;
                _elements.AddRange(bodies[i]);
                _elements.Add(new Grammar.Element(Grammar.ElementKind.End, default, -1));
            }

            _alternatives[index] = starts;
        }

        private int Anonymous(GrammarExpr expr)
        {
            int index = Reserve($"#{_alternatives.Count}");
            Define(index, expr);
            return index;
        }

        private void Emit(GrammarExpr expr, List<Grammar.Element> body)
        {
            switch (expr.Kind)
            {
                case GrammarExpr.ExprKind.Bytes:
                    body.Add(new Grammar.Element(Grammar.ElementKind.Bytes, expr.Set, -1));
                    break;
                case GrammarExpr.ExprKind.Sequence:
                    foreach (var item in expr.Items)
                    {
                        Emit(item, body);
                    }

                    break;
                case GrammarExpr.ExprKind.Reference:
                    body.Add(Reference(RuleIndex(expr.Name)));
                    break;
                case GrammarExpr.ExprKind.Choice:
                    body.Add(Reference(Anonymous(expr)));
                    break;
                default:
                    EmitRepeat(expr, body);
                    break;
            }
        }

        private void EmitRepeat(GrammarExpr expr, List<Grammar.Element> body)
        {
            var item = expr.Items[0];
            // Complex bodies are compiled once and referenced, instead of being copied per repetition.
            var unit = item.Kind is GrammarExpr.ExprKind.Bytes or GrammarExpr.ExprKind.Reference ? item : GrammarExpr.Ref(NameOf(Anonymous(item)));
            for (int i = 0; i < expr.Min; i++)
            {
                Emit(unit, body);
            }

            if (expr.Max < 0)
            {
                // tail ::= unit tail | ε
                int tail = Reserve($"#{_alternatives.Count}");
                Define(tail, GrammarExpr.Choice(GrammarExpr.Seq(unit, GrammarExpr.Ref(NameOf(tail))), GrammarExpr.Empty));
                body.Add(Reference(tail));
                return;
            }

            // opt_k ::= unit opt_(k-1) | ε  with opt_1 ::= unit | ε
            int optional = -1;
            for (int k = 1; k <= expr.Max - expr.Min; k++)
            {
                var more = optional < 0 ? unit : GrammarExpr.Seq(unit, GrammarExpr.Ref(NameOf(optional)));
                optional = Anonymous(GrammarExpr.Choice(more, GrammarExpr.Empty));
            }

            if (optional >= 0)
            {
                body.Add(Reference(optional));
            }
        }

        /// <summary>The name of a compiled rule; generated rules resolve through the index, never through the rule table.</summary>
        private string NameOf(int index) => _names[index];

        private static Grammar.Element Reference(int rule) => new(Grammar.ElementKind.Rule, default, rule);

        public Grammar Finish()
        {
            var grammar = new Grammar([.. _elements], [.. _alternatives], [.. _names]);
            grammar.CheckNoLeftRecursion();
            return grammar;
        }
    }
}
