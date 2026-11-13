namespace Sharplight.Templating;

internal sealed class Parser
{
    private readonly List<Token> _t;
    private int _i;

    private Parser(List<Token> tokens) => _t = tokens;

    public static List<Stmt> Parse(List<Token> tokens)
    {
        var p = new Parser(tokens);
        var body = p.ParseStatements();
        return body;
    }

    // ------------------------------------------------------------------ token helpers

    private Token Cur => _t[_i];
    private Token PeekAt(int n) => _t[Math.Min(_i + n, _t.Count - 1)];

    private Token Next()
    {
        var t = _t[_i];
        if (t.Kind != TokenKind.Eof)
        {
            _i++;
        }

        return t;
    }

    private bool IsOp(string op) => Cur.Kind == TokenKind.Op && Cur.Text == op;
    private bool IsName(string name) => Cur.Kind == TokenKind.Name && Cur.Text == name;

    private bool SkipOp(string op)
    {
        if (!IsOp(op))
        {
            return false;
        }

        _i++;
        return true;
    }

    private bool SkipName(string name)
    {
        if (!IsName(name))
        {
            return false;
        }

        _i++;
        return true;
    }

    private TemplateSyntaxException Fail(string msg, Token? at = null)
    {
        at ??= Cur;
        return new TemplateSyntaxException(msg, at.Line, at.Col);
    }

    private Token ExpectOp(string op)
    {
        if (!IsOp(op))
        {
            throw Fail($"expected '{op}', got {Cur}");
        }

        return Next();
    }

    private Token ExpectName(string? name = null)
    {
        if (Cur.Kind != TokenKind.Name || (name != null && Cur.Text != name))
        {
            throw Fail($"expected {(name != null ? $"'{name}'" : "a name")}, got {Cur}");
        }

        return Next();
    }

    private void ExpectBlockEnd()
    {
        if (Cur.Kind != TokenKind.BlockEnd)
        {
            throw Fail($"expected '%}}', got {Cur}");
        }

        _i++;
    }

    private static T At<T>(T node, Token t) where T : Node
    {
        node.Line = t.Line;
        node.Col = t.Col;
        return node;
    }

    // ------------------------------------------------------------------ statements

    private List<Stmt> ParseStatements(params string[] endTags)
    {
        var body = new List<Stmt>();
        while (true)
        {
            var t = Cur;
            switch (t.Kind)
            {
                case TokenKind.Data:
                    _i++;
                    body.Add(new DataStmt((string)t.Value!));
                    break;
                case TokenKind.VarBegin:
                {
                    _i++;
                    var e = ParseExpr();
                    if (Cur.Kind != TokenKind.VarEnd)
                        {
                            throw Fail($"expected '}}}}', got {Cur}");
                        }

                        _i++;
                    body.Add(At(new PrintStmt(e), t));
                    break;
                }
                case TokenKind.BlockBegin:
                {
                    var nameTok = PeekAt(1);
                    if (nameTok.Kind != TokenKind.Name)
                        {
                            throw Fail("expected a tag name after '{%'", nameTok);
                        }

                        if (Array.IndexOf(endTags, nameTok.Text) >= 0)
                        {
                            return body;
                        }

                        _i += 2;
                    var s = ParseBlockTag(nameTok);
                    if (s != null)
                        {
                            body.Add(At(s, t));
                        }

                        break;
                }
                case TokenKind.Eof:
                    if (endTags.Length == 0)
                    {
                        return body;
                    }

                    throw Fail($"Unexpected end of template. Expected one of: {string.Join(", ", endTags.Select(e => "'" + e + "'"))}");
                default:
                    throw Fail($"unexpected {t}");
            }
        }
    }

    /// <summary>Consumes <c>{% name %}</c> for an end tag, optionally followed by a repeated name.</summary>
    private void EatEndTag(string name)
    {
        _i++; // BlockBegin
        ExpectName(name);
        if (Cur.Kind == TokenKind.Name)
        {
            _i++;
        }

        ExpectBlockEnd();
    }

    private Stmt? ParseBlockTag(Token tag)
    {
        switch (tag.Text)
        {
            case "if": return ParseIf();
            case "for": return ParseFor();
            case "set": return ParseSet();
            case "macro": return ParseMacro();
            case "call": return ParseCall();
            case "filter": return ParseFilterBlock();
            case "do":
            {
                var e = ParseExpr();
                ExpectBlockEnd();
                return new DoStmt(e);
            }
            case "break":
                ExpectBlockEnd();
                return new LoopControlStmt(Flow.Break);
            case "continue":
                ExpectBlockEnd();
                return new LoopControlStmt(Flow.Continue);
            case "generation":
            {
                ExpectBlockEnd();
                var body = ParseStatements("endgeneration");
                EatEndTag("endgeneration");
                return new BlockStmt(body);
            }
            case "block":
            {
                ExpectName();
                ExpectBlockEnd();
                var body = ParseStatements("endblock");
                EatEndTag("endblock");
                return new BlockStmt(body);
            }
            case "include":
            case "import":
            case "from":
            case "extends":
                throw Fail($"The '{tag.Text}' tag is not supported", tag);
            default:
                throw Fail($"Encountered unknown tag '{tag.Text}'", tag);
        }
    }

    private Stmt ParseIf()
    {
        var branches = new List<(Expr, List<Stmt>)>();
        List<Stmt>? otherwise = null;
        var cond = ParseExpr();
        ExpectBlockEnd();
        branches.Add((cond, ParseStatements("elif", "else", "endif")));
        while (true)
        {
            var name = PeekAt(1).Text;
            if (name == "elif")
            {
                _i += 2;
                var c = ParseExpr();
                ExpectBlockEnd();
                branches.Add((c, ParseStatements("elif", "else", "endif")));
            }
            else if (name == "else")
            {
                _i += 2;
                ExpectBlockEnd();
                otherwise = ParseStatements("endif");
            }
            else
            {
                EatEndTag("endif");
                break;
            }
        }
        return new IfStmt(branches, otherwise);
    }

    private Stmt ParseFor()
    {
        var target = ParseTargetList();
        ExpectName("in");
        var iter = ParseOr();
        Expr? filter = null;
        if (SkipName("if"))
        {
            filter = ParseExpr();
        }

        if (IsName("recursive"))
        {
            throw Fail("recursive loops are not supported");
        }

        ExpectBlockEnd();
        var body = ParseStatements("else", "endfor");
        List<Stmt>? otherwise = null;
        if (PeekAt(1).Text == "else")
        {
            _i += 2;
            ExpectBlockEnd();
            otherwise = ParseStatements("endfor");
        }
        EatEndTag("endfor");
        return new ForStmt(target, iter, filter, body, otherwise);
    }

    private Target ParseTargetList()
    {
        var first = Cur;
        var items = new List<Target>();
        bool comma = false;
        while (true)
        {
            items.Add(ParseSingleTarget());
            if (!IsOp(","))
            {
                break;
            }

            _i++;
            comma = true;
            if (IsName("in") || IsOp("=") || Cur.Kind == TokenKind.BlockEnd || IsOp(")"))
            {
                break;
            }
        }
        if (items.Count == 1 && !comma)
        {
            return items[0];
        }

        return At(new TupleTarget(items), first);
    }

    private Target ParseSingleTarget()
    {
        var t = Cur;
        if (SkipOp("("))
        {
            var inner = ParseTargetList();
            ExpectOp(")");
            return inner is TupleTarget ? inner : At(new TupleTarget(new List<Target> { inner }), t);
        }
        var name = ExpectName();
        if (SkipOp("."))
        {
            var attr = ExpectName();
            return At(new AttrTarget(name.Text, attr.Text), name);
        }
        return At(new NameTarget(name.Text), name);
    }

    private Stmt ParseSet()
    {
        var target = ParseTargetList();
        if (SkipOp("="))
        {
            var value = ParseExprList();
            ExpectBlockEnd();
            return new SetStmt(target, value);
        }
        var filters = new List<FilterSpec>();
        while (IsOp("|"))
        {
            filters.Add(ParseFilterSpec());
        }

        ExpectBlockEnd();
        var body = ParseStatements("endset");
        EatEndTag("endset");
        return new SetBlockStmt(target, filters, body);
    }

    private FilterSpec ParseFilterSpec()
    {
        ExpectOp("|");
        var name = ExpectName();
        string fname = name.Text;
        while (IsOp(".") && PeekAt(1).Kind == TokenKind.Name)
        {
            _i++;
            fname += "." + Next().Text;
        }
        var args = new List<Expr>();
        var named = new List<(string, Expr)>();
        if (IsOp("("))
        {
            ParseCallArgs(args, named);
        }

        return new FilterSpec(fname, args, named, name);
    }

    private List<(string, Expr?)> ParseParams()
    {
        var ps = new List<(string, Expr?)>();
        ExpectOp("(");
        while (!IsOp(")"))
        {
            if (ps.Count > 0)
            {
                ExpectOp(",");
            }

            if (IsOp(")"))
            {
                break;
            }

            var n = ExpectName();
            Expr? def = null;
            if (SkipOp("="))
            {
                def = ParseExpr();
            }

            ps.Add((n.Text, def));
        }
        ExpectOp(")");
        return ps;
    }

    private Stmt ParseMacro()
    {
        var name = ExpectName();
        var ps = ParseParams();
        ExpectBlockEnd();
        var body = ParseStatements("endmacro");
        EatEndTag("endmacro");
        return new MacroStmt(new MacroDef(name.Text, ps!, body));
    }

    private Stmt ParseCall()
    {
        var startTok = Cur;
        var ps = IsOp("(") ? ParseParams() : new List<(string, Expr?)>();
        var callTok = Cur;
        var e = ParseExpr();
        if (e is not CallExpr ce)
        {
            throw Fail("expected a call expression after 'call'", callTok);
        }

        ExpectBlockEnd();
        var body = ParseStatements("endcall");
        EatEndTag("endcall");
        return new CallBlockStmt(new MacroDef("caller", ps!, body), ce);
    }

    private Stmt ParseFilterBlock()
    {
        var filters = new List<FilterSpec>();
        // first filter has no leading '|'
        var name = ExpectName();
        var args = new List<Expr>();
        var named = new List<(string, Expr)>();
        if (IsOp("("))
        {
            ParseCallArgs(args, named);
        }

        filters.Add(new FilterSpec(name.Text, args, named, name));
        while (IsOp("|"))
        {
            filters.Add(ParseFilterSpec());
        }

        ExpectBlockEnd();
        var body = ParseStatements("endfilter");
        EatEndTag("endfilter");
        return new FilterBlockStmt(filters, body);
    }

    // ------------------------------------------------------------------ expressions

    /// <summary>Expression list without parentheses: <c>a, b</c> becomes a tuple.</summary>
    private Expr ParseExprList()
    {
        var first = Cur;
        var e = ParseExpr();
        if (!IsOp(","))
        {
            return e;
        }

        var items = new List<Expr> { e };
        while (SkipOp(","))
        {
            if (Cur.Kind == TokenKind.BlockEnd || Cur.Kind == TokenKind.VarEnd)
            {
                break;
            }

            items.Add(ParseExpr());
        }
        return At(new ListExpr(items, true), first);
    }

    public Expr ParseExpr()
    {
        var t = Cur;
        var e = ParseOr();
        if (!IsName("if"))
        {
            return e;
        }
        // 'a if c else b' (right associative)
        while (SkipName("if"))
        {
            var cond = ParseOr();
            Expr? other = null;
            if (SkipName("else"))
            {
                other = ParseExpr();
            }

            e = At(new CondExpr(cond, e, other), t);
        }
        return e;
    }

    private Expr ParseOr()
    {
        var t = Cur;
        var l = ParseAnd();
        while (IsName("or"))
        {
            _i++;
            l = At(new OrExpr(l, ParseAnd()), t);
        }
        return l;
    }

    private Expr ParseAnd()
    {
        var t = Cur;
        var l = ParseNot();
        while (IsName("and"))
        {
            _i++;
            l = At(new AndExpr(l, ParseNot()), t);
        }
        return l;
    }

    private Expr ParseNot()
    {
        var t = Cur;
        if (IsName("not"))
        {
            _i++;
            return At(new NotExpr(ParseNot()), t);
        }
        return ParseCompare();
    }

    private Expr ParseCompare()
    {
        var t = Cur;
        var first = ParseAdd();
        List<(string, Expr)>? rest = null;
        while (true)
        {
            string? op = null;
            if (Cur.Kind == TokenKind.Op && Cur.Text is "==" or "!=" or "<" or "<=" or ">" or ">=")
            {
                op = Cur.Text;
                _i++;
            }
            else if (IsName("in"))
            {
                op = "in";
                _i++;
            }
            else if (IsName("not") && PeekAt(1).Kind == TokenKind.Name && PeekAt(1).Text == "in")
            {
                op = "not in";
                _i += 2;
            }
            if (op == null)
            {
                break;
            }

            (rest ??= new()).Add((op, ParseAdd()));
        }
        return rest == null ? first : At(new CompareExpr(first, rest), t);
    }

    private Expr ParseAdd()
    {
        var t = Cur;
        var l = ParseConcat();
        while (Cur.Kind == TokenKind.Op && Cur.Text is "+" or "-")
        {
            string op = Next().Text;
            l = At(new BinaryExpr(op, l, ParseConcat()), t);
        }
        return l;
    }

    private Expr ParseConcat()
    {
        var t = Cur;
        var first = ParseMul();
        if (!IsOp("~"))
        {
            return first;
        }

        var parts = new List<Expr> { first };
        while (SkipOp("~"))
        {
            parts.Add(ParseMul());
        }

        return At(new ConcatExpr(parts), t);
    }

    private Expr ParseMul()
    {
        var t = Cur;
        var l = ParsePow();
        while (Cur.Kind == TokenKind.Op && Cur.Text is "*" or "/" or "//" or "%")
        {
            string op = Next().Text;
            l = At(new BinaryExpr(op, l, ParsePow()), t);
        }
        return l;
    }

    private Expr ParsePow()
    {
        var t = Cur;
        var l = ParseUnary(true);
        while (IsOp("**"))
        {
            _i++;
            l = At(new BinaryExpr("**", l, ParseUnary(true)), t);
        }
        return l;
    }

    private Expr ParseUnary(bool withFilter)
    {
        var t = Cur;
        Expr node;
        if (IsOp("-"))
        {
            _i++;
            node = At(new NegExpr(ParseUnary(false), true), t);
        }
        else if (IsOp("+"))
        {
            _i++;
            node = At(new NegExpr(ParseUnary(false), false), t);
        }
        else
        {
            node = ParsePrimary();
        }

        node = ParsePostfix(node);
        if (withFilter)
        {
            node = ParseFilterTail(node);
        }

        return node;
    }

    private Expr ParsePrimary()
    {
        var t = Cur;
        switch (t.Kind)
        {
            case TokenKind.Name:
                _i++;
                switch (t.Text)
                {
                    case "true": case "True": return At(new ConstExpr(true), t);
                    case "false": case "False": return At(new ConstExpr(false), t);
                    case "none": case "None": return At(new ConstExpr(null), t);
                }
                return At(new NameExpr(t.Text), t);
            case TokenKind.String:
            {
                _i++;
                string s = (string)t.Value!;
                while (Cur.Kind == TokenKind.String)
                    {
                        s += (string)Next().Value!;
                    }

                    return At(new ConstExpr(s), t);
            }
            case TokenKind.Int:
            case TokenKind.Float:
                _i++;
                return At(new ConstExpr(t.Value), t);
            case TokenKind.Op when t.Text == "(":
            {
                _i++;
                if (SkipOp(")"))
                    {
                        return At(new ListExpr(new List<Expr>(), true), t);
                    }

                    var first = ParseExpr();
                if (IsOp(","))
                {
                    var items = new List<Expr> { first };
                    while (SkipOp(","))
                    {
                        if (IsOp(")"))
                            {
                                break;
                            }

                            items.Add(ParseExpr());
                    }
                    ExpectOp(")");
                    return At(new ListExpr(items, true), t);
                }
                ExpectOp(")");
                return first;
            }
            case TokenKind.Op when t.Text == "[":
            {
                _i++;
                var items = new List<Expr>();
                while (!IsOp("]"))
                {
                    if (items.Count > 0)
                        {
                            ExpectOp(",");
                        }

                        if (IsOp("]"))
                        {
                            break;
                        }

                        items.Add(ParseExpr());
                }
                ExpectOp("]");
                return At(new ListExpr(items), t);
            }
            case TokenKind.Op when t.Text == "{":
            {
                _i++;
                var items = new List<(Expr, Expr)>();
                while (!IsOp("}"))
                {
                    if (items.Count > 0)
                        {
                            ExpectOp(",");
                        }

                        if (IsOp("}"))
                        {
                            break;
                        }

                        var k = ParseExpr();
                    ExpectOp(":");
                    var v = ParseExpr();
                    items.Add((k, v));
                }
                ExpectOp("}");
                return At(new DictExpr(items), t);
            }
            default:
                throw Fail($"unexpected {t}");
        }
    }

    private Expr ParsePostfix(Expr node)
    {
        while (true)
        {
            var t = Cur;
            if (IsOp("."))
            {
                _i++;
                var n = Cur;
                if (n.Kind == TokenKind.Name)
                {
                    _i++;
                    node = At(new GetAttrExpr(node, n.Text), t);
                }
                else if (n.Kind == TokenKind.Int)
                {
                    _i++;
                    node = At(new GetItemExpr(node, At(new ConstExpr(n.Value), n)), t);
                }
                else
                {
                    throw Fail($"expected attribute name, got {n}");
                }
            }
            else if (IsOp("["))
            {
                _i++;
                node = At(new GetItemExpr(node, ParseSubscript()), t);
                ExpectOp("]");
            }
            else if (IsOp("("))
            {
                var args = new List<Expr>();
                var named = new List<(string, Expr)>();
                ParseCallArgs(args, named);
                node = At(new CallExpr(node, args, named), t);
            }
            else
            {
                return node;
            }
        }
    }

    private Expr ParseSubscript()
    {
        var t = Cur;
        Expr? start = null, stop = null, step = null;
        if (!IsOp(":"))
        {
            start = ParseExpr();
        }

        if (!IsOp(":"))
        {
            return start!;
        }

        _i++; // first ':'
        if (!IsOp(":") && !IsOp("]"))
        {
            stop = ParseExpr();
        }

        if (SkipOp(":") && !IsOp("]"))
        {
            step = ParseExpr();
        }

        return At(new SliceExpr(start, stop, step), t);
    }

    private void ParseCallArgs(List<Expr> args, List<(string, Expr)> named)
    {
        ExpectOp("(");
        while (!IsOp(")"))
        {
            if (args.Count + named.Count > 0)
            {
                ExpectOp(",");
            }

            if (IsOp(")"))
            {
                break;
            }

            if (Cur.Kind == TokenKind.Name && PeekAt(1).Kind == TokenKind.Op && PeekAt(1).Text == "=")
            {
                string n = Next().Text;
                _i++;
                named.Add((n, ParseExpr()));
            }
            else
            {
                if (named.Count > 0)
                {
                    throw Fail("positional argument follows keyword argument");
                }

                args.Add(ParseExpr());
            }
        }
        ExpectOp(")");
    }

    private Expr ParseFilterTail(Expr node)
    {
        while (true)
        {
            var t = Cur;
            if (IsOp("|"))
            {
                node = At(new FilterExpr(node, ParseFilterSpec()), t);
            }
            else if (IsName("is"))
            {
                _i++;
                bool negate = SkipName("not");
                var name = ExpectName();
                string tname = name.Text;
                while (IsOp(".") && PeekAt(1).Kind == TokenKind.Name)
                {
                    _i++;
                    tname += "." + Next().Text;
                }
                var args = new List<Expr>();
                var named = new List<(string, Expr)>();
                if (IsOp("("))
                {
                    ParseCallArgs(args, named);
                }
                else if ((Cur.Kind is TokenKind.Name or TokenKind.String or TokenKind.Int or TokenKind.Float
                          || (Cur.Kind == TokenKind.Op && Cur.Text is "(" or "[" or "{"))
                         && !(Cur.Kind == TokenKind.Name && Cur.Text is "else" or "or" or "and" or "if" or "is" or "in" or "not"))
                {
                    var a = ParsePostfix(ParsePrimary());
                    args.Add(a);
                }
                node = At(new TestExpr(node, tname, args, named, negate), t);
            }
            else
            {
                return node;
            }
        }
    }
}
