using System.Text;

namespace Sharplight.Templating;

/// <summary>Lexical scope used while rendering.</summary>
internal sealed class Frame
{
    public readonly Dictionary<string, object?> Vars;
    public readonly Frame? Parent;
    public readonly JinjaEnvironment Env;
    public int LoopDepth;

    public Frame(JinjaEnvironment env, Frame? parent, Dictionary<string, object?>? vars = null)
    {
        Env = env;
        Parent = parent;
        Vars = vars ?? new Dictionary<string, object?>();
        LoopDepth = parent?.LoopDepth ?? 0;
    }

    public bool Strict => Env.StrictUndefined;

    public object? Lookup(string name)
    {
        for (var f = this; f != null; f = f.Parent)
            if (f.Vars.TryGetValue(name, out var v)) return v;
        if (Env.Globals.TryGetValue(name, out var g)) return g;
        return new Undefined(name, Strict);
    }

    public void Set(string name, object? value) => Vars[name] = value;
}

internal enum Flow { Normal, Break, Continue }

internal abstract class Node
{
    public int Line, Col;
}

internal abstract class Expr : Node
{
    public object? Eval(Frame f)
    {
        try
        {
            return Core(f);
        }
        catch (TemplateException e) when (e.Annotate(Line, Col))
        {
            throw;
        }
    }

    protected abstract object? Core(Frame f);
    public virtual string Describe() => "expression";
}

internal abstract class Stmt : Node
{
    public abstract Flow Exec(Frame f, StringBuilder o);

    public static Flow ExecAll(List<Stmt> body, Frame f, StringBuilder o)
    {
        foreach (var s in body)
        {
            var flow = s.Exec(f, o);
            if (flow != Flow.Normal) return flow;
        }
        return Flow.Normal;
    }
}

// ============================================================================ expressions

internal sealed class ConstExpr(object? value) : Expr
{
    protected override object? Core(Frame f) => value;
    public override string Describe() => Ops.Repr(value);
}

internal sealed class NameExpr(string name) : Expr
{
    public string Name => name;
    protected override object? Core(Frame f) => f.Lookup(name);
    public override string Describe() => name;
}

internal sealed class ListExpr(List<Expr> items, bool tuple = false) : Expr
{
    protected override object? Core(Frame f)
    {
        List<object?> l = tuple ? new JinjaTuple() : new List<object?>(items.Count);
        foreach (var e in items) l.Add(e.Eval(f));
        return l;
    }
}

internal sealed class DictExpr(List<(Expr Key, Expr Value)> items) : Expr
{
    protected override object? Core(Frame f)
    {
        var d = new JinjaDict();
        foreach (var (k, v) in items)
        {
            var key = k.Eval(f);
            if (key is not (string or long or bool or double))
                throw new TemplateRuntimeException($"unhashable or unsupported dict key of type '{Ops.TypeName(key)}'");
            d[key is double dk && dk == Math.Floor(dk) ? (long)dk : key!] = v.Eval(f);
        }
        return d;
    }
}

internal sealed class GetAttrExpr(Expr obj, string name) : Expr
{
    public Expr Object => obj;
    public string AttrName => name;

    protected override object? Core(Frame f) => Resolve(f, false);

    public object? Resolve(Frame f, bool forCall)
    {
        var o = obj.Eval(f);
        var r = Ops.GetAttr(o, name, f.Strict, forCall);
        if (r is Undefined { Name: null } u) return new Undefined(Describe(), u.Strict);
        return r;
    }

    public override string Describe() => obj.Describe() + "." + name;
}

internal sealed class GetItemExpr(Expr obj, Expr key) : Expr
{
    protected override object? Core(Frame f)
    {
        var o = obj.Eval(f);
        var k = key.Eval(f);
        var r = Ops.GetItem(o, k, f.Strict);
        if (r is Undefined { Name: null } u) return new Undefined(Describe(), u.Strict);
        return r;
    }

    public override string Describe() => obj.Describe() + "[...]";
}

internal sealed class SliceExpr(Expr? start, Expr? stop, Expr? step) : Expr
{
    protected override object? Core(Frame f) => new SliceValue
    {
        Start = start?.Eval(f),
        Stop = stop?.Eval(f),
        Step = step?.Eval(f),
    };
}

internal sealed class CallExpr(Expr callee, List<Expr> args, List<(string Name, Expr Value)> named) : Expr
{
    protected override object? Core(Frame f) => Run(f, null);

    public object? EvalWithCaller(Frame f, JinjaCallable caller)
    {
        try
        {
            return Run(f, caller);
        }
        catch (TemplateException e) when (e.Annotate(Line, Col))
        {
            throw;
        }
    }

    private object? Run(Frame f, JinjaCallable? caller)
    {
        object? fn = callee is GetAttrExpr ga ? ga.Resolve(f, true) : callee.Eval(f);
        var pos = new List<object?>(args.Count);
        foreach (var a in args) pos.Add(a.Eval(f));
        Dictionary<string, object?>? nm = null;
        if (named.Count > 0)
        {
            nm = new Dictionary<string, object?>();
            foreach (var (n, v) in named) nm[n] = v.Eval(f);
        }
        if (caller != null) (nm ??= new Dictionary<string, object?>())["caller"] = caller;
        if (fn is JinjaCallable c) return c.Invoke(new CallArgs(pos, nm));
        if (fn is Undefined u) throw new TemplateRuntimeException($"'{u.Name ?? callee.Describe()}' is undefined");
        throw new TemplateRuntimeException($"'{callee.Describe()}' is not callable (type {Ops.TypeName(fn)})");
    }

    public override string Describe() => callee.Describe() + "(...)";
}

internal sealed class FilterSpec(string name, List<Expr> args, List<(string Name, Expr Value)> named, Token tok)
{
    public string Name => name;
    public int Line => tok.Line;
    public int Col => tok.Col;

    public object? Apply(Frame f, object? value)
    {
        if (!f.Env.Filters.TryGetValue(name, out var fn)) throw new TemplateRuntimeException($"No filter named '{name}'");
        var pos = new List<object?>(args.Count);
        foreach (var a in args) pos.Add(a.Eval(f));
        Dictionary<string, object?>? nm = null;
        if (named.Count > 0)
        {
            nm = new Dictionary<string, object?>();
            foreach (var (n, v) in named) nm[n] = v.Eval(f);
        }
        try
        {
            return fn(f.Env, value, new CallArgs(pos, nm));
        }
        catch (TemplateException e) when (e.Annotate(Line, Col))
        {
            throw;
        }
    }
}

internal sealed class FilterExpr(Expr value, FilterSpec spec) : Expr
{
    protected override object? Core(Frame f) => spec.Apply(f, value.Eval(f));
    public override string Describe() => value.Describe() + "|" + spec.Name;
}

internal sealed class TestExpr(Expr value, string name, List<Expr> args, List<(string Name, Expr Value)> named, bool negate) : Expr
{
    protected override object? Core(Frame f)
    {
        if (!f.Env.Tests.TryGetValue(name, out var fn)) throw new TemplateRuntimeException($"No test named '{name}'");
        var v = value.Eval(f);
        var pos = new List<object?>(args.Count);
        foreach (var a in args) pos.Add(a.Eval(f));
        Dictionary<string, object?>? nm = null;
        if (named.Count > 0)
        {
            nm = new Dictionary<string, object?>();
            foreach (var (n, nv) in named) nm[n] = nv.Eval(f);
        }
        bool r = fn(f.Env, v, new CallArgs(pos, nm));
        return negate ? !r : r;
    }
}

internal sealed class BinaryExpr(string op, Expr left, Expr right) : Expr
{
    protected override object? Core(Frame f)
    {
        var a = left.Eval(f);
        var b = right.Eval(f);
        return op switch
        {
            "+" => Ops.Add(a, b),
            "-" => Ops.Sub(a, b),
            "*" => Ops.Mul(a, b),
            "/" => Ops.Div(a, b),
            "//" => Ops.FloorDiv(a, b),
            "%" => Ops.Mod(a, b),
            "**" => Ops.Pow(a, b),
            _ => throw new InvalidOperationException(op),
        };
    }
}

internal sealed class ConcatExpr(List<Expr> parts) : Expr
{
    protected override object? Core(Frame f)
    {
        if (parts.Count == 2) return string.Concat(Ops.Str(parts[0].Eval(f)), Ops.Str(parts[1].Eval(f)));
        var sb = new StringBuilder();
        foreach (var p in parts) sb.Append(Ops.Str(p.Eval(f)));
        return sb.ToString();
    }
}

internal sealed class CompareExpr(Expr first, List<(string Op, Expr Rhs)> rest) : Expr
{
    protected override object? Core(Frame f)
    {
        var left = first.Eval(f);
        bool result = true;
        foreach (var (op, rhsExpr) in rest)
        {
            var right = rhsExpr.Eval(f);
            result = op switch
            {
                "==" => Ops.Eq(left, right),
                "!=" => !Ops.Eq(left, right),
                "<" => Ops.Compare(left, right, "<") < 0,
                "<=" => Ops.Compare(left, right, "<=") <= 0,
                ">" => Ops.Compare(left, right, ">") > 0,
                ">=" => Ops.Compare(left, right, ">=") >= 0,
                "in" => Ops.Contains(right, left),
                "not in" => !Ops.Contains(right, left),
                _ => throw new InvalidOperationException(op),
            };
            if (!result) return false;
            left = right;
        }
        return result;
    }
}

internal sealed class AndExpr(Expr l, Expr r) : Expr
{
    protected override object? Core(Frame f)
    {
        var a = l.Eval(f);
        return Ops.Truthy(a) ? r.Eval(f) : a;
    }
}

internal sealed class OrExpr(Expr l, Expr r) : Expr
{
    protected override object? Core(Frame f)
    {
        var a = l.Eval(f);
        return Ops.Truthy(a) ? a : r.Eval(f);
    }
}

internal sealed class NotExpr(Expr e) : Expr
{
    protected override object? Core(Frame f) => !Ops.Truthy(e.Eval(f));
}

internal sealed class NegExpr(Expr e, bool neg) : Expr
{
    protected override object? Core(Frame f) => neg ? Ops.Neg(e.Eval(f)) : Ops.Pos(e.Eval(f));
}

internal sealed class CondExpr(Expr cond, Expr then, Expr? otherwise) : Expr
{
    protected override object? Core(Frame f)
    {
        if (Ops.Truthy(cond.Eval(f))) return then.Eval(f);
        return otherwise != null ? otherwise.Eval(f) : new Undefined("conditional expression", f.Strict);
    }
}

// ============================================================================ assignment targets

internal abstract class Target : Node
{
    public abstract void Assign(Frame f, object? value);
}

internal sealed class NameTarget(string name) : Target
{
    public override void Assign(Frame f, object? value) => f.Set(name, value);
}

internal sealed class AttrTarget(string objName, string attr) : Target
{
    public override void Assign(Frame f, object? value)
    {
        var o = f.Lookup(objName);
        if (o is JinjaNamespace ns) ns.Attributes[attr] = value;
        else throw new TemplateRuntimeException($"cannot assign attribute '{attr}' on '{objName}' (type {Ops.TypeName(o)}); only namespace objects support attribute assignment");
    }
}

internal sealed class TupleTarget(List<Target> items) : Target
{
    public override void Assign(Frame f, object? value)
    {
        if (value is Undefined u && u.Strict) throw u.Fail();
        if (!Ops.IsIterable(value)) throw new TemplateRuntimeException($"cannot unpack non-iterable {Ops.TypeName(value)} object");
        var l = Ops.Iterate(value);
        if (l.Count != items.Count)
            throw new TemplateRuntimeException(l.Count < items.Count
                ? $"not enough values to unpack (expected {items.Count}, got {l.Count})"
                : $"too many values to unpack (expected {items.Count})");
        for (int i = 0; i < items.Count; i++) items[i].Assign(f, l[i]);
    }
}

// ============================================================================ statements

internal sealed class DataStmt(string text) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        o.Append(text);
        return Flow.Normal;
    }
}

internal sealed class PrintStmt(Expr e) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        var v = e.Eval(f);
        try
        {
            o.Append(Ops.Str(v));
        }
        catch (TemplateException ex) when (ex.Annotate(Line, Col))
        {
            throw;
        }
        return Flow.Normal;
    }
}

internal sealed class IfStmt(List<(Expr Cond, List<Stmt> Body)> branches, List<Stmt>? otherwise) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        foreach (var (cond, body) in branches)
        {
            bool t;
            try
            {
                t = Ops.Truthy(cond.Eval(f));
            }
            catch (TemplateException ex) when (ex.Annotate(Line, Col))
            {
                throw;
            }
            if (t) return ExecAll(body, f, o);
        }
        return otherwise != null ? ExecAll(otherwise, f, o) : Flow.Normal;
    }
}

internal sealed class ForStmt(Target target, Expr iter, Expr? filter, List<Stmt> body, List<Stmt>? otherwise) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        List<object?> seq;
        try
        {
            seq = Ops.Iterate(iter.Eval(f));
        }
        catch (TemplateException ex) when (ex.Annotate(Line, Col))
        {
            throw;
        }
        var inner = new Frame(f.Env, f);
        var items = seq;
        if (filter != null)
        {
            items = new List<object?>();
            foreach (var x in seq)
            {
                target.Assign(inner, x);
                if (Ops.Truthy(filter.Eval(inner))) items.Add(x);
            }
        }
        if (items.Count == 0)
        {
            if (otherwise != null) ExecAll(otherwise, f, o);
            return Flow.Normal;
        }
        var loop = new LoopContext { Items = items };
        inner.Vars["loop"] = loop;
        inner.LoopDepth = f.LoopDepth + 1;
        for (int i = 0; i < items.Count; i++)
        {
            loop.Index0 = i;
            try
            {
                target.Assign(inner, items[i]);
            }
            catch (TemplateException ex) when (ex.Annotate(Line, Col))
            {
                throw;
            }
            var flow = ExecAll(body, inner, o);
            if (flow == Flow.Break) break;
        }
        return Flow.Normal;
    }
}

internal sealed class SetStmt(Target target, Expr value) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        var v = value.Eval(f);
        try
        {
            target.Assign(f, v);
        }
        catch (TemplateException ex) when (ex.Annotate(Line, Col))
        {
            throw;
        }
        return Flow.Normal;
    }
}

internal sealed class SetBlockStmt(Target target, List<FilterSpec> filters, List<Stmt> body) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        var sb = new StringBuilder();
        ExecAll(body, f, sb);
        object? v = sb.ToString();
        foreach (var fl in filters) v = fl.Apply(f, v);
        target.Assign(f, v);
        return Flow.Normal;
    }
}

internal sealed class FilterBlockStmt(List<FilterSpec> filters, List<Stmt> body) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        var sb = new StringBuilder();
        ExecAll(body, f, sb);
        object? v = sb.ToString();
        foreach (var fl in filters) v = fl.Apply(f, v);
        o.Append(Ops.Str(v));
        return Flow.Normal;
    }
}

internal sealed class DoStmt(Expr e) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        e.Eval(f);
        return Flow.Normal;
    }
}

internal sealed class BlockStmt(List<Stmt> body) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o) => ExecAll(body, f, o);
}

internal sealed class LoopControlStmt(Flow flow) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o) => flow;
}

internal sealed class MacroDef(string name, List<(string Name, Expr? Default)> parameters, List<Stmt> body)
{
    public string Name => name;
    public List<(string Name, Expr? Default)> Parameters => parameters;
    public List<Stmt> Body => body;
}

internal sealed class MacroStmt(MacroDef def) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        f.Set(def.Name, new Macro(def, f));
        return Flow.Normal;
    }
}

internal sealed class CallBlockStmt(MacroDef caller, CallExpr call) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        o.Append(Ops.Str(call.EvalWithCaller(f, new Macro(caller, f))));
        return Flow.Normal;
    }
}

internal sealed class Macro(MacroDef def, Frame closure) : JinjaCallable
{
    public override string Name => def.Name;

    public override object? Invoke(CallArgs args)
    {
        var frame = new Frame(closure.Env, closure);
        var ps = def.Parameters;
        for (int i = 0; i < ps.Count; i++)
        {
            var (pn, pd) = ps[i];
            if (i < args.Positional.Count) frame.Vars[pn] = args.Positional[i];
            else if (args.Named != null && args.Named.TryGetValue(pn, out var nv)) frame.Vars[pn] = nv;
            else if (pd != null) frame.Vars[pn] = pd.Eval(frame);
            else frame.Vars[pn] = new Undefined(pn, closure.Strict);
        }
        var varargs = new JinjaTuple();
        for (int i = ps.Count; i < args.Positional.Count; i++) varargs.Add(args.Positional[i]);
        var kwargs = new JinjaDict();
        if (args.Named != null)
        {
            foreach (var (k, v) in args.Named)
            {
                if (k == "caller") frame.Vars["caller"] = v;
                else if (!ps.Exists(p => p.Name == k)) kwargs[k] = v;
            }
        }
        frame.Vars["varargs"] = varargs;
        frame.Vars["kwargs"] = kwargs;
        var sb = new StringBuilder();
        Stmt.ExecAll(def.Body, frame, sb);
        return sb.ToString();
    }
}
