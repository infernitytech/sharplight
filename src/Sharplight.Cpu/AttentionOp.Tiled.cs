namespace Sharplight.Cpu;

// Tiled ("flash") attention for the common case: one sequence, f32/f16 K and V, causal or full, optional sliding window and
// sinks. A task owns the query rows of one KV head that fit into the vector lanes (heads of the GQA group x consecutive
// tokens, one row per lane), streams the keys in blocks and keeps a running softmax per lane, so every K/V row is converted
// once per task and every multiply-add works on a full vector of rows. Short queries (decode) split the keys over several
// tasks whose partial results are merged afterwards.
internal sealed unsafe partial class AttentionOp
{
    private const int KeyBlock = 32;
    private const int ScoreKeys = 8;
    private const int ValueColumns = 8;
    private const int MinKeysPerSplit = 4 * KeyBlock;

    /// <summary>Up to this many query tokens use the dot-product row kernel instead of the lane-tiled one.</summary>
    private const int RowPathMaxTokens = 2;

    /// <summary>K/V rows the decode kernel prefetches ahead (the cache rows of one head are strided, which hardware prefetchers follow poorly).</summary>
    private const int PrefetchRows = 8;

    private int _tiledLanes;
    private Action<int>? _tiled256;
    private Action<int>? _tiled512;
    private Action<int>? _rowTask;
    private int _headsPerTask;
    private int _tokensPerTask;
    private int _headBlocks;
    private int _tokenBlocks;
    private int _splits;
    private long _splitKeys;
    private float* _partials;

    /// <summary>Chooses the tiled kernel's vector width, or 0 when this attention needs the general path.</summary>
    private void InitTiled()
    {
        bool symmetricWindow = _window > 0 && _offsetScalar is null && !_causal;
        bool plain = _rows is null && _spans is null && _pages is null && _mask is null && _alibiSlopes is null && _kQuant is null && !symmetricWindow
            && _vQuant is null && _chunk == 0 && _softcap == 0f && _dv % ValueColumns == 0
            && _k.Strides[2] == 1 && _v.Strides[2] == 1;
        _tiledLanes = !plain ? 0 : CpuIsa.Use512 ? FloatLanes512.Count : CpuIsa.Use256 ? FloatLanes256.Count : 0;
        _tiled256 = TiledTask<FloatLanes256>;
        _tiled512 = TiledTask<FloatLanes512>;
        _rowTask = RowTask;
    }

    /// <summary>Visible keys [first, end) of query row <paramref name="t"/>.</summary>
    private void VisibleKeys(long t, out long first, out long end)
    {
        end = _offsetScalar is not null ? _past + t + 1 : _causal ? Math.Min(_s, t + _s - _t + 1) : _s;
        first = _window > 0 ? Math.Max(0, end - _window) : 0;
    }

    private void RunTiled()
    {
        int group = _hq / _hkv;
        bool rowPath = _t <= RowPathMaxTokens;
        int lanes = rowPath ? group : _tiledLanes;
        _headsPerTask = rowPath ? group : Math.Min(group, lanes);
        _tokensPerTask = rowPath ? 1 : Math.Max(1, lanes / _headsPerTask);
        _headBlocks = (group + _headsPerTask - 1) / _headsPerTask;
        _tokenBlocks = (int)((_t + _tokensPerTask - 1) / _tokensPerTask);
        long baseTasks = (long)_hkv * _headBlocks * _tokenBlocks;
        VisibleKeys(_t - 1, out _, out long maxEnd);
        _splits = 1;
        int threads = CpuParallel.Threads;
        if (baseTasks < 2L * threads)
        {
            long wanted = (2L * threads + baseTasks - 1) / baseTasks;
            _splits = (int)Math.Clamp(Math.Min(wanted, maxEnd / MinKeysPerSplit), 1, 64);
        }

        _splitKeys = (maxEnd + _splits - 1) / _splits;
        _splitKeys = (_splitKeys + KeyBlock - 1) / KeyBlock * KeyBlock;
        if (_splits > 1)
        {
            _partials = (float*)ThreadScratch.Get(ThreadScratch.Caller, baseTasks * _splits * lanes * (_dv + 2L) * sizeof(float));
        }

        var body = rowPath ? _rowTask! : lanes == FloatLanes512.Count ? _tiled512! : _tiled256!;
        CpuParallel.For((int)(baseTasks * _splits), body);
        if (_splits > 1)
        {
            MergeSplits(baseTasks, lanes);
        }
    }

    /// <summary>
    /// Decode-sized queries: the few query heads of one KV head and token against a range of keys, in dot-product form
    /// (each K/V row is converted once and used by every head of the group).
    /// </summary>
    private void RowTask(int task)
    {
        int split = task % _splits;
        int rest = task / _splits;
        long t = rest % _tokenBlocks;
        int hk = rest / _tokenBlocks;
        int group = _hq / _hkv;
        int h0 = hk * group;
        int d = _d;
        int dv = _dv;
        VisibleKeys(t, out long f, out long e);
        f = Math.Max(f, split * _splitKeys);
        e = Math.Min(e, (split + 1) * _splitKeys);
        int n = (int)Math.Max(0, e - f);
        long floats = (long)group * d + Math.Max(d, dv) + (long)group * n + (long)group * dv + 2L * group;
        float* qs = (float*)ThreadScratch.Get(ThreadScratch.Worker, floats * sizeof(float));
        float* row = qs + (long)group * d;
        float* sc = row + Math.Max(d, dv);
        float* o = sc + (long)group * n;
        float* state = o + (long)group * dv;
        for (int r = 0; r < group; r++)
        {
            Simd.Scale(_q.F(t * _hq + h0 + r), qs + (long)r * d, _scale, d);
        }

        long ks0 = _k.Strides[0];
        long ks1 = _k.Strides[1];
        long vs0 = _v.Strides[0];
        long vs1 = _v.Strides[1];
        int kRowBytes = d * (_kHalf ? sizeof(ushort) : sizeof(float));
        int vRowBytes = dv * (_vHalf ? sizeof(ushort) : sizeof(float));
        for (int j = 0; j < n; j++)
        {
            if (j + PrefetchRows < n)
            {
                Simd.Prefetch(_k.P + ((f + j + PrefetchRows) * ks0 + hk * ks1) * (_kHalf ? 2 : 4), kRowBytes);
            }

            float* k = KeyRow(row, (f + j) * ks0 + hk * ks1);
            for (int r = 0; r < group; r++)
            {
                sc[(long)r * n + j] = Simd.Dot(qs + (long)r * d, k, d);
            }
        }

        for (int r = 0; r < group; r++)
        {
            bool sink = _sinks is not null && split == 0;
            float m = sink ? ((float*)_sinks!.P)[(h0 + r) * _sinks.Strides[0]] : float.NegativeInfinity;
            float* sr = sc + (long)r * n;
            m = MathF.Max(m, n > 0 ? Simd.Max(sr, n) : float.NegativeInfinity);
            float l = 0f;
            if (m != float.NegativeInfinity)
            {
                l = Simd.ExpShifted(sr, n, m) + (sink ? MathF.Exp(((float*)_sinks!.P)[(h0 + r) * _sinks.Strides[0]] - m) : 0f);
            }

            state[r] = m;
            state[group + r] = l;
        }

        new Span<float>(o, group * dv).Clear();
        for (int j = 0; j < n; j++)
        {
            if (j + PrefetchRows < n)
            {
                Simd.Prefetch(_v.P + ((f + j + PrefetchRows) * vs0 + hk * vs1) * (_vHalf ? 2 : 4), vRowBytes);
            }

            float* v = ValueRow(row, (f + j) * vs0 + hk * vs1);
            for (int r = 0; r < group; r++)
            {
                Simd.Axpy(o + (long)r * dv, v, sc[(long)r * n + j], dv);
            }
        }

        for (int r = 0; r < group; r++)
        {
            float m = state[r];
            float l = state[group + r];
            float* or = o + (long)r * dv;
            if (_splits == 1)
            {
                Simd.Scale(or, (float*)_y.P + (t * _hq + h0 + r) * dv, l > 0f ? 1f / l : 0f, dv);
                continue;
            }

            float* pr = _partials + ((long)task * group + r) * (dv + 2);
            pr[0] = m;
            pr[1] = l;
            Buffer.MemoryCopy(or, pr + 2, dv * sizeof(float), dv * sizeof(float));
        }
    }

    /// <summary>The K row at element offset <paramref name="offset"/> as f32 (converted into <paramref name="buffer"/> when f16).</summary>
    private float* KeyRow(float* buffer, long offset)
    {
        if (!_kHalf)
        {
            return (float*)_k.P + offset;
        }

        Simd.HalfToFloat((ushort*)_k.P + offset, buffer, _d);
        return buffer;
    }

    /// <summary>The V row at element offset <paramref name="offset"/> as f32 (converted into <paramref name="buffer"/> when f16).</summary>
    private float* ValueRow(float* buffer, long offset)
    {
        if (!_vHalf)
        {
            return (float*)_v.P + offset;
        }

        Simd.HalfToFloat((ushort*)_v.P + offset, buffer, _dv);
        return buffer;
    }

    private void TiledTask<TL>(int task) where TL : struct, IFloatLanes<TL>
    {
        int lanes = TL.Count;
        int split = task % _splits;
        int rest = task / _splits;
        int tokenBlock = rest % _tokenBlocks;
        rest /= _tokenBlocks;
        int headBlock = rest % _headBlocks;
        int hk = rest / _headBlocks;
        int group = _hq / _hkv;
        int h0 = hk * group + headBlock * _headsPerTask;
        int heads = Math.Min(_headsPerTask, group - headBlock * _headsPerTask);
        long t0 = (long)tokenBlock * _tokensPerTask;
        int tokens = (int)Math.Min(_tokensPerTask, _t - t0);
        int d = _d;
        int dv = _dv;

        long floats = (long)d * lanes + (long)KeyBlock * d + (long)KeyBlock * dv + (long)KeyBlock * lanes + (long)dv * lanes + 4L * lanes;
        float* qt = (float*)ThreadScratch.Get(ThreadScratch.Worker, floats * sizeof(float));
        float* kb = qt + (long)d * lanes;
        float* vb = kb + (long)KeyBlock * d;
        float* sc = vb + (long)KeyBlock * dv;
        float* ot = sc + (long)KeyBlock * lanes;
        float* state = ot + (long)dv * lanes;
        int* first = (int*)(state + 2L * lanes);
        int* end = first + lanes;

        long splitBegin = split * _splitKeys;
        long splitEnd = splitBegin + _splitKeys;
        long kbeg = long.MaxValue;
        long kend = 0;
        for (int r = 0; r < lanes; r++)
        {
            int ti = r / _headsPerTask;
            int hi = r % _headsPerTask;
            state[r] = float.NegativeInfinity;
            state[lanes + r] = 0f;
            first[r] = 0;
            end[r] = 0;
            if (ti >= tokens || hi >= heads)
            {
                continue;
            }

            VisibleKeys(t0 + ti, out long f, out long e);
            f = Math.Max(f, splitBegin);
            e = Math.Min(e, splitEnd);
            if (e > f)
            {
                first[r] = (int)f;
                end[r] = (int)e;
                kbeg = Math.Min(kbeg, f);
                kend = Math.Max(kend, e);
            }

            if (_sinks is not null && split == 0)
            {
                state[r] = ((float*)_sinks.P)[(h0 + hi) * _sinks.Strides[0]];
                state[lanes + r] = 1f;
            }
        }

        LoadQueries(qt, lanes, t0, tokens, h0, heads);
        new Span<float>(ot, dv * lanes).Clear();
        var m = TL.Load(state);
        var l = TL.Load(state + lanes);
        for (long jb = kbeg; jb < kend; jb += KeyBlock)
        {
            int nk = (int)Math.Min(KeyBlock, kend - jb);
            LoadKeyValueBlock(kb, vb, hk, jb, nk);
            Scores<TL>(qt, kb, sc, d, lanes);

            var blockMax = TL.Broadcast(float.NegativeInfinity);
            for (int j = 0; j < KeyBlock; j++)
            {
                var s = TL.KeepInRange(TL.Load(sc + j * lanes), (int)(jb + j), first, end);
                TL.Store(s, sc + j * lanes);
                blockMax = TL.Max(blockMax, s);
            }

            var mNew = TL.Max(m, blockMax);
            var mSafe = TL.ZeroIfNegativeInfinity(mNew);
            var alpha = TL.Exp(TL.Subtract(m, mSafe));
            var sum = TL.Zero;
            for (int j = 0; j < KeyBlock; j++)
            {
                var p = TL.Exp(TL.Subtract(TL.Load(sc + j * lanes), mSafe));
                TL.Store(p, sc + j * lanes);
                sum = TL.Add(sum, p);
            }

            l = TL.Fma(l, alpha, sum);
            m = mNew;
            AccumulateValues<TL>(sc, vb, ot, alpha, dv, lanes);
        }

        TL.Store(m, state);
        TL.Store(l, state + lanes);
        if (_splits == 1)
        {
            WriteRows(ot, state, lanes, t0, tokens, h0, heads);
        }
        else
        {
            float* part = _partials + (long)task * lanes * (dv + 2);
            for (int r = 0; r < lanes; r++)
            {
                float* pr = part + (long)r * (dv + 2);
                pr[0] = state[r];
                pr[1] = state[lanes + r];
                for (int c = 0; c < dv; c++)
                {
                    pr[2 + c] = ot[(long)c * lanes + r];
                }
            }
        }
    }

    /// <summary>Q rows transposed to [d][lanes] and pre-scaled; unused lanes are zero.</summary>
    private void LoadQueries(float* qt, int lanes, long t0, int tokens, int h0, int heads)
    {
        new Span<float>(qt, _d * lanes).Clear();
        for (int r = 0; r < lanes; r++)
        {
            int ti = r / _headsPerTask;
            int hi = r % _headsPerTask;
            if (ti >= tokens || hi >= heads)
            {
                continue;
            }

            float* q = _q.F((t0 + ti) * _hq + h0 + hi);
            for (int i = 0; i < _d; i++)
            {
                qt[(long)i * lanes + r] = q[i] * _scale;
            }
        }
    }

    /// <summary>Converts <paramref name="nk"/> K and V rows of KV head <paramref name="hk"/> to f32 and zero-fills the block.</summary>
    private void LoadKeyValueBlock(float* kb, float* vb, int hk, long jb, int nk)
    {
        long ks0 = _k.Strides[0];
        long ks1 = _k.Strides[1];
        long vs0 = _v.Strides[0];
        long vs1 = _v.Strides[1];
        for (int j = 0; j < nk; j++)
        {
            long ko = (jb + j) * ks0 + hk * ks1;
            long vo = (jb + j) * vs0 + hk * vs1;
            if (_kHalf)
            {
                Simd.HalfToFloat((ushort*)_k.P + ko, kb + (long)j * _d, _d);
            }
            else
            {
                Buffer.MemoryCopy((float*)_k.P + ko, kb + (long)j * _d, _d * sizeof(float), _d * sizeof(float));
            }

            if (_vHalf)
            {
                Simd.HalfToFloat((ushort*)_v.P + vo, vb + (long)j * _dv, _dv);
            }
            else
            {
                Buffer.MemoryCopy((float*)_v.P + vo, vb + (long)j * _dv, _dv * sizeof(float), _dv * sizeof(float));
            }
        }

        if (nk < KeyBlock)
        {
            new Span<float>(kb + (long)nk * _d, (KeyBlock - nk) * _d).Clear();
            new Span<float>(vb + (long)nk * _dv, (KeyBlock - nk) * _dv).Clear();
        }
    }

    /// <summary>sc[j][lanes] = sum_i qt[i][lanes] * kb[j][i] for the whole key block, eight keys at a time.</summary>
    private static void Scores<TL>(float* qt, float* kb, float* sc, int d, int lanes) where TL : struct, IFloatLanes<TL>
    {
        for (int j = 0; j < KeyBlock; j += ScoreKeys)
        {
            float* k0 = kb + (long)j * d;
            float* k1 = k0 + d;
            float* k2 = k1 + d;
            float* k3 = k2 + d;
            float* k4 = k3 + d;
            float* k5 = k4 + d;
            float* k6 = k5 + d;
            float* k7 = k6 + d;
            float* q = qt;
            var a0 = TL.Zero;
            var a1 = TL.Zero;
            var a2 = TL.Zero;
            var a3 = TL.Zero;
            var a4 = TL.Zero;
            var a5 = TL.Zero;
            var a6 = TL.Zero;
            var a7 = TL.Zero;
            for (nint i = 0; i < d; i++, q += lanes)
            {
                var qv = TL.Load(q);
                a0 = TL.Fma(qv, TL.Broadcast(k0[i]), a0);
                a1 = TL.Fma(qv, TL.Broadcast(k1[i]), a1);
                a2 = TL.Fma(qv, TL.Broadcast(k2[i]), a2);
                a3 = TL.Fma(qv, TL.Broadcast(k3[i]), a3);
                a4 = TL.Fma(qv, TL.Broadcast(k4[i]), a4);
                a5 = TL.Fma(qv, TL.Broadcast(k5[i]), a5);
                a6 = TL.Fma(qv, TL.Broadcast(k6[i]), a6);
                a7 = TL.Fma(qv, TL.Broadcast(k7[i]), a7);
            }

            float* s = sc + (long)j * lanes;
            TL.Store(a0, s);
            TL.Store(a1, s + lanes);
            TL.Store(a2, s + 2 * lanes);
            TL.Store(a3, s + 3 * lanes);
            TL.Store(a4, s + 4 * lanes);
            TL.Store(a5, s + 5 * lanes);
            TL.Store(a6, s + 6 * lanes);
            TL.Store(a7, s + 7 * lanes);
        }
    }

    /// <summary>ot[c][lanes] = alpha * ot[c][lanes] + sum_j p[j][lanes] * vb[j][c], eight value columns at a time.</summary>
    private static void AccumulateValues<TL>(float* p, float* vb, float* ot, TL alpha, int dv, int lanes) where TL : struct, IFloatLanes<TL>
    {
        for (int c = 0; c < dv; c += ValueColumns)
        {
            float* o = ot + (long)c * lanes;
            var o0 = TL.Multiply(TL.Load(o), alpha);
            var o1 = TL.Multiply(TL.Load(o + lanes), alpha);
            var o2 = TL.Multiply(TL.Load(o + 2 * lanes), alpha);
            var o3 = TL.Multiply(TL.Load(o + 3 * lanes), alpha);
            var o4 = TL.Multiply(TL.Load(o + 4 * lanes), alpha);
            var o5 = TL.Multiply(TL.Load(o + 5 * lanes), alpha);
            var o6 = TL.Multiply(TL.Load(o + 6 * lanes), alpha);
            var o7 = TL.Multiply(TL.Load(o + 7 * lanes), alpha);
            float* v = vb + c;
            float* pj = p;
            for (int j = 0; j < KeyBlock; j++, v += dv, pj += lanes)
            {
                var pv = TL.Load(pj);
                o0 = TL.Fma(pv, TL.Broadcast(v[0]), o0);
                o1 = TL.Fma(pv, TL.Broadcast(v[1]), o1);
                o2 = TL.Fma(pv, TL.Broadcast(v[2]), o2);
                o3 = TL.Fma(pv, TL.Broadcast(v[3]), o3);
                o4 = TL.Fma(pv, TL.Broadcast(v[4]), o4);
                o5 = TL.Fma(pv, TL.Broadcast(v[5]), o5);
                o6 = TL.Fma(pv, TL.Broadcast(v[6]), o6);
                o7 = TL.Fma(pv, TL.Broadcast(v[7]), o7);
            }

            TL.Store(o0, o);
            TL.Store(o1, o + lanes);
            TL.Store(o2, o + 2 * lanes);
            TL.Store(o3, o + 3 * lanes);
            TL.Store(o4, o + 4 * lanes);
            TL.Store(o5, o + 5 * lanes);
            TL.Store(o6, o + 6 * lanes);
            TL.Store(o7, o + 7 * lanes);
        }
    }

    /// <summary>Normalizes the lanes' accumulators and writes them to their output rows.</summary>
    private void WriteRows(float* ot, float* state, int lanes, long t0, int tokens, int h0, int heads)
    {
        for (int r = 0; r < lanes; r++)
        {
            int ti = r / _headsPerTask;
            int hi = r % _headsPerTask;
            if (ti >= tokens || hi >= heads)
            {
                continue;
            }

            float l = state[lanes + r];
            float inv = l > 0f ? 1f / l : 0f;
            float* o = (float*)_y.P + ((t0 + ti) * _hq + h0 + hi) * _dv;
            for (int c = 0; c < _dv; c++)
            {
                o[c] = ot[(long)c * lanes + r] * inv;
            }
        }
    }

    /// <summary>Combines the key splits of every task: rescales each split's sum to the common maximum.</summary>
    private void MergeSplits(long baseTasks, int lanes)
    {
        int dv = _dv;
        float* acc = stackalloc float[dv];
        for (long bt = 0; bt < baseTasks; bt++)
        {
            int tokenBlock = (int)(bt % _tokenBlocks);
            long rest = bt / _tokenBlocks;
            int headBlock = (int)(rest % _headBlocks);
            int hk = (int)(rest / _headBlocks);
            int group = _hq / _hkv;
            int h0 = hk * group + headBlock * _headsPerTask;
            int heads = Math.Min(_headsPerTask, group - headBlock * _headsPerTask);
            long t0 = (long)tokenBlock * _tokensPerTask;
            int tokens = (int)Math.Min(_tokensPerTask, _t - t0);
            for (int r = 0; r < lanes; r++)
            {
                int ti = r / _headsPerTask;
                int hi = r % _headsPerTask;
                if (ti >= tokens || hi >= heads)
                {
                    continue;
                }

                float max = float.NegativeInfinity;
                for (int s = 0; s < _splits; s++)
                {
                    max = MathF.Max(max, _partials[((bt * _splits + s) * lanes + r) * (dv + 2)]);
                }

                new Span<float>(acc, dv).Clear();
                float total = 0f;
                if (max != float.NegativeInfinity)
                {
                    for (int s = 0; s < _splits; s++)
                    {
                        float* pr = _partials + ((bt * _splits + s) * lanes + r) * (dv + 2);
                        float w = MathF.Exp(pr[0] - max);
                        total += w * pr[1];
                        Simd.Axpy(acc, pr + 2, w, dv);
                    }
                }

                float* o = (float*)_y.P + ((t0 + ti) * _hq + h0 + hi) * dv;
                Simd.Scale(acc, o, total > 0f ? 1f / total : 0f, dv);
            }
        }
    }
}
