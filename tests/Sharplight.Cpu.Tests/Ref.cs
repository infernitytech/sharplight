namespace Sharplight.Cpu.Tests;

/// <summary>Straightforward scalar (double precision) reference implementations.</summary>
internal static class Ref
{
    public static float[] RmsNorm(float[] x, int rows, int n, float[]? w, float eps)
    {
        var y = new float[x.Length];
        for (int r = 0; r < rows; r++)
        {
            double s = 0;
            for (int i = 0; i < n; i++)
            {
                s += (double)x[r * n + i] * x[r * n + i];
            }

            double sc = 1.0 / Math.Sqrt(s / n + eps);
            for (int i = 0; i < n; i++)
            {
                y[r * n + i] = (float)(x[r * n + i] * sc * (w?[i] ?? 1f));
            }
        }
        return y;
    }

    public static float[] L2Norm(float[] x, int rows, int n, float eps)
    {
        var y = new float[x.Length];
        for (int r = 0; r < rows; r++)
        {
            double s = 0;
            for (int i = 0; i < n; i++)
            {
                s += (double)x[r * n + i] * x[r * n + i];
            }

            double sc = 1.0 / Math.Sqrt(s + eps);
            for (int i = 0; i < n; i++)
            {
                y[r * n + i] = (float)(x[r * n + i] * sc);
            }
        }
        return y;
    }

    public static float[] MatMul(float[] x, int rows, int k, float[] w, int n)
    {
        var y = new float[rows * n];
        for (int t = 0; t < rows; t++)
        {
            for (int r = 0; r < n; r++)
            {
                double s = 0;
                for (int c = 0; c < k; c++)
                {
                    s += (double)x[t * k + c] * w[r * k + c];
                }
                y[t * n + r] = (float)s;
            }
        }

        return y;
    }

    public static float Silu(float x) => (float)(x / (1 + Math.Exp(-x)));
    public static float Sigmoid(float x) => (float)(1 / (1 + Math.Exp(-x)));
    public static float Softplus(float x) => x > 20 ? x : (float)Math.Log(1 + Math.Exp(x));

    /// <summary>x [T,H,D]; positions [T,A].</summary>
    public static float[] Rope(float[] x, int T, int H, int D, int[] pos, int A, int dims, float @base, string style, int[]? sections, bool interleaved, float freqScale, float[]? ff = null)
    {
        var y = (float[])x.Clone();
        int half = dims / 2;
        for (int t = 0; t < T; t++)
        {
            var cs = new (double c, double s)[half];
            for (int i = 0; i < half; i++)
            {
                int axis = 0;
                if (sections is { Length: 4 })
                {
                    int sd = sections.Sum();
                    int sec = i % sd;
                    int secW = sections[0] + sections[1];
                    if (interleaved)
                    {
                        if (sec % 3 == 1 && sec < 3 * sections[1])
                        {
                            axis = 1;
                        }
                        else if (sec % 3 == 2 && sec < 3 * sections[2])
                        {
                            axis = 2;
                        }
                        else if (sec % 3 == 0 && sec < 3 * sections[0])
                        {
                            axis = 0;
                        }
                        else
                        {
                            axis = 3;
                        }
                    }
                    else if (sec >= sections[0] && sec < secW)
                    {
                        axis = 1;
                    }
                    else if (sec >= secW && sec < secW + sections[2])
                    {
                        axis = 2;
                    }
                    else if (sec >= secW + sections[2])
                    {
                        axis = 3;
                    }
                }
                double p = axis < A ? pos[t * A + axis] : 0;
                double theta = p * Math.Pow(@base, -2.0 * i / dims) * freqScale / (ff is null ? 1.0 : ff[i]);
                cs[i] = (Math.Cos(theta), Math.Sin(theta));
            }
            for (int h = 0; h < H; h++)
            {
                int o = (t * H + h) * D;
                for (int i = 0; i < half; i++)
                {
                    int i0 = style == "gptj" ? 2 * i : i;
                    int i1 = style == "gptj" ? 2 * i + 1 : i + half;
                    double x0 = x[o + i0];
                    double x1 = x[o + i1];
                    y[o + i0] = (float)(x0 * cs[i].c - x1 * cs[i].s);
                    y[o + i1] = (float)(x0 * cs[i].s + x1 * cs[i].c);
                }
            }
        }
        return y;
    }

    /// <summary>q [T,Hq,D]; k [S,Hkv,D]; v [S,Hkv,Dv]. Query t sees keys [0, past+t] (causal) or all S keys.</summary>
    public static float[] Attention(float[] q, int T, int Hq, int D, float[] k, float[] v, int S, int Hkv, int Dv, float scale, int past, bool causal, float[]? mask = null, int window = 0, float softcap = 0f, float[]? sinks = null)
    {
        var o = new float[T * Hq * Dv];
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < Hq; h++)
            {
                int hk = h / (Hq / Hkv);
                int n = causal ? past + t + 1 : S;
                var sc = new double[n];
                int j0 = window > 0 ? Math.Max(0, n - window) : 0;
                for (int j = j0; j < n; j++)
                {
                    double d = 0;
                    for (int i = 0; i < D; i++)
                    {
                        d += (double)q[(t * Hq + h) * D + i] * k[(j * Hkv + hk) * D + i];
                    }

                    sc[j] = d * scale + (mask is null ? 0 : mask[t * S + j]);
                    if (softcap > 0)
                    {
                        sc[j] = softcap * Math.Tanh(sc[j] / softcap);
                    }
                }
                double mx = sinks is null ? double.NegativeInfinity : sinks[h];
                for (int j = j0; j < n; j++)
                {
                    mx = Math.Max(mx, sc[j]);
                }

                double sum = sinks is null ? 0 : Math.Exp(sinks[h] - mx);
                for (int j = j0; j < n; j++)
                {
                    sc[j] = Math.Exp(sc[j] - mx);
                    sum += sc[j];
                }
                for (int i = 0; i < Dv; i++)
                {
                    double a = 0;
                    for (int j = j0; j < n; j++)
                    {
                        a += sc[j] / sum * v[(j * Hkv + hk) * Dv + i];
                    }

                    o[(t * Hq + h) * Dv + i] = (float)a;
                }
            }
        }

        return o;
    }

    /// <summary>x [T,C], w [C,K], state [(K-1),C] is updated.</summary>
    public static float[] Conv(float[] x, int T, int C, float[] w, int K, float[] state, bool silu)
    {
        var win = new float[(K - 1 + T) * C];
        Array.Copy(state, win, (K - 1) * C);
        Array.Copy(x, 0, win, (K - 1) * C, T * C);
        var y = new float[T * C];
        for (int t = 0; t < T; t++)
        {
            for (int c = 0; c < C; c++)
            {
                double s = 0;
                for (int j = 0; j < K; j++)
                {
                    s += (double)w[c * K + j] * win[(t + j) * C + c];
                }

                y[t * C + c] = silu ? Silu((float)s) : (float)s;
            }
        }

        Array.Copy(win, T * C, state, 0, (K - 1) * C);
        return y;
    }

    /// <summary>state [Hv, Dv, Dk] updated in place.</summary>
    public static float[] Delta(float[] q, float[] k, float[] v, float[] g, float[] beta, float[] S, int T, int Hk, int Hv, int Dk, int Dv, float scale)
    {
        var o = new float[T * Hv * Dv];
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < Hv; h++)
            {
                int hq = h % Hk;
                double decay = Math.Exp(g[t * Hv + h]);
                for (int j = 0; j < Dv; j++)
                {
                    int b = (h * Dv + j) * Dk;
                    double kv = 0;
                    for (int i = 0; i < Dk; i++)
                    {
                        S[b + i] = (float)(S[b + i] * decay);
                        kv += (double)S[b + i] * k[(t * Hk + hq) * Dk + i];
                    }
                    double delta = (v[(t * Hv + h) * Dv + j] - kv) * beta[t * Hv + h];
                    double acc = 0;
                    for (int i = 0; i < Dk; i++)
                    {
                        S[b + i] = (float)(S[b + i] + k[(t * Hk + hq) * Dk + i] * delta);
                        acc += (double)S[b + i] * q[(t * Hk + hq) * Dk + i];
                    }
                    o[(t * Hv + h) * Dv + j] = (float)(acc * scale);
                }
            }
        }

        return o;
    }

    public static float[] Softmax(float[] x, int rows, int n, float scale)
    {
        var y = new float[x.Length];
        for (int r = 0; r < rows; r++)
        {
            double mx = double.NegativeInfinity;
            for (int i = 0; i < n; i++)
            {
                mx = Math.Max(mx, (double)x[r * n + i] * scale);
            }

            double s = 0;
            for (int i = 0; i < n; i++)
            {
                s += Math.Exp(x[r * n + i] * scale - mx);
            }

            for (int i = 0; i < n; i++)
            {
                y[r * n + i] = (float)(Math.Exp(x[r * n + i] * scale - mx) / s);
            }
        }
        return y;
    }
}
