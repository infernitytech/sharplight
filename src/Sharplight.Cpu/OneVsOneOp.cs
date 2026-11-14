using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>One-vs-one voting and optional pairwise-coupled probabilities (see <c>StandardOps.OneVsOne</c>).</summary>
internal sealed unsafe class OneVsOneOp(TRef decisions, TRef winner, TRef? probabilities, int classes, float[] probA, float[] probB) : CpuOp
{
    private readonly int _pairs = classes * (classes - 1) / 2;

    public override string Name => "OneVsOne";

    public override void Run(RunParameters p)
    {
        long rows = decisions.Dim(0, p);
        var votes = new int[classes];
        var pairwise = new double[classes, classes];
        var estimates = new double[classes];
        for (long r = 0; r < rows; r++)
        {
            float* d = (float*)decisions.P + (r * _pairs);
            Array.Clear(votes);
            int pair = 0;
            for (int i = 0; i < classes; i++)
            {
                for (int j = i + 1; j < classes; j++, pair++)
                {
                    votes[d[pair] > 0 ? i : j]++;
                    if (probabilities is not null)
                    {
                        double pi = Math.Clamp(Platt(d[pair], probA[pair], probB[pair]), 1e-7, 1 - 1e-7);
                        pairwise[i, j] = pi;
                        pairwise[j, i] = 1 - pi;
                    }
                }
            }

            int best = 0;
            for (int c = 1; c < classes; c++)
            {
                best = votes[c] > votes[best] ? c : best;
            }

            ((int*)winner.P)[r] = best;
            if (probabilities is null)
            {
                continue;
            }

            float* o = (float*)probabilities.P + (r * classes);
            if (classes == 2)
            {
                o[0] = (float)pairwise[0, 1];
                o[1] = (float)pairwise[1, 0];
                continue;
            }

            Couple(pairwise, estimates);
            for (int c = 0; c < classes; c++)
            {
                o[c] = (float)estimates[c];
            }
        }
    }

    /// <summary>libsvm's sigmoid_predict: 1 / (1 + exp(d·A + B)), evaluated without overflow.</summary>
    private static double Platt(double decision, double a, double b)
    {
        double f = (decision * a) + b;
        return f >= 0 ? Math.Exp(-f) / (1 + Math.Exp(-f)) : 1 / (1 + Math.Exp(f));
    }

    /// <summary>libsvm's multiclass_probability (Wu, Lin and Weng 2004, method 2).</summary>
    private void Couple(double[,] r, double[] p)
    {
        int k = classes;
        var q = new double[k, k];
        var qp = new double[k];
        for (int t = 0; t < k; t++)
        {
            p[t] = 1.0 / k;
            for (int j = 0; j < t; j++)
            {
                q[t, t] += r[j, t] * r[j, t];
                q[t, j] = q[j, t];
            }

            for (int j = t + 1; j < k; j++)
            {
                q[t, t] += r[j, t] * r[j, t];
                q[t, j] = -r[j, t] * r[t, j];
            }
        }

        double eps = 0.005 / k;
        for (int iteration = 0; iteration < Math.Max(100, k); iteration++)
        {
            double pqp = 0;
            for (int t = 0; t < k; t++)
            {
                qp[t] = 0;
                for (int j = 0; j < k; j++)
                {
                    qp[t] += q[t, j] * p[j];
                }

                pqp += p[t] * qp[t];
            }

            double maxError = 0;
            for (int t = 0; t < k; t++)
            {
                maxError = Math.Max(maxError, Math.Abs(qp[t] - pqp));
            }

            if (maxError < eps)
            {
                break;
            }

            for (int t = 0; t < k; t++)
            {
                double diff = (-qp[t] + pqp) / q[t, t];
                p[t] += diff;
                pqp = (pqp + (diff * ((diff * q[t, t]) + (2 * qp[t])))) / (1 + diff) / (1 + diff);
                for (int j = 0; j < k; j++)
                {
                    qp[j] = (qp[j] + (diff * q[t, j])) / (1 + diff);
                    p[j] /= 1 + diff;
                }
            }
        }
    }
}
