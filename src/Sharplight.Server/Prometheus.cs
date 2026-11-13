using System.Globalization;
using System.Text;
using Sharplight.Llm.Serving;

namespace Sharplight.Server;

/// <summary>Renders an <see cref="EngineMetricsSnapshot"/> in the Prometheus text exposition format.</summary>
public static class Prometheus
{
    public static string Render(EngineMetricsSnapshot m, string model)
    {
        var sb = new StringBuilder();
        string lbl = $"{{model=\"{Escape(model)}\"}}";
        void Gauge(string name, string help, double v) => Metric(sb, name, "gauge", help, lbl, v);
        void Counter(string name, string help, double v) => Metric(sb, name, "counter", help, lbl, v);

        Gauge("sharplight_queue_depth", "Requests waiting for a slot.", m.QueueDepth);
        Gauge("sharplight_active_sequences", "Sequences currently holding a slot.", m.ActiveSequences);
        Gauge("sharplight_slots", "Sequence slots of the decoder.", m.SlotCount);
        Gauge("sharplight_slot_utilization", "Fraction of slots in use.", m.SlotUtilization);
        Gauge("sharplight_tokens_per_second", "Completion tokens per second over the last 10 seconds.", m.RecentTokensPerSecond);
        Gauge("sharplight_prefix_cache_hit_ratio", "Fraction of admitted requests that reused cached context.", m.PrefixCacheHitRate);
        Counter("sharplight_requests_submitted_total", "Requests accepted into the queue.", m.RequestsSubmitted);
        Counter("sharplight_requests_rejected_total", "Requests refused (queue full, prompt too long, shutting down).", m.RequestsRejected);
        Counter("sharplight_prompt_tokens_total", "Prompt tokens of admitted requests.", m.PromptTokens);
        Counter("sharplight_cached_prompt_tokens_total", "Prompt tokens served from the prefix cache.", m.CachedPromptTokens);
        Counter("sharplight_completion_tokens_total", "Generated tokens.", m.CompletionTokens);
        Counter("sharplight_prefix_cache_lookups_total", "Prefix cache lookups (one per admitted request).", m.PrefixCacheLookups);
        Counter("sharplight_prefix_cache_hits_total", "Lookups that reused cached context.", m.PrefixCacheHits);
        Counter("sharplight_steps_total", "Forward passes executed.", m.Steps);

        sb.Append("# HELP sharplight_requests_finished_total Finished requests by reason.\n# TYPE sharplight_requests_finished_total counter\n");
        foreach (var (reason, n) in m.RequestsFinished)
            sb.Append($"sharplight_requests_finished_total{{model=\"{Escape(model)}\",reason=\"{reason.ToString().ToLowerInvariant()}\"}} {n}\n");

        Histogram(sb, "sharplight_time_to_first_token_seconds", "Time from submission to the first token.", m.TimeToFirstToken, model);
        Histogram(sb, "sharplight_inter_token_latency_seconds", "Time between consecutive tokens of a request.", m.InterTokenLatency, model);
        Histogram(sb, "sharplight_queue_wait_seconds", "Time a request waited for a slot.", m.QueueWait, model);
        Histogram(sb, "sharplight_request_duration_seconds", "Total request time.", m.RequestDuration, model);
        Histogram(sb, "sharplight_step_duration_seconds", "Duration of one forward pass.", m.StepDuration, model);
        Histogram(sb, "sharplight_step_rows", "Rows (tokens) per forward pass.", m.StepRows, model);
        return sb.ToString();
    }

    static string Escape(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n");
    static string Num(double v) => v.ToString("R", CultureInfo.InvariantCulture);

    static void Metric(StringBuilder sb, string name, string type, string help, string labels, double v)
        => sb.Append($"# HELP {name} {help}\n# TYPE {name} {type}\n{name}{labels} {Num(v)}\n");

    static void Histogram(StringBuilder sb, string name, string help, HistogramSnapshot h, string model)
    {
        sb.Append($"# HELP {name} {help}\n# TYPE {name} histogram\n");
        long acc = 0;
        for (int i = 0; i < h.Counts.Length; i++)
        {
            acc += h.Counts[i];
            string le = i < h.UpperBounds.Length ? Num(h.UpperBounds[i]) : "+Inf";
            sb.Append($"{name}_bucket{{model=\"{Escape(model)}\",le=\"{le}\"}} {acc}\n");
        }
        sb.Append($"{name}_sum{{model=\"{Escape(model)}\"}} {Num(h.Sum)}\n{name}_count{{model=\"{Escape(model)}\"}} {h.Count}\n");
    }
}
