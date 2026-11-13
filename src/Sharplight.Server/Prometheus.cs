using System.Globalization;
using System.Text;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Server;

/// <summary>Renders an <see cref="EngineMetricsSnapshot"/> in the Prometheus text exposition format.</summary>
internal static class Prometheus
{
    public static string Render(EngineMetricsSnapshot m, string model) => Render([(model, m)]);

    /// <summary>One block per metric (HELP/TYPE once) with a sample per model.</summary>
    public static string Render(IReadOnlyList<(string Model, EngineMetricsSnapshot M)> models)
    {
        var sb = new StringBuilder();
        void Each(string name, string type, string help, Func<EngineMetricsSnapshot, double> v)
        {
            if (models.Count == 0)
            {
                return;
            }

            sb.Append($"# HELP {name} {help}\n# TYPE {name} {type}\n");
            foreach (var (model, m) in models)
            {
                sb.Append($"{name}{{model=\"{Escape(model)}\"}} {Num(v(m))}\n");
            }
        }
        void Gauge(string name, string help, Func<EngineMetricsSnapshot, double> v) => Each(name, "gauge", help, v);
        void Counter(string name, string help, Func<EngineMetricsSnapshot, double> v) => Each(name, "counter", help, v);

        Gauge("sharplight_queue_depth", "Requests waiting for a slot.", m => m.QueueDepth);
        Gauge("sharplight_active_sequences", "Sequences currently holding a slot.", m => m.ActiveSequences);
        Gauge("sharplight_slots", "Sequence slots of the decoder.", m => m.SlotCount);
        Gauge("sharplight_slot_utilization", "Fraction of slots in use.", m => m.SlotUtilization);
        Gauge("sharplight_tokens_per_second", "Completion tokens per second over the last 10 seconds.", m => m.RecentTokensPerSecond);
        Gauge("sharplight_prefix_cache_hit_ratio", "Fraction of admitted requests that reused cached context.", m => m.PrefixCacheHitRate);
        Counter("sharplight_requests_submitted_total", "Requests accepted into the queue.", m => m.RequestsSubmitted);
        Counter("sharplight_requests_rejected_total", "Requests refused (queue full, prompt too long, shutting down).", m => m.RequestsRejected);
        Counter("sharplight_prompt_tokens_total", "Prompt tokens of admitted requests.", m => m.PromptTokens);
        Counter("sharplight_cached_prompt_tokens_total", "Prompt tokens served from the prefix cache.", m => m.CachedPromptTokens);
        Counter("sharplight_completion_tokens_total", "Generated tokens.", m => m.CompletionTokens);
        Counter("sharplight_prefix_cache_lookups_total", "Prefix cache lookups (one per admitted request).", m => m.PrefixCacheLookups);
        Counter("sharplight_prefix_cache_hits_total", "Lookups that reused cached context.", m => m.PrefixCacheHits);
        Counter("sharplight_steps_total", "Forward passes executed.", m => m.Steps);
        Gauge("sharplight_kv_capacity_tokens", "KV-cache capacity in tokens (slots x context per slot, or the paged pool).", m => m.KvCapacityTokens);
        Gauge("sharplight_kv_tokens", "Tokens held in slot contexts, including idle slots kept for prefix reuse.", m => m.KvTokens);
        Gauge("sharplight_kv_active_tokens", "Tokens held by running sequences.", m => m.KvActiveTokens);
        Gauge("sharplight_kv_utilization", "Fraction of the KV capacity held by running sequences.", m => m.KvUtilization);
        Counter("sharplight_spec_draft_tokens_total", "Draft tokens proposed by speculative decoding.", m => m.DraftTokens);
        Counter("sharplight_kv_preemptions_total", "Sequences preempted for KV blocks (paged cache) and recomputed.", m => m.Preemptions);
        Counter("sharplight_kv_prefix_hit_tokens_total", "Prompt tokens served from KV blocks shared between slots.", m => m.SharedPrefixTokens);
        Counter("sharplight_spec_accepted_tokens_total", "Draft tokens accepted by the model.", m => m.AcceptedDraftTokens);

        if (models.Count > 0)
        {
            sb.Append("# HELP sharplight_requests_finished_total Finished requests by reason.\n# TYPE sharplight_requests_finished_total counter\n");
            foreach (var (model, m) in models)
            {
                foreach (var (reason, n) in m.RequestsFinished)
                {
                    sb.Append($"sharplight_requests_finished_total{{model=\"{Escape(model)}\",reason=\"{reason.ToString().ToLowerInvariant()}\"}} {n}\n");
                }
            }
        }

        Histogram(sb, "sharplight_time_to_first_token_seconds", "Time from submission to the first token.", models, m => m.TimeToFirstToken);
        Histogram(sb, "sharplight_inter_token_latency_seconds", "Time between consecutive tokens of a request.", models, m => m.InterTokenLatency);
        Histogram(sb, "sharplight_queue_wait_seconds", "Time a request waited for a slot.", models, m => m.QueueWait);
        Histogram(sb, "sharplight_request_duration_seconds", "Total request time.", models, m => m.RequestDuration);
        Histogram(sb, "sharplight_step_duration_seconds", "Duration of one forward pass.", models, m => m.StepDuration);
        Histogram(sb, "sharplight_step_rows", "Rows (tokens) per forward pass.", models, m => m.StepRows);
        Histogram(sb, "sharplight_batch_size", "Sequences per forward pass.", models, m => m.StepSequences);
        return sb.ToString();
    }

    /// <summary>Model states and device memory of a <see cref="ModelHost"/>.</summary>
    public static string RenderHost(ModelHost host)
    {
        var sb = new StringBuilder();
        var list = host.List();
        if (list.Count > 0)
        {
            sb.Append("# HELP sharplight_model_loaded 1 when the model is loaded.\n# TYPE sharplight_model_loaded gauge\n");
            foreach (var m in list)
            {
                sb.Append($"sharplight_model_loaded{{model=\"{Escape(m.Name)}\",kind=\"{m.Kind.ToString().ToLowerInvariant()}\"}} {(m.State == ModelState.Loaded ? 1 : 0)}\n");
            }

            sb.Append("# HELP sharplight_model_resident_bytes Device memory held by the model.\n# TYPE sharplight_model_resident_bytes gauge\n");
            foreach (var m in list)
            {
                sb.Append($"sharplight_model_resident_bytes{{model=\"{Escape(m.Name)}\"}} {m.ResidentBytes}\n");
            }
        }
        var pools = host.MemoryPools();
        if (pools.Count > 0)
        {
            sb.Append("# HELP sharplight_device_memory_bytes Device memory by kind (available = what this process may still allocate).\n# TYPE sharplight_device_memory_bytes gauge\n");
            foreach (var (d, i) in pools)
            {
                foreach (var (kind, v) in new[] { ("total", i.Total), ("budget", i.Budget), ("used", i.Usage), ("used_by_sharplight", i.UsedByUs), ("available", i.Available) })
                {
                    sb.Append($"sharplight_device_memory_bytes{{device=\"{Escape(d.Name)}\",kind=\"{kind}\"}} {v}\n");
                }
            }
        }
        return sb.ToString();
    }

    private static string Escape(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n");
    private static string Num(double v) => v.ToString("R", CultureInfo.InvariantCulture);

    private static void Histogram(StringBuilder sb, string name, string help, IReadOnlyList<(string Model, EngineMetricsSnapshot M)> models, Func<EngineMetricsSnapshot, HistogramSnapshot> pick)
    {
        if (models.Count == 0)
        {
            return;
        }

        sb.Append($"# HELP {name} {help}\n# TYPE {name} histogram\n");
        foreach (var (model, m) in models)
        {
            HistogramSamples(sb, name, pick(m), model);
        }
    }

    private static void HistogramSamples(StringBuilder sb, string name, HistogramSnapshot h, string model)
    {
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
