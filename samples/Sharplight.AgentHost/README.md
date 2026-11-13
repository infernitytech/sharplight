# Sharplight.AgentHost (`sharplight-agenthost`)

Serves a local Sharplight model as a [Microsoft Agent Framework](https://learn.microsoft.com/agent-framework/) agent
named `assistant`, exposed through OpenAI-compatible endpoints (Chat Completions, Responses and Conversations).

```bash
dotnet run -c Release --project samples/Sharplight.AgentHost -- ~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf --urls http://localhost:5100

curl localhost:5100/assistant/v1/chat/completions -H 'content-type: application/json' \
     -d '{"model":"assistant","messages":[{"role":"user","content":"hello"}],"stream":true}'
curl localhost:5100/assistant/v1/responses -H 'content-type: application/json' -d '{"input":"hello"}'
```

Options: `--cpu`, `--device N`, `--ctx N` (context length, default 4096) and `--urls URL`. Stop with Ctrl+C.
