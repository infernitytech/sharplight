"""Sends N greedy completions concurrently (natural prompts, staggered by a few ms so steps mix prompts and decoding) and writes
the texts as JSON lines, one per request index. usage: batch_greedy.py <url> <out.jsonl> [n] [max_tokens]"""
import asyncio, json, sys, httpx
WORDS = "water power history city music river science market family garden island engine forest school ocean".split()
async def one(client, url, i, n_tok):
    await asyncio.sleep(0.004 * i)
    a, b, c = WORDS[i % len(WORDS)], WORDS[(3 * i + 1) % len(WORDS)], WORDS[(7 * i + 2) % len(WORDS)]
    prompt = (f"Write a detailed essay about the role of {a} and {b} in modern {c}. " * (1 + i % 9)) + "Explain the history.\n\n"
    r = await client.post(url + "/v1/completions", json={"model": "m", "prompt": prompt, "max_tokens": n_tok, "temperature": 0})
    return i, r.json()["choices"][0]["text"]
async def main():
    url, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    n_tok = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    async with httpx.AsyncClient(timeout=600) as client:
        res = await asyncio.gather(*(one(client, url, i, n_tok) for i in range(n)))
    with open(out, "w") as f:
        for i, t in sorted(res):
            f.write(json.dumps({"i": i, "text": t}) + "\n")
asyncio.run(main())
