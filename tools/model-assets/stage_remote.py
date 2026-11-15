"""Phase 1 of the sandboxed remote-code generators: downloads only code, configs and tokenizer files (no weights) of the source repositories
into <stage>/<family>. Phase 2 (run_remote_sandboxed.sh) runs the generators with no network access and no credentials.

    python stage_remote.py <stage dir>
"""
import sys
from huggingface_hub import snapshot_download

stage = sys.argv[1]
REPOS = {"spark2_5": "XHToken/Spark-X2.6-4B", "clover_moe": "Diffuspace/Clover-35B-A3B-Preview", "mimo_v2": "XiaomiMiMo/MiMo-V2.6-Pro-RL",
         "llava_onevision2": "microsoft/Mage-VL-Cowork", "lfm3": "LiquidAI/LFM3-1.7B"}
PATTERNS = ["*.py", "config.json", "tokenizer*", "special_tokens_map.json", "generation_config.json", "chat_template.jinja", "vocab.json", "merges.txt", "*.model"]
for name, repo in REPOS.items():
    snapshot_download(repo, local_dir=f"{stage}/{name}", allow_patterns=PATTERNS)
    print("staged", name, flush=True)
