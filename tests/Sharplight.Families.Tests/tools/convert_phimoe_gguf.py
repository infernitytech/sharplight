"""
Converts a transformers phimoe checkpoint (PhimoeForCausalLM, e.g. ~/models/hf/tiny-random-phimoe) to GGUF with llama.cpp's converter, which
only knows the remote-code class name PhiMoEForCausalLM, requires rope_parameters.original_max_position_embeddings and reads a SentencePiece
tokenizer.model. The checkpoint is copied with the class renamed and the key added, and the converter is patched to take the GPT-2 style
tokenizer.json of the tiny checkpoints.

    python convert_phimoe_gguf.py <llama.cpp source> <checkpoint directory> <output .gguf>
"""
import json, os, shutil, sys, tempfile

llama, source, output = sys.argv[1:4]
sys.path.insert(0, llama)
sys.path.insert(0, os.path.join(llama, "gguf-py"))

with tempfile.TemporaryDirectory() as work:
    for name in os.listdir(source):
        shutil.copy(os.path.join(source, name), work)
    path = os.path.join(work, "config.json")
    config = json.load(open(path))
    config["architectures"] = ["PhiMoEForCausalLM"]
    rope = config.setdefault("rope_parameters", {})
    rope.setdefault("original_max_position_embeddings", config["max_position_embeddings"])
    json.dump(config, open(path, "w"), indent=1)

    import convert_hf_to_gguf
    from conversion import phi
    if not os.path.exists(os.path.join(work, "tokenizer.model")):
        phi.PhiMoeModel.set_vocab = lambda self: self._set_vocab_gpt2()
        phi.PhiMoeModel.get_vocab_base_pre = lambda self, tokenizer: "gpt-2"
    sys.argv = [sys.argv[0], work, "--outtype", "f32", "--outfile", output]
    convert_hf_to_gguf.main()
