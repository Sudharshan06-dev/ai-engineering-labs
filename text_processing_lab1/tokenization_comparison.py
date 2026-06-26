from input import TEST_CASES, gpt4o_enc, claude_enc, llama_enc


# ─── TOKENIZATION COMPARISON ──────────────────────────────────────────────────
print("=" * 70)
print(f"{'TEST CASE':<25} {'GPT-4o':>8} {'Claude':>8} {'LLaMA':>8} {'Chars':>8}")
print("=" * 70)

for name, text in TEST_CASES.items():
    gpt_count   = len(gpt4o_enc.encode(text))
    claude_count = len(claude_enc.encode(text))
    llama_count = len(llama_enc.encode(text))
    char_count  = len(text)
    ratio_gpt   = char_count / gpt_count if gpt_count else 0

    print(f"{name:<25} {gpt_count:>8} {claude_count:>8} {llama_count:>8} {char_count:>8}")

print()