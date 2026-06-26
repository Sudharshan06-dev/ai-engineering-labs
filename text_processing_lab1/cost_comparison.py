from input import PRICING, RONALDO_PARAGRAPH, gpt4o_enc, claude_enc, llama_enc

# ─── COST ESTIMATION FOR THE RONALDO PARAGRAPH ────────────────────────────────
def estimate_cost(text, model_name, enc, pricing):
    tokens = len(enc.encode(text))
    input_cost = (tokens / 1_000_000) * pricing["input"]
    return tokens, input_cost

if __name__ == "__main__":
  print("=" * 70)
  print("COST ESTIMATION — Ronaldo Paragraph (Input Only)")
  print("=" * 70)

  for model, pricing in PRICING.items():
      if model == "GPT-4o":
          enc = gpt4o_enc
      elif model == "Claude Sonnet":
          enc = claude_enc
      else:
          enc = llama_enc

      tokens, cost = estimate_cost(RONALDO_PARAGRAPH, model, enc, pricing)
      print(f"{model:<30} {tokens:>6} tokens   ${cost:.6f}")