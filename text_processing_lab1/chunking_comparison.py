from input import RONALDO_PARAGRAPH, gpt4o_enc, llama_enc, claude_enc

def chunk_by_tokens(text, enc, chunk_size):
    """Chunk text into token-limited segments."""
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

if __name__ == "__main__":
    # Use the full Ronaldo paragraph for this
    chunks_512 = chunk_by_tokens(RONALDO_PARAGRAPH, gpt4o_enc, 512)
    chunks_256 = chunk_by_tokens(RONALDO_PARAGRAPH, gpt4o_enc, 256)

    print("=" * 70)
    print("CHUNKING ANALYSIS")
    print("=" * 70)
    print(f"512-token chunks: {len(chunks_512)} chunks")
    print(f"256-token chunks: {len(chunks_256)} chunks")
    print()

    # The KEY insight — look at where the boundary falls
    print("── Last chunk of chunk 1 (512) ──")
    print(repr(chunks_512[-1]))
    print()
    print("── First chunk of chunk 2 (512) ──")
    print(repr(chunks_512[0]))
    print()
    print("── Last chunk of chunk 1 (256) ──")
    print(repr(chunks_256[-1]))
    print()
    print("── First chunk of chunk 2 (256) ──")
    print(repr(chunks_256[0]))
    