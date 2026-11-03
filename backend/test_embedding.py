from app.services.embedding_service import create_embedding


print("=" * 60)
print("EMBEDDING TEST")
print("=" * 60)


text = "I am learning Agentic AI and RAG."

print("\nText:")
print(text)


try:

    embedding = create_embedding(text)

    print("\nEmbedding generated successfully!")

    print(
        "Embedding type:",
        type(embedding)
    )

    print(
        "Embedding dimensions:",
        len(embedding)
    )

    print("\nFirst 10 values:")

    print(
        embedding[:10]
    )


except Exception as e:

    print("\nEmbedding test failed:")

    print(e)


print("\n" + "=" * 60)