from app.tools.web_search import web_search


print("=" * 60)
print("TAVILY WEB SEARCH TEST")
print("=" * 60)

query = "latest AI developments in 2026"

print(f"\nSearching: {query}")

result = web_search(query)

if result["success"]:

    print("\nSearch successful!")

    for index, item in enumerate(result["results"], start=1):

        print(f"\n--- Result {index} ---")
        print("Title:", item["title"])
        print("URL:", item["url"])
        print("Content:", item["content"][:500])

else:

    print("\nSearch failed:")
    print(result["error"])

print("\n" + "=" * 60)