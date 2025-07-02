from tools.local_search_tool import LocalSemanticSearch

search_tool = LocalSemanticSearch("CVE_embeddings.csv")

query = input("Enter your search query: ")
results = search_tool.search(query)

for i, (chunk, score) in enumerate(results):
    print(f"\nResult {i+1} (Score: {score:.4f}):\n{chunk}")
