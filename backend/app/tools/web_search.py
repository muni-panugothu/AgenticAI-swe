import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not configured in .env")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def web_search(query: str):
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )

        results = response.get("results", [])

        if not results:
            return {
                "success": False,
                "error": "No search results found"
            }

        formatted_results = []

        for result in results:
            formatted_results.append({
                "title": result.get("title"),
                "url": result.get("url"),
                "content": result.get("content")
            })

        return {
            "success": True,
            "query": query,
            "results": formatted_results
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }



# def web_search(query):
#     try:
#         url = "https://www.google.com/search"

#         params = {
#             "q" : query
#         }

#         headers = {
#             "User-Agent": "Mozilla/5.0"
#         }

#         response = requests.get(
#             url,
#             params=params,
#             headers=headers,
#             timeout=10
#         )
#         return response.text[:5000]
#     except Exception as e:
#         return f"Search error:{e}"