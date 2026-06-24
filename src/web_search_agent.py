from tavily import TavilyClient

from src.config import TAVILY_API_KEY


client = TavilyClient(
    api_key=TAVILY_API_KEY
)


def web_search(question):

    response = client.search(
        query=question,
        search_depth="advanced",
        max_results=5
    )

    results = []

    for item in response["results"]:

        results.append(
            item["content"]
        )

    return "\n\n".join(results)