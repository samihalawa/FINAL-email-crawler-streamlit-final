from serpapi import GoogleSearch

def do_search():
    params = {
        "q": "software engineer barcelona email contact",
        "location": "Barcelona, Spain",
        "hl": "es",
        "gl": "es",
        "google_domain": "google.es",
        "num": 5
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        for result in results.get("organic_results", []):
            print(f"Title: {result.get(\"title\")}")
            print(f"Link: {result.get(\"link\")}")
            print("---")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    do_search()
