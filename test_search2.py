from googlesearch import search

def do_search():
    query = "software engineer barcelona site:es"
    try:
        for url in search(query, stop=5):
            print(url)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    do_search()
