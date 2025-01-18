import requests
from bs4 import BeautifulSoup

def google_search():
    query = "software engineer barcelona email contact"
    url = f"https://www.google.com/search?q={query}&num=5"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        search_results = soup.find_all("div", class_="g")
        for result in search_results:
            link = result.find("a")
            if link and "href" in link.attrs:
                print(link["href"])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    google_search()
