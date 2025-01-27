import requests
from bs4 import BeautifulSoup
import random
import re
import time

def get_free_proxy():
    proxy_list_url = "https://free-proxy-list.net/"
    try:
      response = requests.get(proxy_list_url, timeout=5)
      response.raise_for_status()
      soup = BeautifulSoup(response.content, "html.parser")
      proxy_table = soup.find("table", id="proxylisttable")
      proxies = []
      for row in proxy_table.find_all("tr"):
        data = row.find_all("td")
        if len(data) >= 2:
            ip = data[0].text
            port = data[1].text
            proxies.append(f"{ip}:{port}")
      if proxies:
          return random.choice(proxies)
      else:
          return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting proxies: {e}")
        return None

def search_google_with_lib(query, num_results=10):
    from googlesearch import search
    results = []
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    ]
    try:
        for i in range(0, num_results, 10):
            batch_results = search(query, num_results=10, start=i, advanced=True, lang="es", region="es", sleep_interval=random.uniform(2,5))
            results.extend(batch_results)
        return results
    except Exception as e:
      print(f"Error in googlesearch-python: {e}")
      return []

def search_google_custom(query, num_results=10):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    ]
    headers = {'User-Agent': random.choice(user_agents)}
    results = []
    for start in range(0, min(num_results, 100), 10):
        url = f"https://www.google.com/search?q={query}&num=10&hl=es&gl=es&start={start}"
        try:
          proxy = get_free_proxy()
          if proxy:
              proxies = {
                  "http": f"http://{proxy}",
                  "https": f"https://{proxy}",
              }
              response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
          else:
            response = requests.get(url, headers=headers, timeout=10)
          response.raise_for_status()

          soup = BeautifulSoup(response.content, "html.parser")
          search_results = soup.find_all("div", class_="g")

          for result in search_results:
              link = result.find("a", href=True)
              if link:
                  title_tag = result.find("h3")
                  title = title_tag.text if title_tag else "N/A"
                  description_tag = result.find("div", class_="IsZvec")
                  description = description_tag.text if description_tag else "N/A"
                  results.append({"url": link["href"], "title": title, "description": description})

        except requests.exceptions.RequestException as e:
            print(f"Error during request: {e}")
            time.sleep(random.uniform(30, 60))
            continue
        except Exception as e:
            print(f"An error occurred: {e}")

        time.sleep(random.uniform(5, 10))
    return results

def search_duckduckgo(query, num_results=10):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:91.0) Gecko/20100101 Firefox/91.0",
    ]
    headers = {'User-Agent': random.choice(user_agents)}
    results = []
    try:
      for start in range(0, num_results, 10):
        url = f"https://duckduckgo.com/html/?q={query}&start={start}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        search_results = soup.find_all("div", class_="result results_links_deep web-result")

        for result in search_results:
          link = result.find("a", class_="result__a")
          if link:
            title_tag = result.find("h2", class_="result__title")
            title = title_tag.text if title_tag else "N/A"
            description_tag = result.find("div", class_="result__snippet")
            description = description_tag.text if description_tag else "N/A"
            results.append({"url": link["href"], "title": title, "description": description})
        time.sleep(random.uniform(2, 5))

    except requests.exceptions.RequestException as e:
       print(f"Error during request: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return results

def search_bing(query, num_results=10):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:91.0) Gecko/20100101 Firefox/91.0",
    ]
    headers = {'User-Agent': random.choice(user_agents)}
    results = []
    try:
      for start in range(0, num_results, 10):
        url = f"https://www.bing.com/search?q={query}&first={start+1}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        search_results = soup.find_all("li", class_="b_algo")
        for result in search_results:
            link = result.find("a", href=True)
            if link:
              title_tag = result.find("h2")
              title = title_tag.text if title_tag else "N/A"
              description_tag = result.find("p")
              description = description_tag.text if description_tag else "N/A"
              results.append({"url": link["href"], "title": title, "description": description})
        time.sleep(random.uniform(2,5))
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return results


def extract_emails_from_results(results, search_engine="Google"):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    extracted_emails = []
    for result in results:
      if "12 de octubre" in result["title"].lower() or "12 de octubre" in result["description"].lower():
        match = re.findall(email_pattern, result["description"])
        if match:
            valid_emails = [email for email in match if "@salud.madrid.es" in email]
            if valid_emails:
              extracted_emails.append(f"{search_engine}: {result['title']}: {', '.join(valid_emails)}")
    return extracted_emails

if __name__ == "__main__":
    query = "doctor medico jefe servicio 12 de octubre hospital email @salud.madrid.es"
    num_results = 100

    print("Starting tests...")

    # Method 1: Google Search with googlesearch-python
    print("\nMethod 1: Google Search with googlesearch-python")
    results_google_lib = search_google_with_lib(query, num_results)
    emails_google_lib = extract_emails_from_results(results_google_lib, "Google with lib")
    if emails_google_lib:
        for email_info in emails_google_lib:
             print(email_info)
    else:
        print("No emails found.")

    # Method 2: Google Search with requests and BeautifulSoup
    print("\nMethod 2: Google Search with requests and BeautifulSoup")
    results_google_custom = search_google_custom(query, num_results)
    emails_google_custom = extract_emails_from_results(results_google_custom, "Google Custom")
    if emails_google_custom:
        for email_info in emails_google_custom:
            print(email_info)
    else:
       print("No emails found.")


    # Method 3: DuckDuckGo Search
    print("\nMethod 3: DuckDuckGo Search")
    results_duckduckgo = search_duckduckgo(query, num_results)
    emails_duckduckgo = extract_emails_from_results(results_duckduckgo, "DuckDuckGo")
    if emails_duckduckgo:
      for email_info in emails_duckduckgo:
          print(email_info)
    else:
      print("No emails found.")

    # Method 4: Bing Search
    print("\nMethod 4: Bing Search")
    results_bing = search_bing(query, num_results)
    emails_bing = extract_emails_from_results(results_bing, "Bing")
    if emails_bing:
        for email_info in emails_bing:
          print(email_info)
    else:
       print("No emails found.")