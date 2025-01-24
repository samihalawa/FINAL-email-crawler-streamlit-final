import sys
import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

url = 'https://www.google.com/search?q=developer+contact+email'
print('Fetching:', url)
response = requests.get(url, headers=headers, verify=False)
print('Status:', response.status_code)
print('Response Headers:')
for key, value in response.headers.items():
    print(f'{key}: {value}')

soup = BeautifulSoup(response.text, 'html.parser')
print('Search Results:')
results = soup.find_all('div', {'class': ['g', 'tF2Cxc']})
print(f'Found {len(results)} results')
for result in results[:2]:
    print('Result:')
    print(result.prettify())