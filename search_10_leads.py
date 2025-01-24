import sys
import urllib.parse
sys.path.append('.')
from working_search import google_search, extract_emails

def search_leads():
    search_terms = [
        'developer contact email',
        'freelance developer email',
        'hire developer contact',
        'web developer contact',
        'software engineer email'
    ]
    
    total_leads = 0
    for term in search_terms:
        encoded_term = urllib.parse.quote(term)
        print('Searching for:', term)
        results = google_search(encoded_term, num_results=10)
        
        if results:
            print('Found', len(results), 'pages to check for emails')
            for result in results:
                try:
                    emails = extract_emails(result['url'])
                    if emails:
                        print('Found emails at', result['url'] + ':')
                        for email in emails:
                            print('-', email)
                            total_leads += 1
                except Exception as e:
                    print('Error processing', result['url'] + ':', str(e))
        else:
            print('No results found')

    print('Total leads found:', total_leads)

if __name__ == '__main__':
    print('Starting lead search with URL encoding...')
    search_leads()
    print('Search completed.')