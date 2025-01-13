import requests
import pandas as pd
import logging
from datetime import datetime
import os
from urllib.parse import quote

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InfoJobsAPI:
    def __init__(self):
        self.base_url = "https://api.infojobs.net/api/7/offer"
        self.client_id = os.getenv("INFOJOBS_CLIENT_ID")
        self.client_secret = os.getenv("INFOJOBS_CLIENT_SECRET")
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Please set INFOJOBS_CLIENT_ID and INFOJOBS_CLIENT_SECRET environment variables")
    
    def search_jobs(self, query, max_results=50):
        """Search for jobs using the InfoJobs API"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {self.client_id}:{self.client_secret}'
        }
        
        params = {
            'q': query,
            'maxResults': max_results,
            'order': 'updated-desc',
            'sinceDate': 'ANY'
        }
        
        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            jobs = []
            
            for item in data.get('items', []):
                job = {
                    'title': item.get('title'),
                    'company': item.get('author', {}).get('name'),
                    'location': f"{item.get('city')}, {item.get('province', {}).get('value')}",
                    'salary': f"{item.get('salaryMin', {}).get('value', 'N/A')} - {item.get('salaryMax', {}).get('value', 'N/A')}",
                    'contract_type': item.get('contractType', {}).get('value'),
                    'experience': item.get('experienceMin', {}).get('value'),
                    'url': item.get('link'),
                    'search_term': query,
                    'published_date': item.get('published'),
                    'updated_date': item.get('updated'),
                    'requirements': item.get('requirementMin'),
                    'description': item.get('description')
                }
                jobs.append(job)
                logging.info(f"Found job: {job['title']} at {job['company']}")
            
            return jobs
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error searching InfoJobs API: {str(e)}")
            return []

def main():
    search_terms = [
        "software engineer spain",
        "data scientist barcelona",
        "tech startup madrid",
        "CTO spain startup",
        "developer spain remote"
    ]
    
    try:
        api = InfoJobsAPI()
        all_jobs = []
        
        for term in search_terms:
            logging.info(f"Searching for: {term}")
            jobs = api.search_jobs(term)
            all_jobs.extend(jobs)
        
        if all_jobs:
            df = pd.DataFrame(all_jobs)
            filename = f"infojobs_listings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            logging.info(f"Saved {len(all_jobs)} jobs to {filename}")
            
            print("\nJob Search Results Summary:")
            print(f"Total jobs found: {len(all_jobs)}")
            
            print("\nJobs by search term:")
            print(df['search_term'].value_counts())
            
            print("\nTop companies:")
            print(df['company'].value_counts().head())
            
            print("\nSample Jobs:")
            for _, job in df.head().iterrows():
                print(f"\nTitle: {job['title']}")
                print(f"Company: {job['company']}")
                print(f"Location: {job['location']}")
                print(f"Salary: {job['salary']}")
                print(f"Contract: {job['contract_type']}")
                print(f"Experience: {job['experience']}")
                print(f"URL: {job['url']}")
                print("-" * 50)
        else:
            print("No jobs found.")
            
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("To use the InfoJobs API, you need to:")
        print("1. Register at https://developer.infojobs.net/")
        print("2. Create an application to get your API credentials")
        print("3. Set the following environment variables:")
        print("   export INFOJOBS_CLIENT_ID='your_client_id'")
        print("   export INFOJOBS_CLIENT_SECRET='your_client_secret'")

if __name__ == "__main__":
    main() 