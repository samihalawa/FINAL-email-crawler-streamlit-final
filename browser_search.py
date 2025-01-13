import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import logging
from bs4 import BeautifulSoup
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def setup_browser():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    return playwright, browser, context

async def search_infojobs(search_term):
    jobs = []
    playwright, browser, context = await setup_browser()
    
    try:
        page = await context.new_page()
        url = f"https://www.infojobs.net/jobsearch/search-results/list.xhtml?keyword={search_term}"
        
        logging.info(f"Searching InfoJobs for: {search_term}")
        await page.goto(url, wait_until='networkidle')
        
        # Wait for job listings to load
        await page.wait_for_selector('div[data-test="list-item"]', timeout=10000)
        
        # Get all job listings
        job_elements = await page.query_selector_all('div[data-test="list-item"]')
        
        for job_elem in job_elements:
            try:
                # Extract job details using updated selectors
                title = await job_elem.query_selector('h2[data-test="title"] a')
                company = await job_elem.query_selector('a[data-test="company-link"]')
                location = await job_elem.query_selector('span[data-test="location"]')
                salary = await job_elem.query_selector('span[data-test="salary"]')
                contract = await job_elem.query_selector('span[data-test="contract-type"]')
                experience = await job_elem.query_selector('span[data-test="experience"]')
                
                # Get text content and URL
                title_text = await title.text_content() if title else 'N/A'
                title_url = await title.get_attribute('href') if title else None
                company_name = await company.text_content() if company else 'N/A'
                location_text = await location.text_content() if location else 'N/A'
                salary_text = await salary.text_content() if salary else 'N/A'
                contract_text = await contract.text_content() if contract else 'N/A'
                experience_text = await experience.text_content() if experience else 'N/A'
                
                job = {
                    'title': title_text.strip(),
                    'company': company_name.strip(),
                    'location': location_text.strip(),
                    'salary': salary_text.strip(),
                    'contract_type': contract_text.strip(),
                    'experience': experience_text.strip(),
                    'url': f"https://www.infojobs.net{title_url}" if title_url else None,
                    'search_term': search_term
                }
                
                jobs.append(job)
                logging.info(f"Found job: {job['title']} at {job['company']}")
                
            except Exception as e:
                logging.error(f"Error extracting job details: {str(e)}")
                continue
        
    except Exception as e:
        logging.error(f"Error searching InfoJobs: {str(e)}")
    
    finally:
        await context.close()
        await browser.close()
        await playwright.stop()
    
    return jobs

async def main():
    search_terms = [
        "software engineer spain",
        "data scientist barcelona",
        "tech startup madrid",
        "CTO spain startup",
        "developer spain remote"
    ]
    
    all_jobs = []
    for term in search_terms:
        jobs = await search_infojobs(term)
        all_jobs.extend(jobs)
        # Add a delay between searches
        await asyncio.sleep(2)
    
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
            print(f"URL: {job['url']}")
            print("-" * 50)
    else:
        print("No jobs found.")

if __name__ == "__main__":
    asyncio.run(main()) 