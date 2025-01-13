import asyncio
import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_url(session, url, ua):
    try:
        headers = {
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        async with session.get(url, headers=headers, ssl=False, timeout=30) as response:
            logging.info(f"Testing {url}")
            logging.info(f"Status: {response.status}")
            if response.status == 200:
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                logging.info(f"Title: {soup.title.string if soup.title else 'No title'}")
                logging.info(f"Content length: {len(content)}")
                return True
            return False
    except Exception as e:
        logging.error(f"Error for {url}: {str(e)}")
        return False

async def main():
    urls = [
        "https://www.linkedin.com/jobs/search?keywords=software%20engineer",
        "https://www.glassdoor.com/Job/spain-software-engineer-jobs-SRCH_IL.0,5_IN219",
        "https://www.indeed.es/jobs?q=software%20engineer",
        "https://www.infojobs.net/jobsearch/search-results/list.xhtml?keyword=software%20engineer",
        "https://www.tecnoempleo.com/busqueda-empleo.php?te=software%20engineer",
        "https://www.jobfluent.com/jobs-software%20engineer",
        "https://www.welcometothejungle.com/es/jobs?query=software%20engineer",
        "https://www.talent.com/jobs?k=software%20engineer&l=Spain",
        "https://es.trabajo.org/empleo-software%20engineer"
    ]
    
    ua = UserAgent()
    async with aiohttp.ClientSession() as session:
        tasks = [test_url(session, url, ua) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        working_sites = [url for url, success in zip(urls, results) if success]
        logging.info("\nWorking sites:")
        for site in working_sites:
            logging.info(site)

if __name__ == "__main__":
    asyncio.run(main()) 