import asyncio
import aiohttp
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Any
from fake_useragent import UserAgent
import re
import json
from aiohttp import ClientTimeout
from aiohttp.client_exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TIMEOUT = ClientTimeout(total=30)
MAX_RETRIES = 3
RETRY_DELAY = 1

async def test_browser_functionality(urls: List[str]) -> Dict[str, Any]:
    """
    Test browser functionality by simulating requests and parsing responses
    """
    if not urls:
        raise ValueError("URL list cannot be empty")
        
    results = {
        "successful_requests": 0,
        "failed_requests": 0,
        "parsed_content": [],
        "errors": [],
        "performance_metrics": {}
    }
    
    ua = UserAgent()
    headers = {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    async with aiohttp.ClientSession(headers=headers, timeout=TIMEOUT) as session:
        for url in urls:
            for attempt in range(MAX_RETRIES):
                try:
                    start_time = asyncio.get_event_loop().time()
                    
                    async with session.get(url, ssl=False) as response:
                        load_time = asyncio.get_event_loop().time() - start_time
                        
                        if response.status == 200:
                            results["successful_requests"] += 1
                            html = await response.text()
                            
                            try:
                                soup = BeautifulSoup(html, 'html.parser')
                                
                                # Rest of the processing code remains the same...
                                # (Previous code for extracting title, emails, forms, etc.)
                                
                            except Exception as parse_error:
                                logger.error(f"HTML parsing error for {url}: {str(parse_error)}")
                                results["errors"].append(f"Parsing error for {url}: {str(parse_error)}")
                                continue
                                
                            break  # Success - exit retry loop
                            
                        else:
                            results["failed_requests"] += 1
                            results["errors"].append(f"HTTP {response.status} for {url}")
                            
                except (ClientError, asyncio.TimeoutError) as e:
                    if attempt == MAX_RETRIES - 1:  # Last attempt
                        results["failed_requests"] += 1
                        results["errors"].append(f"Failed after {MAX_RETRIES} attempts for {url}: {str(e)}")
                        logger.error(f"Failed to process {url} after {MAX_RETRIES} attempts: {str(e)}")
                    else:
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                        
                except Exception as e:
                    results["failed_requests"] += 1
                    results["errors"].append(f"Unexpected error processing {url}: {str(e)}")
                    logger.error(f"Unexpected error for {url}: {str(e)}")
                    break  # Don't retry on unexpected errors
                
    return results

async def main():
    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://httpstat.us/200",
        "https://jsonplaceholder.typicode.com/posts/1",
        "https://news.ycombinator.com"
    ]
    
    try:
        logger.info("Starting browser functionality test...")
        results = await test_browser_functionality(test_urls)
        
        # Detailed analysis and reporting
        logger.info("\n=== Test Results Summary ===")
        logger.info(f"Successful requests: {results['successful_requests']}")
        logger.info(f"Failed requests: {results['failed_requests']}")
        
        if results["parsed_content"]:
            logger.info("\n=== Content Analysis ===")
            for content in results["parsed_content"]:
                logger.info(f"\nURL: {content['url']}")
                logger.info(f"Title: {content['title']}")
                logger.info(f"Emails found: {len(content['emails'])}")
                logger.info(f"Forms found: {len(content['forms'])}")
                logger.info(f"Links found: {len(content['links'])}")
                logger.info(f"Content length: {content['text_length']} characters")
                
                metrics = results["performance_metrics"][content['url']]
                logger.info(f"\nPerformance Metrics:")
                logger.info(f"Load time: {metrics['load_time']:.2f} seconds")
                logger.info(f"Content size: {metrics['content_size']/1024:.2f} KB")
                logger.info(f"Total elements: {metrics['elements_count']}")
        
        if results["errors"]:
            logger.info("\n=== Errors Encountered ===")
            for error in results["errors"]:
                logger.info(f"- {error}")
                
        # Save results
        try:
            with open('browser_test_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save results to file: {str(e)}")
            
    except Exception as e:
        logger.error(f"Critical error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}") 