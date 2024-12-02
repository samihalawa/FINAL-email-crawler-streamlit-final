import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from core.database import init_db, get_db
from core.config import settings
from services.search import SearchService
from services.ai import AIService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_search():
    """Test core search functionality"""
    search_service = SearchService()
    
    # Test basic search
    results = await search_service.search(
        query="CEO startup technology spain",
        excluded_domains=["linkedin.com"]
    )
    logger.info(f"Found {len(results)} results")
    
    # Test batch search
    search_terms = [
        "CTO startup madrid",
        "CEO fintech barcelona",
        "founder tech startup valencia"
    ]
    
    all_results = []
    for term in search_terms:
        results = await search_service.search(term)
        all_results.extend(results)
        logger.info(f"Term '{term}': Found {len(results)} results")
    
    return all_results

async def test_ai():
    """Test AI service functionality"""
    ai_service = AIService()
    
    # Test lead enrichment
    test_lead = {
        'email': 'john@example.com',
        'company': 'Tech Corp',
        'position': 'CEO'
    }
    
    enriched = await ai_service.enrich_lead_data(test_lead)
    logger.info(f"Enriched lead data: {enriched}")
    
    return enriched

async def main():
    """Main test function"""
    logger.info("Starting core functionality test")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Test search
    logger.info("Testing search functionality...")
    search_results = await test_search()
    
    # Test AI
    logger.info("Testing AI functionality...")
    ai_results = await test_ai()
    
    logger.info("All tests completed")

if __name__ == "__main__":
    asyncio.run(main()) 