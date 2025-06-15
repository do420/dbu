import requests
from typing import Dict, Any
from .base import BaseAgent
import asyncio
import time
import logging
from bs4 import BeautifulSoup
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_with_requests(url, min_words=50):
    """Fast text extraction using requests and BeautifulSoup"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements (eski mantık korundu)
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text from paragraphs
        paragraphs = soup.find_all('p')
        texts = []
        word_count = 0
        
        for p in paragraphs[:15]:  # Limit paragraphs
            text = p.get_text().strip()
            if text and len(text) > 20:
                texts.append(text)
                word_count += len(text.split())
                if word_count >= min_words:
                    break
        
        return "\n\n".join(texts) if texts else ""
        
    except Exception as e:
        logger.warning(f"Failed to extract from {url}: {str(e)}")
        return ""

def search_bing_fast(keyword):
    """Fast Bing search using requests"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
           
        }
        
        search_url = f"https://www.bing.com/search?q={urllib.parse.quote(keyword)}"
        response = requests.get(search_url, headers=headers, timeout=8)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find search result links (eski Selenium mantığı korundu)
        links = []
        result_links = soup.find_all('a', href=True)
        
        for link in result_links:
            href = link.get('href')
            # Eski filtreleme mantığı: microsoft.com, bing.com hariç
            if href and href.startswith('http') and not any(x in href for x in ['microsoft.com', 'bing.com', 'msn.com']):
                links.append(href)
                if len(links) >= 3:  # Eski mantık: top 3 result
                    break
        
        return links
        
    except Exception as e:
        logger.error(f"Bing search failed for '{keyword}': {str(e)}")
        return []

class InternetResearchAgent(BaseAgent):
    """Fast internet research agent using requests and BeautifulSoup."""

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Fast internet research with requests instead of Selenium.
        """
        try:
            logger.info(f"Starting fast internet research for: {input_text}")
            context = context or {}
            keywords = [kw.strip() for kw in input_text.split(",") if kw.strip()]
            results_md = "# Internet Research Results\n"

            # Limit to 3 keywords max
            keywords = keywords[:3]

            for i, keyword in enumerate(keywords):
                results_md += f"\n## {keyword}\n"
                try:
                    # Use thread executor for fast parallel processing
                    search_task = asyncio.get_event_loop().run_in_executor(
                        None, self._search_keyword_fast, keyword
                    )
                    search_result = await asyncio.wait_for(search_task, timeout=15)  # 15 second max per keyword
                    results_md += search_result
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout for keyword: {keyword}")
                    results_md += f"_Search timed out for '{keyword}'. Please try a more specific search._\n"
                except Exception as e:
                    logger.error(f"Error processing keyword '{keyword}': {str(e)}")
                    results_md += f"_Error searching for '{keyword}': Connection failed_\n"

            logger.info("Internet research completed successfully")
            return {
                "output": results_md,
                "agent_type": "internet_research", 
                "keywords": keywords,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"InternetResearchAgent process error: {str(e)}")
            return {
                "output": f"# Internet Research Error\n\nAn error occurred: {str(e)}",
                "agent_type": "internet_research",
                "keywords": [],
                "status": "error"
            }
    
    def _search_keyword_fast(self, keyword: str) -> str:
        """Perform very fast search using requests only."""
        try:
            logger.info(f"Searching for: {keyword}")
            
            # Get search results links
            links = search_bing_fast(keyword)
            
            if not links:
                return "_No search results found_\n"

            # Extract content from first working link
            for link in links:
                try:
                    content = extract_text_with_requests(link, min_words=30)
                    if content and len(content.strip()) > 50:
                        return f"{content}\n\n"
                except Exception as e:
                    logger.warning(f"Failed to extract from {link}: {str(e)}")
                    continue
                
            return "_No content could be extracted from search results_\n"
            
        except Exception as e:
            logger.error(f"Search error for '{keyword}': {str(e)}")
            return f"_Search failed: {str(e)}_\n"