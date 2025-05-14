import requests
from typing import Dict, Any
from .base import BaseAgent
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
def extract_body_text_with_selenium(url, min_words=100):
    try:
        options = uc.ChromeOptions()
        options.headless = True
        driver = uc.Chrome(options=options, use_subprocess=False)
        driver.get(url)
        time.sleep(2)  # Wait for page to load

        # Collect paragraphs until at least min_words is reached
        paragraphs = driver.find_elements(By.TAG_NAME, "p")
        texts = []
        word_count = 0
        for p in paragraphs:
            text = p.text.strip()
            if text:
                texts.append(text)
                word_count += len(text.split())
            if word_count >= min_words:
                break
        driver.quit()
        return "\n\n".join(texts) if texts else ""
    except Exception as e:
        try:
            driver.quit()
        except:
            pass
        return f"_Could not extract content: {str(e)}_"

class InternetResearchAgent(BaseAgent):
    """Agent that performs direct internet research using Selenium and undetected_chromedriver."""

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Takes comma-separated keywords, searches the web, and returns structured markdown with only body text.
        Ensures at least 100 words are scraped per keyword.
        """
        context = context or {}
        keywords = [kw.strip() for kw in input_text.split(",") if kw.strip()]
        results_md = "# Internet Research Results\n"

        for keyword in keywords:
            results_md += f"\n## {keyword}\n"
            try:
                # Use Bing search to get result links
                search_url = f"https://www.bing.com/search?q={requests.utils.quote(keyword)}"
                options = uc.ChromeOptions()
                options.headless = True
                driver = uc.Chrome(options=options, use_subprocess=False)
                driver.get(search_url)
                time.sleep(2)  # Wait for search results to load

                # Get top 5 result links
                results = driver.find_elements(By.CSS_SELECTOR, "li.b_algo h2 a")
                links = []
                for result in results:
                    href = result.get_attribute("href")
                    if href:
                        links.append(href)
                    if len(links) >= 5:
                        break
                driver.quit()

                found = False
                collected_texts = []
                total_words = 0
                for link in links:
                    found = True
                    body_text = extract_body_text_with_selenium(link, min_words=100 - total_words)
                    if body_text:
                        collected_texts.append(body_text)
                        total_words += len(body_text.split())
                    if total_words >= 100:
                        break
                if collected_texts:
                    results_md += "\n\n".join(collected_texts) + "\n\n"
                elif not found:
                    results_md += "_No results found or blocked by Bing. Try different keywords._\n"
            except Exception as e:
                try:
                    driver.quit()
                except:
                    pass
                results_md += f"_Error fetching results for '{keyword}': {str(e)}_\n"

        return {
            "output": results_md,
            "agent_type": "internet_research",
            "keywords": keywords
        }