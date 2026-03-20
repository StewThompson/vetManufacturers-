from duckduckgo_search import DDGS
from typing import List, Dict

class ReputationClient:
    def __init__(self):
        self.ddgs = DDGS()

    def search_news(self, manufacturer_name: str, limit: int = 5) -> List[Dict]:
        """
        Searches for news articles related to the manufacturer.
        Returns a list of dictionaries with 'title', 'url', 'body', 'date'.
        """
        query = f"{manufacturer_name} reputation news reviews"
        results = []
        try:
            # First try news search
            news_gen = self.ddgs.news(query, max_results=limit)
            if news_gen:
                for r in news_gen:
                    results.append({
                        'title': r.get('title'),
                        'url': r.get('url'),
                        'body': r.get('body'),
                        'date': r.get('date', 'Unknown'),
                        'source': r.get('source', 'News')
                    })
            
            # If few results, try general search to supplement
            if len(results) < limit:
                remaining = limit - len(results)
                text_gen = self.ddgs.text(query, max_results=remaining * 2) # Get more to filter dupes
                
                existing_urls = {r['url'] for r in results if r.get('url')}
                
                count = 0
                if text_gen:
                    for r in text_gen:
                        url = r.get('href')
                        if url and url not in existing_urls:
                            results.append({
                                'title': r.get('title'),
                                'url': url,
                                'body': r.get('body'),
                                'date': 'Unknown',
                                'source': 'Web Search'
                            })
                            count += 1
                            if count >= remaining:
                                break
                        
            return results[:limit]
        except Exception as e:
            print(f"Error searching for reputation: {e}")
            return []
