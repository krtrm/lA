import requests
import logging
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Create a News API router
router = APIRouter(prefix="/news", tags=["news"])

# Initialize News API client
NEWS_API_KEY = "YOUR_NEWSAPI_AI_KEY"  # Replace with your NewsAPI.ai API key
NEWSAPI_AI_BASE_URL = "https://api.newsapi.ai/api/v1"

# Legal news search terms
LEGAL_KEYWORDS = [
    "supreme court", "high court", "district court", "tribunal", "judiciary", 
    "legal", "law", "lawyer", "advocate", "judge", "justice", "judicial",
    "constitutional", "legislation", "lawsuit", "litigation", "plaintiff", 
    "defendant", "prosecution", "legal rights", "judgement", "verdict",
    "legal reform", "legal industry", "legal tech", "legal technology",
    "bar council", "legal education", "legal profession", "legal practice",
    "criminal law", "civil law", "corporate law", "family law", "property law",
    "human rights", "constitutional law", "contract law", "tort law", "patent law",
    "copyright law", "intellectual property", "labor law", "employment law",
    "environmental law", "immigration law", "bankruptcy law"
]

@router.get("/")
async def get_news(
    q: Optional[str] = None,
    category: Optional[str] = None,
    country: str = "in",
    language: str = "en",
    page_size: int = 20,
    page: int = 1,
) -> Dict[str, Any]:
    """
    Get law and judicial related news articles
    
    Parameters:
    - q: Keywords or phrases to search for in the news
    - category: Category of news to get (business, entertainment, general, health, science, sports, technology)
    - country: Country of news to get (default: in)
    - language: Language of news to get (default: en)
    - page_size: Number of results to return per page (default: 20, max: 100)
    - page: Page number to get (default: 1)
    """
    try:
        # Validate category
        valid_categories = ["business", "entertainment", "general", "health", "science", "sports", "technology"]
        if category and category.lower() not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            )
        
        # Create query for legal and judicial news
        search_query = q if q else " OR ".join([f'"{keyword}"' for keyword in LEGAL_KEYWORDS[:10]])
        
        # Prepare request parameters for NewsAPI.ai
        params = {
            "apiKey": NEWS_API_KEY,
            "language": language,
            "articlesCount": min(page_size, 100),  # Max 100 as per API limits
            "articlesPage": page,
            "resultType": "articles",
            "sortBy": "date",
            "sortByAsc": False  # Get most recent news first
        }
        
        # Add query parameter
        if search_query:
            params["keyword"] = search_query
        
        # Add category if provided
        if category:
            params["categoryUri"] = category.lower()
            
        # Add country filter if provided
        if country:
            params["sourceLocationUri"] = f"http://en.wikipedia.org/wiki/{country.upper()}"
        
        # Make request to NewsAPI.ai
        response = requests.get(f"{NEWSAPI_AI_BASE_URL}/article/getArticles", params=params)
        
        if response.status_code != 200:
            logger.error(f"NewsAPI.ai error: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"NewsAPI.ai error: {response.text}"
            )
        
        result = response.json()
        
        # Transform the response to match the format expected by the frontend
        articles = []
        if "articles" in result and "results" in result["articles"]:
            for article in result["articles"]["results"]:
                articles.append({
                    "source": {
                        "id": article.get("source", {}).get("uri", ""),
                        "name": article.get("source", {}).get("title", "Unknown Source")
                    },
                    "author": article.get("authors", ["Unknown"])[0] if article.get("authors") else "Unknown",
                    "title": article.get("title", ""),
                    "description": article.get("body", "")[:200] + "..." if article.get("body") else "",
                    "url": article.get("url", ""),
                    "urlToImage": article.get("image", ""),
                    "publishedAt": article.get("dateTime", ""),
                    "content": article.get("body", "")
                })
        
        return {
            "status": "ok",
            "totalResults": result.get("articles", {}).get("totalResults", 0),
            "articles": articles
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch news: {str(e)}"
        )

@router.post("/opinion")
async def post_opinion(
    opinion: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Post a user opinion/article. This is a mock endpoint that just returns the posted opinion
    with some additional metadata.
    
    Parameters:
    - opinion: A dictionary containing the opinion data (title, content, author)
    """
    try:
        required_fields = ["title", "content", "author"]
        for field in required_fields:
            if field not in opinion:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # In a real app, this would save to a database
        # For now, just return the opinion with some metadata
        from datetime import datetime
        
        return {
            "id": "op-" + str(hash(opinion["title"] + str(datetime.now().timestamp())))[:8],
            "title": opinion["title"],
            "content": opinion["content"],
            "author": opinion["author"],
            "publishedAt": datetime.now().isoformat(),
            "source": {"name": "User Opinion"},
            "success": True,
            "message": "Opinion posted successfully"
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error posting opinion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to post opinion: {str(e)}"
        )