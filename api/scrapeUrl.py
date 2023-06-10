from bs4 import BeautifulSoup
import requests
import os


def scrape_url(url):
    base_url = "http://api.scraperapi.com"
    params = {
        "api_key": os.environ['SCRAPER_API_KEY'],
        "url": url
    }
    
    
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        content = soup.get_text()
        content = content.strip()
        content = ' '.join(content.split())
       
        return content
    else:
        print(f"Request failed with status code {response.status_code}")
