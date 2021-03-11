import requests
from bs4 import BeautifulSoup

class Scrape:
    def web_scrape(self):

        URL = 'https://www.bbc.co.uk/news/science-environment-56297996'
        page = requests.get(URL)

        soup = BeautifulSoup(page.content, 'html.parser')