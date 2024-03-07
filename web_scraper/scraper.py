# !pip install requests beautifulsoup4

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import re

class ScrapeSite:
    def __init__(self, base_url, max_depth, base_folder):
        self.base_url = base_url
        self.max_depth = max_depth
        self.base_folder = base_folder

    def scrape(self):
        self._scrape_site(self.base_url, self.max_depth, self.base_folder)

    def _scrape_site(self, url, max_depth, base_folder, current_depth=0, visited=None):
        if visited is None:
            visited = set()

        if current_depth > max_depth or url in visited:
            return

        visited.add(url)
        response = requests.get(url)
        if response.status_code != 200:
            return

        soup = BeautifulSoup(response.text, 'html.parser')

        folder_structure = self._create_folder_structure(base_folder, urlparse(url).path)
        filename = self._sanitize_filename(urlparse(url).path.strip('/').split('/')[-1] or 'index') + '.txt'
        filepath = os.path.join(folder_structure, filename)

        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(f"URL: {url}\n\n")
            main_content = soup.find('div', class_='markdown')
            if main_content:
                for tag in main_content.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'pre']):
                    if tag.name.startswith('h'):
                        file.write(f"\n{tag.get_text()}\n\n")
                    elif tag.name in ['ul', 'ol']:
                        file.write('\n')
                        for li in tag.find_all('li'):
                            file.write(f"- {li.get_text()}\n")
                        file.write('\n')
                    elif tag.name == 'pre':
                        file.write(f"\n```python\n{tag.get_text()}\n```\n\n")
                    else:
                        file.write(f"{tag.get_text()}\n")

        for link in soup.find_all('a', href=True):
            next_url = urljoin(url, link['href'])
            if urlparse(next_url).scheme:
                self._scrape_site(next_url, max_depth, base_folder, current_depth + 1, visited)

    def _sanitize_filename(self, filename):
        filename = re.sub(r'[^\w\s-]', '', filename)
        filename = re.sub(r'\s+', '_', filename)
        return filename[:50]

    def _create_folder_structure(self, base_folder, url_path):
        folder_structure = os.path.join(base_folder, *url_path.strip('/').split('/'))
        os.makedirs(folder_structure, exist_ok=True)
        return folder_structure
