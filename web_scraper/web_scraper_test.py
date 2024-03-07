from web_scraper.scraper import ScrapeSite

scraper = ScrapeSite(base_url='https://python.langchain.com/docs/get_started/introduction', max_depth=5, base_folder='scraped_content_v1')
scraper.scrape()
