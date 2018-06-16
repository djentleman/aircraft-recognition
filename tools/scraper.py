import requests
from bs4 import BeautifulSoup
from settings import (
    aircraft_types
)
import json
import time



# small web scraper that outputs a json object of plane names/photo arrays
def scrape_images():
    base_url = 'https://www.jetphotos.com/'

    # iterate over aircraft types
    aircraft_hash = {}
    for aircraft in aircraft_types:
        print("scraping: " + aircraft + '...')
        page_id = 1
        photo_urls = []
        while True:
            print('Page: ' + str(page_id))
            scrape_url = base_url + 'showphotos.php?aircraft=' + aircraft + ';&page=' + str(page_id)
            print(scrape_url)
            data = requests.get(scrape_url).text
            soup = BeautifulSoup(data, 'html.parser')
            results = [base_url + a['href'] for a in soup.findAll('a', {'class': 'result__photoLink'})]
            photo_urls += results
            if len(results) == 0:
                break
            page_id += 1
            # max 10 pages of photos per plane for now
            if page_id > 10:
                break
            time.sleep(5) # be nice
        aircraft_hash[aircraft] = photo_urls
    output = json.dumps(aircraft_hash)
    open('aircraft_photos.json', 'w+').write(output)
            

if __name__ == '__main__':
    scrape_images()
