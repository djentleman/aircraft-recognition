import requests
import json
import time
import os

# hacky tool to unpack images from the source data

def load_dataset():
    data = json.loads(open('../data/aircraft_photos.json', 'r+').read())
    return data

def image_from_url(name, idx, url):
    base_html = requests.get(url).text
    # awful awful hack
    image_url = base_html.split('<img class="large-photo__img" src="')[1].split('"')[0]
    image_response = requests.get(image_url, stream=True)
    if image_response.status_code == 200:
        with open('../data/' + name + '/image_' + str(idx) + '.jpg', 'wb') as f:
            for chunk in image_response.iter_content(1024):
                f.write(chunk)
    

def unpack():
    dataset = load_dataset()
    for aircraft in dataset.keys():
        formatted_name = aircraft.lower().replace(' ', '_').replace('/', '_')
        print(formatted_name)
        if not os.path.exists('../data/' + formatted_name):
            os.makedirs('../data/' + formatted_name)
        for idx in range(len(dataset[aircraft])):
            print(idx)
            image_from_url(formatted_name, idx+1, dataset[aircraft][idx])
            time.sleep(1) # be nice

if __name__ == '__main__':
    unpack()
