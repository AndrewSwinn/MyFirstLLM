import os
import requests
import time
import pickle
from   collections import deque
from   PIL import Image
from   io import BytesIO

import   duckduckgo_search as duck

from transformers import BlipProcessor, BlipForConditionalGeneration

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#Load Blip Models
try:
    model = BlipForConditionalGeneration.from_pretrained("blip-captioning-base-model")
    processor = BlipProcessor.from_pretrained("blip-captioning-base-processor", use_fast=True)
except:
    model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    model.save_pretrained("blip-captioning-base-model")
    processor.save_pretrained("blip-captioning-base-processor")

max_depth, branching = 10,10
visited = set()
results = []
stack = deque()
stack.append((0, 'dog', None))
last_time = time.time()
duck_wait = 1

while stack:

    (depth, text, image_url) = stack.pop()  # pop from the end (LIFO)
    if depth < max_depth and text not in visited:
        print(depth, text)
        visited.add(text)
        time.sleep(max(duck_wait - (time.time() - last_time), 0))
        results = duck.DDGS().images(keywords=text, max_results=20)
        last_time = time.time()
        count = 0
        for i, result in enumerate(results):
            if count >= branching: break
            try:
                image_url = result['image']
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                inputs = processor(image, return_tensors="pt")
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                stack.append((depth + 1, caption, image_url))
                results.append((depth + 1, caption, image_url, image))
                count += 1
            except:
                pass
with open('file.pkl', 'wb') as file:
    pickle.dump(results, file)
