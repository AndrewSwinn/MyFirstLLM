import os
import requests
import time
import pickle
import argparse

from   collections import deque
from   PIL import Image, ImageChops
from   io import BytesIO
from   transformers import BlipProcessor, BlipForConditionalGeneration
import   duckduckgo_search as duck

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", type=str, help="Root query")




os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



class node:

    next_node_id = 0

    def __init__(self, depth, parent_id, text, image_url, image):
        self.depth     = depth
        self.id        = node.next_node_id
        self.parent_id = parent_id
        self.text      = text
        self.image_url = image_url
        self.image     = image

        node.next_node_id += 1

    def __eq__(self, other):

        equals = False

        if self.image_url == other.image_url: equals=True

        elif self.image.size == other.image.size and ImageChops.difference(self.image, other.image).getbbox() is None: equals=True

        return equals

    def __str__(self):
        return str(self.depth) + ' ' + str(self.id) +' ' + self.text + ' ' + self.image_url

    def explore(self):
        unexplored = True
        while unexplored:
            try:
                children   = []
                query = duck.DDGS().images(keywords=self.text, max_results=branching)
                for i, hit in enumerate(query):
                    image_url = hit['image']
                    response  = requests.get(image_url, timeout=5.00)
                    response.raise_for_status()
                    image     = Image.open(BytesIO(response.content)).convert('RGB')
                    inputs    = processor(image, return_tensors="pt")
                    out       = model.generate(**inputs)
                    caption    = processor.decode(out[0], skip_special_tokens=True)

                    child = node(self.depth + 1, self.id, caption, image_url, image)

                    children.append(child)
                    unexplored = False
            except:
                if not unexplored:
                    pass
                else:
                    unexplored = True
                    time.sleep(5)

        return children

if __name__ == "__main__":

    args  = parser.parse_args()
    query = args.query

    #Load Blip Models
    try:
        model = BlipForConditionalGeneration.from_pretrained("blip-captioning-base-model")
        processor = BlipProcessor.from_pretrained("blip-captioning-base-processor", use_fast=True)
    except:
        model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        model.save_pretrained("blip-captioning-base-model")
        processor.save_pretrained("blip-captioning-base-processor")

    max_depth, branching = 10,5
    visited = []
    results = []

    root = node(0,  -1, query, ' ', None)

    stack = deque()
    stack.append(root)

    while stack:
        print(len(stack) )
        test_node = stack.popleft()
        if test_node.depth < max_depth:
            leaves = test_node.explore()
            for leaf in leaves:
                if leaf not in visited:
                    stack.append(leaf)
                    visited.append(leaf)

            with open(query + '.pkl', 'wb') as file:
                pickle.dump(visited, file)
