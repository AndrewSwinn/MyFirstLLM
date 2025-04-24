import os
import requests
import time
from   PIL import Image
from   io import BytesIO

import threading

import   duckduckgo_search as duck

import matplotlib.pyplot as plt
from   matplotlib.backend_bases import KeyEvent
from   matplotlib.widgets       import TextBox

#from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print('Imports Complete')

# Load the model and processor
#model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

#Load Blip Models
try:
    model = BlipForConditionalGeneration.from_pretrained("blip-captioning-base-model")
    processor = BlipProcessor.from_pretrained("blip-captioning-base-processor", use_fast=True)
except:
    model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    model.save_pretrained("blip-captioning-base-model")
    processor.save_pretrained("blip-captioning-base-processor")



class ImageViewer:
    def __init__(self):

        # Create the initial figure and axis
        self.fig, self.ax = plt.subplot_mosaic("A;B", height_ratios=[1,8], figsize=(8, 8))

        # Connect event handlers for key press
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.create_axes()

        # Start background process
        self.search = False
        thread = threading.Thread(target=self.background_work, daemon=True)
        thread.start()

        # Display the plot
        plt.show()

    def background_work(self):

        while True:
            if self.search:
                results = duck.DDGS().images(keywords=self.query_string, max_results=10)
                self.zquery(results)
                print('Done')
                self.search = False
            time.sleep(0.5)

    def query(self, query):
        print('submit')
        self.query_string = query
        self.search = True


    def zquery(self, results):

        self.index = 0
        self.images_list = []
        self.descriptions_list = []



        for i, result in enumerate(results):
            image_url = result['image']

            try:
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                self.images_list.append(image)

                inputs = processor(image, return_tensors="pt")
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                self.descriptions_list.append(caption)


            except:
                pass




    def update_screen(self):

        self.ax['B'].imshow(self.images_list[self.index])
        self.ax['B'].set_title(self.descriptions_list[self.index])

        self.fig.canvas.draw()






    def create_axes(self):

        self.ax['A'].clear()
        self.ax['A'].axis('off')
        self.ax['B'].clear()
        self.ax['B'].axis('off')


        self.text_box = TextBox(self.ax['A'], "DuckDuckGo Search", textalignment="center", initial='',  color='grey')
        self.text_box.on_submit(self.query)



    def on_key_press(self, event: KeyEvent):
        """Handles key press events to navigate through the images."""
        update = False
        if event.key == 'right':  # Right arrow key
            self.index = self.index + 1 # Go to next image
            if self.index == len(self.images_list): self.index = 0
            update=True


        elif event.key == 'left':  # Left arrow key
            self.index = self.index - 1  # Go to previous image
            if self.index == - 1: self.index = len(self.images_list) - 1
            update=True

        if update: self.update_screen()


# Create and run the viewer
viewer = ImageViewer()
