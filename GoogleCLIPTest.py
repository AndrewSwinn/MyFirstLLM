import os
import requests
from   PIL import Image
from   io import BytesIO

import   duckduckgo_search as duck

import matplotlib.pyplot as plt
from   matplotlib.backend_bases import KeyEvent
from   matplotlib.widgets       import TextBox

from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

        # Display the plot
        plt.show()


    def query(self, query):

        self.index = 0
        self.images_list = []

        results = duck.DDGS().images(keywords=query, max_results=10)

        for i, result in enumerate(results):
            image_url = result['image']

            try:
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                self.images_list.append(image)
            except:
                pass

        update_screen()

        self.ax['B'].imshow(self.images_list[self.index])

        self.fig.canvas.draw()

    def update_screen(self):
        ,         self.ax['B'].imshow(self.images_list[self.index])
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
        if event.key == 'right':  # Right arrow key
            self.index = self.index + 1 # Go to next image
            if self.index == len(self.images_list): self.index = 0


        elif event.key == 'left':  # Left arrow key
            self.index = self.index - 1  # Go to previous image
            if self.index == - 1: self.index = len(self.images_list) - 1

        elif event.key == 'up':  # Up arrow key
            self.class_id = self.class_id - 1  # Go to next class
            if self.class_id == 0: self.class_id = 200
            self.images_list = self.get_indexes(self.class_id)
            self.index = 0

        elif event.key == 'down':  # Down arrow key
            self.class_id = self.class_id + 1  # Go to previous class
            if self.class_id == 201: self.class_id = 1
            self.images_list = self.get_indexes(self.class_id)
            self.index = 0

        self.ax['B'].imshow(self.images_list[self.index])
        # Redraw the canvas
        self.fig.canvas.draw()


# Create and run the viewer
viewer = ImageViewer()
