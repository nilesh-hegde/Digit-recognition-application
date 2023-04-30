import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import Task1A_E
import cv2

# Load the trained model
model = Task1A_E.MyModel()
model.load_weights('DNNweights').expect_partial()

# Create a canvas to draw on
canvas_width = 200
canvas_height = 200
canvas_color = 'white'


'''
This class is created to implement the canvas on which digits are traced
'''
class PaintApp:

    def __init__(self, master):
        self.master = master
        master.title('Digit Recognition')
        
        # Create the canvas
        self.canvas = tk.Canvas(master, width=canvas_width, height=canvas_height, bg=canvas_color)
        self.canvas.grid(row=0, column=0, padx=5, pady=5)
        self.canvas.bind('<B1-Motion>', self.draw)
        
        # Create the predict button
        self.predict_button = tk.Button(master, text='Predict', command=self.predict)
        self.predict_button.grid(row=1, column=0, padx=5, pady=5)
        
        # Create the clear button
        self.clear_button = tk.Button(master, text='Clear', command=self.clear)
        self.clear_button.grid(row=2, column=0, padx=5, pady=5)
        
        # Create the label for the predicted digit
        self.prediction_label = tk.Label(master, text='')
        self.prediction_label.grid(row=3, column=0, padx=5, pady=5)
        
        # Create the image object to draw on
        self.img = Image.new('L', (canvas_width, canvas_height), canvas_color)
        self.draw = ImageDraw.Draw(self.img)

    '''
    Arguments - Mouse event
    Return - None
    Description - This function helps you draw digit on canvas
    '''
    def draw(self, event):
        x, y = event.x, event.y
        r = 5
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        self.draw.ellipse((x-r, y-r, x+r, y+r), fill='black')
        return

    '''
    Arguments - None
    Return - None
    Description - This function clears the canvas
    '''
    def clear(self):
        self.canvas.delete('all')
        self.img = Image.new('L', (canvas_width, canvas_height), canvas_color)
        self.draw = ImageDraw.Draw(self.img)
        self.prediction_label.config(text='')
        return

    '''
    Arguments - None
    Return - None
    Description - This function predicts the digit drawn in canvas
    '''
    def predict(self):
        # Process the image drawn
        np_array = np.array(self.img)
        image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        inverted = 255 - resized
        normalized = inverted / 255.0
        normalized = np.reshape(normalized, (1, 28, 28, 1))
        
        # Predict the digit
        pred = model.predict(normalized)
        digit = np.argmax(pred)
        
        # Update the label with the predicted digit
        self.prediction_label.config(text='Predicted digit: ' + str(digit))
        return

'''
Arguments - None
Return - None
Description - This function starts execution and implements the extension by calling the canvas class
'''
def main():
    root = tk.Tk()
    PaintApp(root)
    root.mainloop()
    return
    
if __name__ == "__main__":
   main()
