import os
import argparse
from tkinter import *
from tkinter.tix import *
from PIL import ImageTk, Image

class Statistics:
    results = {}
    
    def toggle_image_status(self, panel, image_id):
        bg_color = panel["bg"]
        new_bg_color = "red" if bg_color == "green" else "green"

        panel.config(bg=new_bg_color)

        self.results[image_id]["valid"] = not self.results[image_id]["valid"]

    def main(self):
        tk_root = Tk()
        tk_imgs = []

        tk_root.wm_title("Image statistic generator.")
        frame = Frame(width="1920", height="1080")
        frame.pack()
        swin = ScrolledWindow(frame, width=1920, height=1080)
        swin.pack()
        win = swin.window

        # Load files 
        x = 0
        y = 0
        max_x = 6
        for root, dirs, files in os.walk("/home/lino/Downloads"):
            for file in files:
                if os.path.splitext(file)[1] in [".jpeg", ".jpg", ".png"]:
                    image_id = y * max_x + x
                    self.results[image_id] = {"valid": True, "path": os.path.join(root, file)}

                    image = Image.open(os.path.join(root, file))
                    image = image.resize((300, 300))
                    img = ImageTk.PhotoImage(image)
            
                    panel = Label(win, text=file, image=img)
                    panel.grid(column=x, row=y, padx=5, pady=5)
                    panel.config(bg="green")
                    panel.bind("<Button-1>",lambda _e, panel=panel, image_id=image_id: self.toggle_image_status(panel, image_id))

                    tk_imgs.append((img, panel))

                    x += 1
                    if x == max_x:
                        x = 0
                        y += 1
        
        tk_root.mainloop()

        print(self.results)

if __name__ == "__main__":
    s = Statistics()

    s.main()
