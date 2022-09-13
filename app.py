from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from image_captioning.caption_generator import model_captioning


app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def getImage():
    if request.method == "GET":
        return render_template('index.html')
    else:
        if request.files:
            f = request.files['image_input']
            save_path = 'images/'+f.filename
            f.save(save_path)

            captions = model_captioning('images')
            # img = plt.imread(save_path)
            # print(img)
            print(captions)
        return f"<h1>img</h1>"   
    


if __name__ == '__main__':
    print('run')
    app.run(debug=True)