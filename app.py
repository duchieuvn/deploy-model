from flask import Flask, render_template, request
import numpy as np
import cv2 

app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def getImage():
    if request.method == "GET":
        return render_template('index.html')
    else:
        if request.files:
            fObj = request.files['image_input']

            #convert string data to numpy array
            file_bytes = np.fromstring(fObj.read(), np.uint8)
            # convert numpy array to image
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

            print(img)
        return f"<h1>img</h1>"   
    


if __name__ == '__main__':
    app.run(debug=True)