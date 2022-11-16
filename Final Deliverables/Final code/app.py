from flask import Flask,render_template,request
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename

app=Flask(__name__,template_folder="templates")
model=load_model('disaster.h5')
print("Loaded model from disk")

@app.route('/',methods=['GET'])
def home():
    return render_template("index.html")
@app.route('/home',methods=['GET'])
def back():
    return render_template("index.html")
@app.route('/upload',methods=['GET'])
def index():
     
    cap=cv2.VideoCapture(0)
    H=None
    W= None
    while True:
            (grabbed,frame)= cap.read()

            if not grabbed:
                break
            if W is None or H is None:
                (H,W)= frame.shape[:2]
            output= frame.copy()

            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame,(64,64))
            x=np.expand_dims(frame,axis=0)
            
            result = np.argmax(model.predict(x),axis=-1)
            index =['Cyclone','Earthquake','Flood','Wildfire']
           
            output=str(index[result[0]])
           
            print(result)
            return render_template("output.html",output=output)
            # cv2.putText(output,"activity:{}".format(result),(10,120),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255),1 )
            # cv2.imshow("output",output)
    #         if cv2.waitKey(2) & 0xFF==ord('x'):
    #             break
    # print("[info] cleaning up....")
    # cap.release()
    # cv2.destroyAllWindows()

    # return render_template("output.html",output=result)
if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000,debug=False)        
