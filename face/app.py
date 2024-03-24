import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier      #scikit-learn
import pandas as pd
import joblib

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")  #date fullMonthName year

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#try:
#    cap = cv2.VideoCapture(1)
#except:
cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:  #creates excel file "Attendance-{datetoday}
        f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    if img != []:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #COLOR_BGR2GRAY - BGR is converted to GRAY
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)  #detectMultiScale function is used to detect the faces
        #This function will return a rectangle with coordinates(x,y,w,h) around the detected face.
        # (imputImage-gray, scaleFactor-1.3, minNeighbors-5)
        #scaleFactor specifies how much the image size is reduced with each scale.
        #minNeighbours specifies how many neighbours each candidate rectangle should have to retain it.
        return face_points
    else:
        return []


#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')    #Joblib is a set of tools to provide lightweight pipelining in Python
    #Pickle can be used to serialize Python object structures, which refers to the process of converting an object in the memory to a byte stream that can be stored as a binary file on disk.
    return model.predict(facearray) #The predict() function is used to predict the values based on the previous data behaviors and thus by fitting that data to the model


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')   #returns a list containing the names of the entries in 'static/faces' directory
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):  #returns length from the list of static -> faces
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            #imread() Loads an image from a file.the function returns an empty matrix
            resized_face = cv2.resize(img, (50, 50))    #resizes width and height to (50,50)
            faces.append(resized_face.ravel())  #ravel() returns 1D array of resized_face
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)   # KNeighborsClassifier looks for the 5 nearest neighbors.
    knn.fit(faces, labels)  #
    joblib.dump(knn, 'static/face_recognition_model.pkl')
    #joblib is faster in saving/loading large NumPy arrays, whereas pickle is faster with large collections of Python objects
    #dump - used to write any object to the binary file
    #load - used to read object from the binary file.


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]   #a
    userid = name.split('_')[1] #1
    current_time = datetime.now().strftime("%H:%M:%S")  #hour:minute:second

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')  #read_csv() takes to 'Attendance/Attendance-{datetoday}.csv' and reads the data into a Pandas DataFrame object(like spreadsheet)
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        if extract_faces(frame) != ():
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        #if cv2.waitKey(1) == 27:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


#### This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        #if cv2.waitKey(1) == 27:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)

