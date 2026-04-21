import cv2
import os
from flask import Flask, request, render_template, send_file
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Defining Flask App
app = Flask(__name__)

nimgs = 100

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):

    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time,Classroom,Course\n')




# get a number of total registered users
def totalreg():
    csv_path = 'Attendance/User.csv'
    if not os.path.exists(csv_path):
        return [], [], 0  # Return empty lists if the file doesn't exist
    df = pd.read_csv(csv_path, names=['Name', 'Roll'])
    return len(df)-1


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    classrooms = df['Classroom']
    courses = df['Course']
    l = len(df)
    return names, rolls, times, classrooms, courses, l


# Add Attendance of a specific user
def add_attendance(name, classroom, course_name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

    csv_path = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('Name,Roll,Time,Classroom,Course\n')

        # Append attendance data
    # with open(csv_path, 'a') as f:
    #     f.write(f'{username},{userid},{current_time},{classroom},{course_name}\n')
    print(df['Roll'])
    print(userid)
    if userid not in df['Roll'].values:
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{classroom},{course_name}')



## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names, rolls = zip(*[user.split('_') for user in userlist])
    return userlist, list(names), list(rolls), len(userlist)


def getalluserslist():
    csv_path = 'Attendance/User.csv'
    if not os.path.exists(csv_path):
        return [], [], 0  # Return empty lists if the file doesn't exist
    df = pd.read_csv(csv_path, names=['Name', 'Roll'])
    names = df['Name'].tolist()
    rolls = df['Roll'].tolist()
    return names, rolls, len(df)


## A function to delete a user folder
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser + '/' + i)
    os.rmdir(duser)


################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    names, rolls, times, classrooms, courses, l = extract_attendance()
    print("testing data ",l)
    return render_template('home.html', names=names, rolls=rolls, times=times, classrooms=classrooms, courses=courses,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2)


## List users page
@app.route('/listusers')
def listusers():
    names, rolls, times, classrooms, courses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, classrooms=classrooms, courses=courses,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main Face Recognition functionality.
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    session_minutes = float(request.args.get('session_time', 0.5))  # Default to 0.5 hours (30 minutes) if not provided
    session_time = int(session_minutes * 60)  # Convert hours to seconds
    print()
    start_time = datetime.now()
    names, rolls, times, classrooms, courses, l = extract_attendance()
    classroom = request.args.get('classroom', 'Unknown')
    course_name = request.args.get('course_name', 'Unknown')

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while (datetime.now() - start_time).seconds < session_time:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person, classroom, course_name)
            cv2.putText(frame, f'{identified_person}', (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Attendance', cv2.WND_PROP_VISIBLE) < 1:
            csv_path = f'Attendance/Attendance-{datetoday}.csv'
            df = pd.read_csv(csv_path)
            df_unique = df.drop_duplicates(subset=['Name'], keep='first')
            df_unique.to_csv(csv_path, index=False)
            break
    cap.release()
    cv2.destroyAllWindows()
    csv_path = f'Attendance/Attendance-{datetoday}.csv'
    df = pd.read_csv(csv_path)
    df_unique = df.drop_duplicates(subset=['Name'], keep='first')
    df_unique.to_csv(csv_path, index=False)
    names, rolls, times, classrooms, courses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, classrooms=classrooms, courses=courses,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2)


# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    with open(f'Attendance/User.csv', 'a') as f:
        f.write(f'\n{newusername},{newuserid}')
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == nimgs * 5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, classrooms, courses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, classrooms=classrooms, courses=courses,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/download_csv', methods=['GET'])
def download_csv():
    # Replace 'your_csv_file.csv' with the actual filename of your CSV file in the 'Attendance' folder
    csv_filename = 'Attendance\Attendance-' + datetoday + '.csv'

    return send_file(csv_filename, as_attachment=True)


@app.route('/download_user_csv', methods=['GET'])
def download_user_csv():
    # Replace 'your_csv_file.csv' with the actual filename of your CSV file in the 'Attendance' folder
    csv_filename = r'Attendance\User.csv'

    return send_file(csv_filename, as_attachment=True)


@app.route('/users')
def users():
    names, rolls, total_users = getalluserslist()
    return render_template('users.html', names=names, rolls=rolls, total_users=total_users)


@app.route('/deleteusernew', methods=['GET'])
def deleteusernew():
    duster = request.args.get('user')

    if not duster:
        return "Error: No user specified", 400  # Return a bad request response


    # Read the CSV file
    csv_path = 'Attendance/User.csv'
    df = pd.read_csv(csv_path)



    # Remove user from CSV
    initial_rows = df.shape[0]
    df = df[df['Name'] != duster]  # Simply match and remove

    # Check if deletion was successful
    if df.shape[0] == initial_rows:
        print(f"User {duster} not found in CSV!")
    else:
        print(f"User {duster} removed successfully.")

    # Save the updated CSV
    df.to_csv(csv_path, index=False, encoding='utf-8')

    # If all faces are deleted, remove the trained model
    if not os.listdir('static/faces/'):
        if os.path.exists('static/face_recognition_model.pkl'):
            os.remove('static/face_recognition_model.pkl')

    # Retrain model
    try:
        train_model()
    except Exception as e:
        print("Training model failed after deletion:", str(e))

    # Reload user list
    names, rolls, total_users = getalluserslist()
    return render_template('users.html', names=names, rolls=rolls, total_users=total_users)
# Our main function which runs the Flask App


@app.route('/deleteuserattendance', methods=['GET'])
def deleteuserattendance():
    duster = request.args.get('user')

    if not duster:
        return "Error: No user specified", 400  # Return a bad request response

    print(f"Trying to delete user: {duster}")  # Debugging output

    # Read the CSV file
    csv_path = f'Attendance/Attendance-{datetoday}.csv'
    df = pd.read_csv(csv_path)

    print("CSV file before deletion:\n", df)  # Debugging

    # Remove user from CSV
    initial_rows = df.shape[0]
    df = df[df['Name'] != duster]  # Simply match and remove

    # Check if deletion was successful
    if df.shape[0] == initial_rows:
        print(f"User {duster} not found in CSV!")
    else:
        print(f"User {duster} removed successfully.")

    # Save the updated CSV
    df.to_csv(csv_path, index=False, encoding='utf-8')

    names, rolls, times, classrooms, courses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, classrooms=classrooms, courses=courses,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2)
# Our main function which runs the Flask App

if __name__ == '__main__':
    app.run(debug=True, port=5001)
