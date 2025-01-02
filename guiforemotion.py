import tkinter as tk
from tkinter import *
import tkinter.filedialog as fd
from keras.models import model_from_json, Sequential
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import dlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

final_data = []

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def create_graph():
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv('report.csv')

    # Set the row index as a proxy for time
    df.set_index(pd.Index(range(len(df))), inplace=True)

    # Set the start and end index values of the 2-minute interval to plot
    start_index = 0
    end_index = 1000  # Assuming 60 frames per second for 2 minutes of video

    # Select a slice of the data between the start and end index values
    subset = df.loc[start_index:end_index]

    # Calculate the percentage of each emotion value in the selected slice of data
    emotions = df['Emotion']
    unique_set = set(emotions)

    # Convert the set back to an array
    emotions = list(unique_set)

    percentages = [df['Emotion'].value_counts(normalize=True)[e] * 100 for e in emotions]

    # Create a bar chart of the percentage of emotion values
    fig1 = Figure(figsize=(6, 6), dpi=100)
    ax1 = fig1.add_subplot(111)
    ax1.bar(emotions, percentages)
    ax1.set_xlabel('Emotions')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Emotions in 2-Minute Interval')

    # Create a pie chart of emotion counts
    fig2 = Figure(figsize=(6, 6), dpi=100)
    ax2 = fig2.add_subplot(111)
    df['Emotion'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_ylabel('')
    ax2.set_title('Relative Proportion of Emotion Values')

    # Create a tkinter window and add the Matplotlib figures
    root = tk.Tk()
    root.title("Graphs")

    # Add the first figure
    canvas1 = FigureCanvasTkAgg(fig1, master=root)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add the second figure
    canvas2 = FigureCanvasTkAgg(fig2, master=root)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    root.mainloop()

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def show_Detect_button():
    detect_b = Button(top, text="Show Report", command=lambda: create_graph(), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground="white", font=("arial", 10, "bold"))
    detect_b.place(relx=0.79, rely=0.46)

model = FacialExpressionModel("emotion_model (1).json", "face_emotion_model.h5")
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

top = tk.Tk()
top.geometry("800x600")
top.title("Emotion Detection")
top.configure(background="#CDCDCD")

label1 = Label(top, background="#CDCDCD", font=("arial", 15, "bold"))
sign_image = Label(top)

def Detect(faces,gray_image):
    # global Label_packed

    # image = cv2.imread(file_path)
    # gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray_image)
    try:
        # print(face)
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            roi = cv2.resize(fc,(48,48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
            # print("Predicted Emotion is:" + pred)
            final_data.append([pred])

        return pred
            # label1.configure(foreground="#011368",text=pred)
    except:
        # label1.configure(foreground="#011368",text="Unable to Detect") 
        pass

# Define function to preprocess each frame
def preprocess_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Crop the image to only include the face
        face = frame[y:y+h, x:x+w]
        # Align the face using facial landmarks
        # TODO: add code for facial landmark alignment
        # face = align_face(face)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Normalize lighting conditions
        face = cv2.equalizeHist(face)
        # TODO: add code for emotion prediction using your trained model
        emotion_label = Detect(faces,gray)
        print(emotion_label)
        # Draw predicted emotion label on frame
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return frame

def upload_image():
    try:
        file_path = fd.askopenfilename()
        videoCapture(file_path)
        # uploaded = Image.open(file_path)
        # uploaded.thumbnail(((top.winfo_width()/2.3),(top.winfo_height()/2.3)))
        # # im = ImageTk.PhotoImage(uploaded)
        # im = ImageTk.PhotoImage(uploaded.resize((200, 200)))


        # sign_image.configure(image=im)
        # sign_image.image = im
        # label1.configure(text="")
        # show_Detect_button(file_path)
    except:
        pass

def videoCapture(filepath):
    cap = cv2.VideoCapture(filepath)
    # Loop over video frames
    while cap.isOpened():
        # Read frame from video capture
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
    
        # Print the fps
        print('fps:', fps)


        # If end of video, break loop
        if not ret:
            break

        # Preprocess frame
        processed_frame = preprocess_frame(frame)

        # Display processed frame
        cv2.imshow('Emotion Analysis', processed_frame)

        # If 'q' key is pressed, break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close window
    cap.release()
    cv2.destroyAllWindows()
    import csv

    # take input from userq
    # lecture_id = input("Enter the lecture ID: ")
    # student_id = input("Enter the student ID: ")
    def handle_button_click():
    # Retrieve the values entered in the two input fields
        input1_value = input1.get()
        input2_value = input2.get()
        
        # Do something with the input values
        print(f"Input 1 value: {input1_value}")
        print(f"Input 2 value: {input2_value}")

            # create a list of emotion predictions (for example purposes)
        pred = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

        # write data to csv file
        with open('report.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            # write header row if file is empty
            if file.tell() == 0:
                writer.writerow(['Lecture ID', 'Student ID', 'Emotion'])
            # write data rows
            for emotion in final_data:
                writer.writerow([input1_value, input2_value, emotion[0]])


    input1_label = tk.Label(top, text="Enter the lecture ID:")
    input1_label.place(relx=0.29, rely=0.26)
    input1 = tk.Entry(top)
    input1.place(relx=0.44, rely=0.26)

    input2_label = tk.Label(top, text="Enter the student ID:")
    input2_label.place(relx=0.29, rely=0.36)
    input2 = tk.Entry(top)
    input2.place(relx=0.44, rely=0.36)

    button = tk.Button(top, text="Submit", command=handle_button_click)
    button.place(relx=0.44, rely=0.46)

    show_Detect_button()

def start_webcam():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = preprocess_frame(frame)
        cv2.imshow('Real-Time Emotion Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

upload = Button(top, text="Upload Video", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground="white", font=("arial", 25, "bold"))
upload.pack(side="bottom", pady=20)

webcam = Button(top, text="Real-Time Capture", command=start_webcam, padx=10, pady=5)
webcam.configure(background="#364156", foreground="white", font=("arial", 25, "bold"))
webcam.pack(side="bottom", pady=20)

sign_image.pack(side="bottom", expand="True")
label1.pack(side="bottom", expand="True")
heading = Label(top, text="Emotion Detector", pady=20, font=("arial", 25, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()
top.mainloop()
