import face_recognition
import os, sys
import cv2
import numpy as np
import math
import datetime

# Helper
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:    #
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        img_del = 0
        for image in os.listdir('faces'):
            try:
                face_image = face_recognition.load_image_file(f"faces/{image}")
                face_encoding = face_recognition.face_encodings(face_image)[0]

                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)
                #print(self.known_face_names)
                print(image)
               # print(f" {len(self.known_face_encodings)}  bilder lästes in.")

            except IndexError:
                #os.remove(f"faces/{image}")
                os.rename(f"faces/{image}", f"removedfaces/{image}")
                print("Fil togs togs bort!")
                img_del =+1
                os.listdir =+1

        print(f" {len(self.known_face_encodings)}  bilder lästes in.")
        print(f" {img_del}  bilder togs bort.")



    def run_recognition(self):
        print(f"__name__:{__name__}")
        global name, confidence
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                #rgb_small_frame = small_frame[:, :, ::-1]
                # fix crash: https://stackoverflow.com/questions/75926662/face-recognition-problem-with-face-encodings-function
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                #print(f"fl:{self.face_locations}")
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    # Calculate the shortest distance to face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                   # self.face_names.append(f'{name} ({confidence})')
                    self.face_names.append(f'{name}')

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                if name == "Unknown":
                #if name == "Unknown (???)":
                    new_image = frame[top - 100:bottom + 100, left-100:right + 100]
                    now = datetime.datetime.now()
                    img_name = "New " + now.strftime("%Y-%m-%d_%H-%M-%S.png")

                    try:
                       # cv2.imwrite("unknown/" + img_name, new_image)
                        cv2.imwrite("faces/" + img_name, new_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        print(f"Bilden har sparats.")

                        try:

                            new_face_image = face_recognition.load_image_file(f"faces/{img_name}")
                            face_encoding = face_recognition.face_encodings(new_face_image)[0]

                            self.known_face_encodings.append(face_encoding)
                            self.known_face_names.append(img_name)

                            print("Bilden har lagts till.")

                        except  IndexError:
                        #except OSError:
                            print("Bilden kunde inte läggas till.")

                    except cv2.error:
                        print(f"kunde inte spara {img_name} {new_image}")

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                    cv2.putText(frame, confidence, (left + 10, bottom - 5), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                                (255, 255, 255), 1)


             # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()

