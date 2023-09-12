import cv2
import face_recognition
from gtts import gTTS
import pygame
import time
import os
import json
import threading

class FaceRecognition:
    def __init__(self, encodings, names):
        self.known_faces = encodings
        self.known_names = names
        self.face_locations = face_recognition.face_locations
        self.face_encodings = face_recognition.face_encodings
        self.compare_faces = face_recognition.compare_faces
        self.last_spoken_time = {}
        self.faces_folder = 'faces'
        self.face_data_file = 'face_data.json'
        self.last_unknown_face_time = 0  # هذا وقت يحسب متى اخر مره طلع فيا وجه عشانه ماينطق الاسم الى بعد 30 ثانيه من ظهور الوجه
        self.recognized_buffer_time = 3  # كم ثانيه لازم يطلع الوجه الي مو معرف عشان يطلب منه تعريفه
        self.recognized_names_buffer = {}  # يحفظ الوجوه قبل حفظها في json
        self.is_asking_user = False
        self.last_asked_time = 0 # هذا عشان يحسب متى اخر مره سال



    def load_face_data(self):
        if os.path.exists(self.face_data_file):
            with open(self.face_data_file, 'r') as file:
                data = json.load(file)
            return data
        else:
            return {}

    def save_face_data(self, data):
        with open(self.face_data_file, 'w') as file:
            json.dump(data, file)

    def add_new_face(self, face_encoding, name):
        data = self.load_face_data()
        if name not in data:  # تحقق من أن الوجه غير معروف قبل إضافته
            data[name] = face_encoding
            self.save_face_data(data)

    def is_face_known(self, face_encoding):
        data = self.load_face_data()
        for name, encoding in data.items():
            if face_recognition.compare_faces([encoding], face_encoding, tolerance=0.45)[0]:
                return name
        return "Unknown"

    def is_face_unknown(self, face_encoding):
        # كم ثانيه لازم يطلع فيها الوجه عشان يطلب منه تعريفه
        allowed_time_diff = 3

        # الفرق  بين الكاميرا الحاليه وآخر مرة تم فيها ظهور وجه غير معروف
        current_time = time.time()
        time_diff = current_time - self.last_unknown_face_time

        # إذا كان الفرق الزمني أكبر من المدة المسموح بها، فإن الوجه غير معروف
        if time_diff >= allowed_time_diff:
            data = self.load_face_data()
            for name, encoding in data.items():
                if face_recognition.compare_faces([encoding], face_encoding, tolerance=0.45)[0]:
                    return False
            return True
        else:
            return False


    def ask_user_to_define_face(self):
        if not self.is_asking_user and time.time() - self.last_asked_time >= 20:
            self.is_asking_user = True
            print("وجه غير معروف تم التقاطه. الرجاء الإجابة على السؤال التالي:")
            print("هل ترغب في تعريف الوجه الجديد؟ (نعم/لا)")
            user_response = input().strip().lower()
            self.is_asking_user = False
            self.last_asked_time = time.time()
            return user_response
        else:
            return "لا"

    def handle_unknown_face(self, unknown_face_encoding):
        if self.ask_user_to_define_face() == "نعم":
            print("الرجاء إدخال اسم الشخص:")
            name = input()
            if name and name != "نعم":
                self.add_new_face(unknown_face_encoding.tolist(), name)
                print("تمت إضافة الوجه بنجاح.")
        self.unknown_face_detected = False

    def run_recognition(self):
        cap = cv2.VideoCapture(0)
        pygame.mixer.init()

        recognized_faces = {}
        self.unknown_face_detected = False
        self.unknown_face_start_time = 0

        while True:
            ret, frame = cap.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = self.face_locations(rgb_small_frame)
            face_encodings = self.face_encodings(rgb_small_frame, face_locations)

            recognized_names = ["Unknown"] * len(face_locations)
            recognized_index = -1

            for face_encoding in face_encodings:
                recognized_index += 1
                if self.is_face_unknown(face_encoding):
                    if not self.unknown_face_detected:
                        self.unknown_face_detected = True
                        self.unknown_face_start_time = time.time()
                        # استدعاء الوظيفة المستقلة لطرح السؤال للمستخدم وتحديد unknown_face_encoding
                        threading.Thread(target=self.handle_unknown_face, args=(face_encoding,)).start()

                else:
                    recognized_name = self.is_face_known(face_encoding)
                    recognized_names[recognized_index] = recognized_name

                    if recognized_name != "Unknown":
                        self.recognized_names_buffer[recognized_index] = recognized_name

            if self.unknown_face_detected:
                current_time = time.time()
                elapsed_time = current_time - self.unknown_face_start_time
                if elapsed_time >= self.recognized_buffer_time:
                    self.unknown_face_detected = False

            for (top, right, bottom, left), recognized_name in zip(face_locations, recognized_names):
                if recognized_name == "Unknown":
                    continue

                current_time = time.time()
                if recognized_name not in self.last_spoken_time or current_time - self.last_spoken_time[recognized_name] >= 30:
                    # إعادة تشغيل التنبيه الصوتي وتحديث وقت التنبيه الأخير
                    if recognized_name == "messi":
                        pygame.mixer.music.load("messi.mp3")
                        pygame.mixer.music.play()
                    elif recognized_name == "khlid_ad":
                        pygame.mixer.music.load("khlid_ad.mp3")
                        pygame.mixer.music.play()
                    else:
                        self.speak_name(recognized_name)
                    self.last_spoken_time[recognized_name] = current_time

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, recognized_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            for i, name in enumerate(recognized_names):
                cv2.putText(frame, name, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Face -  Procject - SAQR', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    def speak_name(self, recognized_name):
        text_val = recognized_name
        language = 'en'
        obj = gTTS(text=text_val, lang=language, slow=False)
        obj.save("exam.mp3")
        pygame.mixer.music.load("exam.mp3")
        pygame.mixer.music.play()



# تهيئة كائن FaceRecognition مع الترميزات والأسماء
face_recognition = FaceRecognition([], [])
face_recognition.run_recognition()
