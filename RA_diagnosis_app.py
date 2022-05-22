# Import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.video import Video
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.dropdown import DropDown
from kivy.core.window import Window

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Recognition imports
import csv
import copy
import argparse
import itertools
import time

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier

# Define arguments
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


class RADiagnosisApp(App):
    def build(self):
        # background colour
        Window.clearcolor = (240/255, 240/255, 240/255, 1)

        # Parse arguments for camera detection
        self.args = get_args()

        self.cap_device = self.args.device
        self.cap_width = self.args.width
        self.cap_height = self.args.height

        self.use_static_image_mode = self.args.use_static_image_mode
        self.min_detection_confidence = self.args.min_detection_confidence
        self.min_tracking_confidence = self.args.min_tracking_confidence

        self.use_brect = True

        # Camera preparation
        self.cap = cv.VideoCapture(self.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

        # Model load
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.keypoint_classifier = KeyPointClassifier()

        # Read labels
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]

        # FPS Measurement
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)

        self.start_time, self.end_time = 0, 0

        # App
        # home
        self.menu_title = Label(text='Rheumatoid Arthritis Diagnosis', color=(10/255,86/255,136/255,1), font_size='38sp', size_hint=(1, .3), font_name='Seguisli')
        self.fist_title = Label(text='Fist Clenching Test', color=(255,255,255,1), font_size='25sp', size_hint=(1, .1))
        self.sample_vid_title = Label(text='Clench your fists (Fist), open your hand fully (Open), and clench your fists again (Fist). \n'
                                           'Repeat this action in a cycle: Fist - Open - Fist',
                                      color=(255, 255, 255, 1), font_size='17sp', size_hint=(1, .2))

        # will update immediately once start so start with 3 (that gets replaced to 1 immediately)
        self.sample = Image(source="3_Sample Fist.png")
        self.sample_vid_btn = Button(text="Next", on_press=self.fist_clench_test, background_color=(65/255,100/255,223/255,0.8),
                                     background_normal='', color=(1,1,1,1), size_hint=(1, .2), font_size='17sp')
        self.home_wid = [self.menu_title, self.fist_title, self.sample_vid_title,
                         self.sample, self.sample_vid_btn]


        # fist_clench_test
        self.web_cam_title = Label(text='Repeat the action as fast as you can. '
                                        'Time will be taken from (start) fist - open - fist (end).\n '
                                        'You have unlimited tries and the best result will be taken.'
                                   , color=(255, 255, 255, 1), font_size='17sp', size_hint=(1, .2))
        self.web_cam = Image(size=(self.cap_width, self.cap_height))
        self.web_cam_valid = False
        self.time_label = Label(text="Time taken: {:.4f}s".format(0),
                                color=(255, 255, 255, 1), font_size='17sp', size_hint=(1, .1))
        self.best_time_label = Label(text="Shortest time taken: {:.4f}s".format(0),
                                     color=(255, 255, 255, 1), font_size='17sp', size_hint=(1, .1))
        self.web_cam_btn = Button(text='Next', on_press=self.question, background_color=(65/255,100/255,223/255,0.8),
                                     background_normal='', color=(1,1,1,1), size_hint=(1, .2), font_size='17sp')
        self.fist_clench_test_wid = [self.fist_title, self.web_cam_title,
                                     self.web_cam, self.time_label, self.best_time_label, self.web_cam_btn]

        # question
        self.qn_title = Label(text='Questionnaire', color=(255,255,255,1), font_size='25sp', size_hint=(1, .1))
        values = ['Yes', 'No']
        self.qn_desc = Label(text='Please answer the following questions to the best of your abilities\n'
                                  'to help us diagnose your risk of Rheumatoid Arthritis.',
                             color=(255, 255, 255, 1), font_size='17sp', size_hint=(1, .1), halign='center')

        self.qn1 = Label(text='Q1. Do you have a family history of Rheumatoid Arthritis?'
                         , color=(255, 255, 255, 1), font_size='17sp', size_hint=(1, .1))
        dropdown1 = DropDown()
        for index in range(2):
            # specify height manually (disable size_hint_y)
            # so the dropdown can calculate the area it needs
            btn = Button(text=values[index], size_hint=(1, None), height=30, background_color=(0.3, 0.3, 0.3, 1),
                                background_normal='')
            # attach a callback that will call the select() method on the dropdown
            # pass the text of the button as the data of the selection
            btn.bind(on_release=lambda btn: dropdown1.select(btn.text))
            # add button inside the dropdown
            dropdown1.add_widget(btn)
        self.q1_btn = Button(text='', size_hint=(1, None), height=36, background_color=(0.3, 0.3, 0.3, 1),
                                background_normal='')
        self.q1_btn.bind(on_release=dropdown1.open)
        dropdown1.bind(on_select=lambda instance, x: setattr(self.q1_btn, 'text', x))

        self.qn2 = Label(text='Q2. Are you a current smoker?'
                         , color=(255, 255, 255, 1), font_size='17sp', size_hint=(1, .1))
        dropdown2 = DropDown()
        for index in range(2):
            btn = Button(text=values[index], size_hint=(1, None), height=30, background_color=(0.3, 0.3, 0.3, 1),
                                background_normal='')
            btn.bind(on_release=lambda btn: dropdown2.select(btn.text))
            dropdown2.add_widget(btn)
        self.q2_btn = Button(text='', size_hint=(1, None), height=38, background_color=(0.3, 0.3, 0.3, 1),
                                background_normal='')
        self.q2_btn.bind(on_release=dropdown2.open)
        dropdown2.bind(on_select=lambda instance, x: setattr(self.q2_btn, 'text', x))

        self.qn3 = Label(text='Q3. Do you have morning stiffness for 60 minutes or more?'
                         , color=(255, 255, 255, 1), font_size='17sp', size_hint=(1, .1))
        dropdown3 = DropDown()
        for index in range(2):
            btn = Button(text=values[index], size_hint=(1, None), height=30, background_color=(0.3, 0.3, 0.3, 1),
                                background_normal='')
            btn.bind(on_release=lambda btn: dropdown3.select(btn.text))
            dropdown3.add_widget(btn)
        self.q3_btn = Button(text='', size_hint=(1, None), height=40, background_color=(0.3, 0.3, 0.3, 1),
                                background_normal='')
        self.q3_btn.bind(on_release=dropdown3.open)
        dropdown3.bind(on_select=lambda instance, x: setattr(self.q3_btn, 'text', x))

        self.qn_btn = Button(text='Next', on_press=self.result, background_color=(65/255,100/255,223/255,0.8),
                                     background_normal='', color=(1,1,1,1), size_hint=(1, .1), font_size='17sp')
        self.question_wid = [self.qn_title, self.qn_desc, self.qn1, self.q1_btn, self.qn2,
                             self.q2_btn, self.qn3, self.q3_btn, self.qn_btn]

        # result
        self.result_title = Label(text='Results', color=(255,255,255,1), font_size='25sp', size_hint=(1, .1))
        self.result_text = Label(text=''
                                 , color=(255, 255, 255, 1), font_size='17sp', size_hint=(1, .3))
        self.result_btn = Button(text='Restart', on_press=self.home, background_color=(65/255,100/255,223/255,0.8),
                                     background_normal='', color=(1,1,1,1), size_hint=(1, .1), font_size='17sp')
        self.result_wid = [self.result_title, self.result_text, self.result_btn]

        # Add items to layout
        self.layout = BoxLayout(orientation='vertical')
        for wid in self.home_wid:
            self.layout.add_widget(wid)

        Clock.schedule_interval(self.generateResult, 1.0)
        Clock.schedule_interval(self.loop_sample, 1.0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return self.layout

    def generateResult(self, *args):
        result = ''
        risk_eval = []
        # Fist Clench Test
        benchmark = 0.3  # passing benchmark, can be changed in the future
        threshold = 0.2  # threshold to determine the amt of risk for the user, split into 3 classes of risks
        if self.time_label.text == "Time taken: {:.4f}s".format(0):  # if user have not done FCT at all
            self.result_text.text = 'There was no data obtained from the Fist Clench Test, please redo your test.'
            return
        else:
            shortest_time = float(self.best_time_label.text[21:-1])
            result = 'Shortest time you took in the Fist Clench Test: {:.4f}s\n'.format(shortest_time)
            # Class 1 - No/Low Risk
            if shortest_time <= benchmark:
                risk_eval += [1]
            # Class 2 - Medium Risk
            elif shortest_time <= benchmark + threshold:
                risk_eval += [2]
            # Class 3 - High Risk
            else:
                risk_eval += [3]

        # Questionnaire
        answers = [self.q1_btn.text, self.q2_btn.text, self.q3_btn.text]
        for answer in answers:
            if answer == 'Yes':
                risk_eval += [3]
            else:
                risk_eval += [1]

        result += '\nAfter calculation of results from the Fist Clench Test and the Questionnaire,\n' \
                  'we have diagnosed your risk of developing Rheumatoid Arthritis to be:\n'

        # Calculate with weights, here we set FCT (index 0) as biggest weightage
        # Index: FCT, q1, q2, q3
        weights = [0.4, 0.2, 0.2, 0.2]
        risk_score = 0
        for i in range(len(weights)):
            risk_score += risk_eval[i] * weights[i]
        risk_score /= 3
        # risk_score range: (lowest) 0.33 to 1.0 (highest)
        # Low risk
        if risk_score <= 0.5:
            result += 'NO/LOW RISK\n'
        elif risk_score <= 0.7:
            result += 'MEDIUM RISK\n'
        else:
            result += 'HIGH RISK\n'

        self.result_text.text = result

    def fist_clench_test(self, instance):
        for widget in self.home_wid:
            self.layout.remove_widget(widget)
        for widget in self.fist_clench_test_wid:
            self.layout.add_widget(widget)
        self.web_cam_valid = True

    def question(self, instance):
        for widget in self.fist_clench_test_wid:
            self.layout.remove_widget(widget)
        for widget in self.question_wid:
            self.layout.add_widget(widget)

    def result(self, instance):
        for widget in self.question_wid:
            self.layout.remove_widget(widget)
        for widget in self.result_wid:
            self.layout.add_widget(widget)

    def home(self, instance):
        for widget in self.result_wid:
            self.layout.remove_widget(widget)
        for widget in self.home_wid:
            self.layout.add_widget(widget)

        #  Init
        self.start_time, self.end_time = 0, 0
        self.time_label.text = "Time taken: {:.4f}s".format(0)
        self.best_time_label.text = "Shortest time taken: {:.4f}s".format(0)


    def loop_sample(self, *args):
        if self.sample.source[0] == '1':
            self.sample.source = "2_Sample Open.png"
        elif self.sample.source[0] == '2':
            self.sample.source = "3_Sample Fist.png"
        else:
            self.sample.source = "1_Sample Fist.png"

    def update(self, *args):
        if not self.web_cam_valid: #webcam is hidden
            return

        fps = self.cvFpsCalc.get()

        # Camera capture
        ret, image = self.cap.read()
        if not ret:
            return
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = self.pre_process_landmark(
                    landmark_list)

                # Hand sign classification
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id != -1:
                    if hand_sign_id == 0:  # Open
                        if self.start_time == -1:  # Fist previously
                            self.start_time = time.time()
                    elif hand_sign_id == 1:  # Fist
                        if self.start_time not in [0, -1]:  # Not init/Moving/Fist - Was Open previously
                            self.end_time = time.time()
                            time_taken = self.end_time - self.start_time
                            self.time_label.text = "Time taken: {:.4f}s".format(time_taken)
                            if float(self.best_time_label.text[21:-1]) > time_taken or self.best_time_label.text == "Shortest time taken: {:.4f}s".format(0):
                                self.best_time_label.text = "Shortest time taken: {:.4f}s".format(time_taken)
                        self.start_time, self.end_time = -1, 0  # Reset

                    # Drawing
                    debug_image = self.draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        self.keypoint_classifier_labels[hand_sign_id],
                    )

                # Drawing part
                debug_image = self.draw_bounding_rect(self.use_brect, debug_image, brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)

        debug_image = self.draw_info(debug_image, fps)

        # Screen reflection
        #cv.imshow('Hand Gesture Recognition', debug_image)
        buf = cv.flip(debug_image, 0).tostring()
        img_texture = Texture.create(size=(debug_image.shape[1], debug_image.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture


    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            # Index finger
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255, 255, 255), 2)

            # Middle finger
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255, 255, 255), 2)

            # Ring finger
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255, 255, 255), 2)

            # Little finger
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            # Palm
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        return image

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)
        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        return image

    def draw_info(self, image, fps):
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)
        return image

    def hide_widget(self, wid, dohide=True):
        if hasattr(wid, 'saved_attrs'):
            if not dohide:
                wid.height, wid.size_hint_y, wid.opacity, wid.disabled = wid.saved_attrs
                del wid.saved_attrs
        elif dohide:
            wid.saved_attrs = wid.height, wid.size_hint_y, wid.opacity, wid.disabled
            wid.height, wid.size_hint_y, wid.opacity, wid.disabled = 0, None, 0, True

if __name__ == '__main__':
    RADiagnosisApp().run()