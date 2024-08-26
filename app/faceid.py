# importing Kivy dependencies

#  for app layout
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# for UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# building app and layout
class CamApp(App):
    def build(self):
        # Main layout
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitialized", size_hint = (1,.1))

        # Detailing
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # for loading model
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30.0)

        return layout
    

    # runs continuously for getting webcam feed
    def update(self, *args):
        
        ret, frame = self.capture.read()
        frame = frame[115:365, 195:445, :]

        # flip horizontal and convert image to texture
        buf = cv2.flip(frame,0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt = 'bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture


    # function for resizing image
    def preprocess(self, img_path):
        byte_img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img
    

    # function to verify the user
    def verify (self, *args):
        detection_threshold = 0.9
        verification_threshold = 0.8

        # for capturing input image
        SAVE_PATH = os.path.join('application_data', 'input_img', 'input_img.jpg')
        ret, frame = self.capture.read()
        frame = frame[115:365, 195:445, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_img')):
            input_image = self.preprocess(os.path.join('application_data', 'input_img', 'input_img.jpg'))
            validation_image = self.preprocess(os.path.join('application_data', 'verification_img', image))
            
            result = self.model.predict(list(np.expand_dims([input_image, validation_image], axis=1)))
            results.append(result)
            
        # Detection
        detection = np.sum(np.array(results) > detection_threshold)

        # verification for number of images passing the detection threshold
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_img')))
        verified = verification > verification_threshold

        # updating verification button/text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)

        return results, verified


if __name__ == '__main__':
    CamApp().run()