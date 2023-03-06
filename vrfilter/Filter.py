import os
import glob
import math

import cv2
import numpy as np
import mediapipe as mp


class Filter:

    def __init__(self):
        # Quick access mediapipe variables.
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        # Headset mask applied to image. IMREAD_UNCHANGED is applied so it also reads the alpha channel.
        self.headset_image = cv2.imread("assets/images/headset/headset.png", cv2.IMREAD_UNCHANGED)
        self.headset_width, self.headset_height, _ = self.headset_image.shape

        # Face mask applied to image. IMREAD_UNCHANGED is applied so it also reads the alpha channel.
        self.face_mask_image = cv2.imread("assets/images/face_mask/face_mask.png", cv2.IMREAD_UNCHANGED)
        self.face_mask__width, self.face_mask__height, _ = self.face_mask_image.shape

        # Increment size of the headset by this percentage.
        self.increment_size = 0.0

    def apply_filter_vr_folder(self, folder_path, output_path):
        files = glob.glob(folder_path + "*.jpg")

        with self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            failed = 0
            failed_image_names = []
            for file in files:
                self.headset_width, self.headset_height, _ = self.headset_image.shape
                success, image_path = self.apply_filter_image_vr(file, output_path, face_mesh)
                if not success:
                    failed += 1
                    failed_image_names.append(image_path)

            return failed_image_names

    def apply_filter_image_vr(self, image_path, output_path, face_mesh):
        self.headset_width, self.headset_height, _ = self.headset_image.shape

        # Read image.
        image = cv2.imread(image_path)
        width, height, _ = image.shape

        # Initialize MediaPipe.

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False

        # Process image.
        results = face_mesh.process(image)

        # If a face bas been detected...
        if results.multi_face_landmarks:
            # Print face_landmarks.
            # for face_landmarks in results.multi_face_landmarks:
            #     self.mp_drawing.draw_landmarks(
            #         image=image,
            #         landmark_list=face_landmarks,
            #         connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            #         landmark_drawing_spec=None,
            #         connection_drawing_spec=self.mp_drawing_styles
            #         .get_default_face_mesh_tesselation_style())

            # Get sides of right eye.
            landmark_right_eye_left = results.multi_face_landmarks[0].landmark[263]
            landmark_right_eye_right = results.multi_face_landmarks[0].landmark[362]
            # Calculate right eye coordinates on the original image.
            coord_right_eye_x_left = int(landmark_right_eye_left.x * height)
            coord_right_eye_y_left = int(landmark_right_eye_left.y * width)
            coord_right_eye_x_right = int(landmark_right_eye_right.x * height)
            coord_right_eye_y_right = int(landmark_right_eye_right.y * width)
            # Get center of right eye.
            center_right_eye_x = int((coord_right_eye_x_left + coord_right_eye_x_right) / 2)
            center_right_eye_y = int((coord_right_eye_y_left + coord_right_eye_y_right) / 2)

            # Get sides of left eye.
            landmark_left_eye_left = results.multi_face_landmarks[0].landmark[133]
            landmark_left_eye_right = results.multi_face_landmarks[0].landmark[33]
            # Calculate left eye coordinates on the original image.
            coord_left_eye_x_left = int(landmark_left_eye_left.x * height)
            coord_left_eye_y_left = int(landmark_left_eye_left.y * width)
            coord_left_eye_x_right = int(landmark_left_eye_right.x * height)
            coord_left_eye_y_right = int(landmark_left_eye_right.y * width)
            # Get center of left eye.
            center_left_eye_x = int((coord_left_eye_x_left + coord_left_eye_x_right) / 2)
            center_left_eye_y = int((coord_left_eye_y_left + coord_left_eye_y_right) / 2)

            # Get center point between eyes.
            landmark_center_x = int((center_left_eye_x + center_right_eye_x) / 2)
            landmark_center_y = int((center_left_eye_y + center_right_eye_y) / 2)

            # Get points at both sides of head.
            landmark_left = results.multi_face_landmarks[0].landmark[127]
            landmark_right = results.multi_face_landmarks[0].landmark[356]

            # Calculate coordinates of both sides of head on the original image.
            landmark_left_x = int(landmark_left.x * height)
            landmark_left_y = int(landmark_left.y * width)
            landmark_right_x = int(landmark_right.x * height)
            landmark_right_y = int(landmark_right.y * width)

            # Get angle of head and desired width for the mask.
            angle = math.atan2(landmark_right_y - landmark_left_y, landmark_right_x - landmark_left_x) * 180 / math.pi
            desired_width = math.dist([landmark_left_x, landmark_left_y], [landmark_right_x, landmark_right_y])

            # Scale and rotate image.
            rotated_and_scaled = self.rotate_and_scale(self.headset_image,
                                                       (desired_width / self.headset_height) + self.increment_size,
                                                       -int(angle))
            self.headset_height, self.headset_width, _ = rotated_and_scaled.shape

            # Add alpha channel to original image.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

            # Locate mask on image.
            img_overlay_rgba = np.array(rotated_and_scaled)
            alpha_mask = img_overlay_rgba[:, :, 3] / 255.0
            self.overlay_image_alpha(image, rotated_and_scaled, int(landmark_center_x - (self.headset_width / 2)),
                                     int(landmark_center_y - (self.headset_height / 2)), alpha_mask)

            # Write image.
            cv2.imwrite(output_path + os.path.basename(image_path), image)

            # Return success.
            return True, os.path.basename(image_path)
        else:
            # Return fail.
            return False, os.path.basename(image_path)
        pass

    def apply_filter_face_mask_folder(self, folder_path, output_path):
        files = glob.glob(folder_path + "*.jpg")

        with self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            failed = 0
            failed_image_names = []
            for file in files:
                self.headset_width, self.headset_height, _ = self.headset_image.shape
                success, image_path = self.apply_filter_image_face_mask(file, output_path, face_mesh)
                if not success:
                    failed += 1
                    failed_image_names.append(image_path)

            return failed_image_names

    def apply_filter_image_face_mask(self, image_path, output_path, face_mesh):
        self.face_mask__width, self.face_mask__height, _ = self.face_mask_image.shape

        # Read image.
        image = cv2.imread(image_path)
        width, height, _ = image.shape

        # Initialize MediaPipe.

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False

        # Process image.
        results = face_mesh.process(image)

        # If a face bas been detected...
        if results.multi_face_landmarks:
            # Print face_landmarks.
            # for face_landmarks in results.multi_face_landmarks:
            #     self.mp_drawing.draw_landmarks(
            #         image=image,
            #         landmark_list=face_landmarks,
            #         connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            #         landmark_drawing_spec=None,
            #         connection_drawing_spec=self.mp_drawing_styles
            #         .get_default_face_mesh_tesselation_style())

            # Get sides of face.
            landmark_face_left = results.multi_face_landmarks[0].landmark[132]
            landmark_face_right = results.multi_face_landmarks[0].landmark[361]
            # Calculate face coordinates on the original image.
            coord_face_left_x = int(landmark_face_left.x * height)
            coord_face_left_y = int(landmark_face_left.y * width)
            coord_face_right_x = int(landmark_face_right.x * height)
            coord_face_right_y = int(landmark_face_right.y * width)

            # Get bottom of nose.
            landmark_nose_left = results.multi_face_landmarks[0].landmark[15]
            # Calculate nose coordinates on the original image.
            coord_nose_left_x = int(landmark_nose_left.x * height)
            coord_nose_left_y = int(landmark_nose_left.y * width)

            # Get angle of mouth and desired width
            # for the mask.
            angle = math.atan2(coord_face_right_y - coord_face_left_y, coord_face_right_x - coord_face_left_x) * 180 / math.pi
            desired_width = math.dist([coord_face_left_x, coord_face_left_y], [coord_face_right_x, coord_face_right_y])

            # Scale and rotate image.
            rotated_and_scaled = self.rotate_and_scale(self.face_mask_image,
                                                       (desired_width / self.face_mask__height) + self.increment_size,
                                                       -int(angle))
            self.face_mask__height, self.face_mask__width, _ = rotated_and_scaled.shape

            # Add alpha channel to original image.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

            # Locate mask on image.
            img_overlay_rgba = np.array(rotated_and_scaled)
            alpha_mask = img_overlay_rgba[:, :, 3] / 255.0
            self.overlay_image_alpha(image, rotated_and_scaled, int(coord_nose_left_x - (self.face_mask__width / 2)),
                                     int(coord_nose_left_y - (self.face_mask__height / 2)), alpha_mask)

            # Write image.
            cv2.imwrite(output_path + os.path.basename(image_path), image)

            # Return success.
            return True, os.path.basename(image_path)
        else:
            # Return fail.
            return False, os.path.basename(image_path)
        pass


    def rotate_and_scale(self, img, scale_factor=0.5, degrees=30):
        # Get image shape. Note: Numpy uses (y,x) convention but most OpenCV functions use (x,y).
        old_y, old_x, _ = img.shape  # note:

        # Calculate rotation matrix. Rotate around center of image.
        M = cv2.getRotationMatrix2D(center=(old_x / 2, old_y / 2), angle=degrees, scale=scale_factor)

        # Calculate new image size.
        new_x, new_y = old_x * scale_factor, old_y * scale_factor

        # Calculate new size of image after rotating.
        r = np.deg2rad(degrees)
        new_x, new_y = (abs(np.sin(r) * new_y) + abs(np.cos(r) * new_x),
                        abs(np.sin(r) * new_x) + abs(np.cos(r) * new_y))

        # The warpAffine function call, below, basically works like this:
        # 1. Apply the M transformation on each pixel of the original image.
        # 2. Save everything that falls within the upper-left "dsize" portion of the resulting image.

        # So, find the translation that moves the result to the center of that region.
        (tx, ty) = ((new_x - old_x) / 2, (new_y - old_y) / 2)
        M[0, 2] += tx  # Third column of matrix holds translation, which takes effect after rotation.
        M[1, 2] += ty

        # Perform rotation.
        rotated_img = cv2.warpAffine(img, M, dsize=(int(new_x), int(new_y)))

        return rotated_img

    def overlay_image_alpha(self, img, img_overlay, x, y, alpha_mask):
        """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

        `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
        """
        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do.
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        # Blend overlay within the determined ranges
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
        alpha_inv = 1.0 - alpha

        img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
