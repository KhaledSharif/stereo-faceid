import numpy as np
import cv2
from imutils import face_utils
import imutils
import dlib
from kalman import KalmanFilter3D

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

N = 68
RESIZE_WIDTH = 400

kalman_filters = []
for n in range(N):
	kalman_filters.append(
		KalmanFilter3D(88.5,  66.0, 2637.0)
	)

while(True):
	points_left_frame = [None] * N
	points_right_frame = [None] * N	
	
	ret, frame = cap1.read()
	gray1 = imutils.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), width=RESIZE_WIDTH)
	rects = detector(gray1, 1)
	cap1_points_found = 0
	for rect in rects:
		shape = face_utils.shape_to_np(predictor(gray1, rect))
		cap1_points_found = len(shape)
		for shape_index, (x, y) in enumerate(shape):
			points_left_frame[shape_index] = (x, y)
			cv2.circle(gray1, (x, y), 1, (0, 0, 255), -1)
		break

	ret, frame = cap2.read()
	gray2 = imutils.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), width=RESIZE_WIDTH)
	rects = detector(gray2, 1)
	cap2_points_found = 0
	for rect in rects:
		shape = face_utils.shape_to_np(predictor(gray2, rect))
		cap2_points_found = len(shape)
		for shape_index, (x, y) in enumerate(shape):
			points_right_frame[shape_index] = (x, y)
			cv2.circle(gray2, (x, y), 1, (0, 0, 255), -1)
		break

	if (cap1_points_found + cap2_points_found) == (2 * N):
		points_center = [None] * N
		points_z = [None] * N

		for point_index in range(N):
			points_center[point_index] = (
				0.5 * (points_left_frame[point_index][0] + points_right_frame[point_index][0]),
				0.5 * (points_left_frame[point_index][1] + points_right_frame[point_index][1])
			)
			points_z[point_index] = \
				(points_left_frame[point_index][0] - points_right_frame[point_index][0]) ** 2.0 + \
				(points_left_frame[point_index][1] - points_right_frame[point_index][1]) ** 2.0

		
		for point_index in [0]:
			kalman_filters[point_index].update(
				measurement_x=points_center[point_index][0], 
				measurement_y=points_center[point_index][1], 
				measurement_z=points_z[point_index],
			)
			x,y,z = kalman_filters[point_index].predict()

			'''
			print("{}. [measured] x = {}, y = {}, z = {}".format(
				point_index+1, 
				points_center[point_index][0], 
				points_center[point_index][1],
				points_z[point_index],
			))
			print("{}. [predicted] x = {}, y = {}, z = {}".format(
				point_index+1, 
				x,y,z
			))
			'''
	else:
		for point_index in [0]:
			kalman_filters[point_index].predict()
			

	hstack = np.hstack((gray1, gray2))

	cv2.imshow('frame', hstack)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
