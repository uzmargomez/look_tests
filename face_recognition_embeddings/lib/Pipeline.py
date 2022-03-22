import glob
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from keras_facenet import FaceNet
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import heapq
import pickle

detector = MTCNN()
embedder = FaceNet()

class DataSet:
	def __init__(self, directory, extension, size, slope_limit, intercept_limit):

		self.train_images = {}
		self.test_images_known = {}
		self.test_images_unknown = {}
		self.test_images_group = []
		self.classes = {}
		self.size=size
		self.slope_limit = slope_limit
		self.intercept_limit = intercept_limit

		aux = sorted(glob.glob( directory + "/Train/*" ))
		for d in aux:
			d=d.replace("\\","/")
			images = glob.glob(d + "/*." + extension)
			self.train_images[d.split("/")[-1]]=images

		self.subjects_number = len(self.train_images)
		self.index_to_subject = {}
		self.subject_to_index = {}

		for i,subject in enumerate(self.train_images):
			self.index_to_subject[i]=subject
			self.subject_to_index[subject]=i

		aux = sorted(glob.glob( directory + "/Test/*" ))
		for d in aux:
			d=d.replace("\\","/")
			images = glob.glob(d + "/*." + extension)
			self.test_images_known[d.split("/")[-1]]=images

		aux = sorted(glob.glob( directory + "/Unknown/*" ))
		for d in aux:
			d=d.replace("\\","/")
			images = glob.glob(d + "/*." + extension)
			self.test_images_unknown[d.split("/")[-1]]=images

		aux = sorted(glob.glob( directory + "/Group" ))
		for d in aux:
			d=d.replace("\\","/")
			images = glob.glob(d+ "/*." + extension)
			for i in images:
				self.test_images_group.append(i)

	def print_dataset_info(self):
		print("\n\n")
		print("*"*50,"\n*             DATASET INFORMATION                *")
		print("*"*50,"\n")
		print("Number of subjects for training:", self.subjects_number)
		print("Images per subject for training:", len(self.train_images[self.index_to_subject[0]]))
		print("Size of the images to identify: ", self.size)
		print("Slope limit:	", self.slope_limit)
		print("Intercept limit:	", self.intercept_limit)
		print("\n")

	def load_model(self,name,train=False):
		if train:
			print("\n\n")
			print("*"*50,"\n*                START TRAINING                  *")
			print("*"*50,"\n")
			if not os.path.exists('models'):
				os.makedirs('models')

			# Getting the arrays for the training
			train_y_subjects, train_x = load_set(self.train_images, size = 160, print_info=True)

			# Getting embeddings from FaceNet and normalizing them
			train_embeddings = embedder.embeddings(train_x)
			train_x = Normalizer(norm='l2').transform(train_embeddings)

			# Encoder in order to associate labels with a certain number
			train_y = []
			for subject in train_y_subjects:
				train_y.append(self.subject_to_index[subject])

			train_y = np.asarray(train_y)

			model = SVC(kernel='linear', probability=True)

			model.fit(train_x, train_y)

			pickle.dump(model, open('models/{}.sav'.format(name), 'wb'))

			self.model = pickle.load(open('models/{}.sav'.format(name), 'rb'))
			# return train_x, train_y
		else:
			self.model = pickle.load(open('models/{}.sav'.format(name), 'rb'))

	def test_model(self,graphs=False,print_info=True,print_detail=False):
		if print_info:
			print("\n\n")
			print("*"*50,"\n*             START TESTING (Known)              *")
			print("*"*50,"\n")
		test_y_subjects, test_x = load_set(self.test_images_known, size = self.size, print_info=print_info)

		test_embeddings = embedder.embeddings(test_x)
		test_x = Normalizer(norm='l2').transform(test_embeddings)

		# Encoder in order to associate labels with a certain number
		test_y = []
		for subject in test_y_subjects:
			test_y.append(self.subject_to_index[subject])

		test_y = np.asarray(test_y)

		# known_test_x = test_x
		# known_test_y = test_y
		y_test_pred = self.model.predict(test_x)
		y_test_proba = self.model.predict_proba(test_x)
		real_pred=[]
		slopes=[]
		intercepts=[]
		y_proba_new = []
		for pred, proba in zip(y_test_pred,y_test_proba):
			result, y_proba, slope, intercept=identify_unknown(probabilities=proba,index=pred,slope_limit=self.slope_limit,intercept_limit=self.intercept_limit)
			real_pred.append(result)
			slopes.append(slope)
			intercepts.append(intercept)
			y_proba_new.append(y_proba)

		if print_detail==True:

			print("\n*************************************************************************")
			print("*                       Testing known images                            *")
			print("*************************************************************************")
			print("TEST SUBJECT                  | CLASSIFICATION                | RESULT   ")
			print("------------------------------|-------------------------------|----------")
			for pred, test in zip(real_pred,test_y):
				if pred==-1:
					result="incorrect"
					print("{:<30}| {:<30}| {:<10}".format(self.index_to_subject[test], "* NOT IN DB *", result))
				else:
					if pred==test:
						result="correct"
					else:
						result="incorrect"
					print("{:<30}| {:<30}| {:<10}".format(self.index_to_subject[test],self.index_to_subject[pred], result))


		if graphs==True:

			if not os.path.exists('Graphs_{}/Known'.format(self.size)):
				os.makedirs('Graphs_{}/Known'.format(self.size))

			counter=0
			for prob_vec,class_result,slope,intercept in zip(y_proba_new,y_test_pred,slopes,intercepts):
				x = range(len(prob_vec))
				x_val = np.array(x)
				y_val = intercept + slope*x_val
				plt.plot(x,prob_vec,'ro')
				plt.plot(x_val,y_val,'--')
				plt.xlabel("Class")
				plt.ylabel("Probability")
				plt.ylim(0.,1.05)
				plt.savefig('Graphs_{}/Known/plot_class_{}_{}.png'.format(self.size,class_result,counter), dpi=300)
				plt.close()
				counter = counter +1

		score_test_known = accuracy_score(test_y,real_pred)
		print("Accuracy on known:",score_test_known*100,"%\n\n")



		if print_info:
			print("\n\n")
			print("*"*50,"\n*            START TESTING (Unknown)             *")
			print("*"*50,"\n")
		test_y_subjects, test_x = load_set(self.test_images_unknown, size = self.size,print_info=print_info)

		test_embeddings = embedder.embeddings(test_x)
		test_x = Normalizer(norm='l2').transform(test_embeddings)

		test_y = [-1 for i in range(len(test_y_subjects))]

		test_y = np.asarray(test_y)

		y_test_pred = self.model.predict(test_x)
		y_test_proba = self.model.predict_proba(test_x)
		real_pred=[]
		slopes=[]
		intercepts=[]
		y_proba_new = []
		for pred, proba in zip(y_test_pred,y_test_proba):
			result, y_proba, slope, intercept =identify_unknown(probabilities=proba,index=pred,slope_limit=self.slope_limit,intercept_limit=self.intercept_limit)
			real_pred.append(result)
			slopes.append(slope)
			intercepts.append(intercept)
			y_proba_new.append(y_proba)

		if graphs==True:

			if not os.path.exists('Graphs_{}/Unknown'.format(self.size)):
				os.makedirs('Graphs_{}/Unknown'.format(self.size))

			counter=0
			for prob_vec,class_result,slope,intercept in zip(y_proba_new,y_test_pred,slopes,intercepts):
				x = range(len(prob_vec))
				x_val = np.array(x)
				y_val = intercept + slope*x_val
				plt.plot(x,prob_vec,'ro')
				plt.plot(x_val,y_val,'--')
				plt.xlabel("Class")
				plt.ylabel("Probability")
				plt.ylim(0.,1.05)
				plt.savefig('Graphs_{}/Unknown/plot_class_{}_{}.png'.format(self.size,class_result,counter), dpi=300)
				plt.close()
				counter = counter +1

		score_test_unknown = accuracy_score(test_y,real_pred)

		print("Accuracy on unknown:",score_test_unknown*100,"%\n\n")

		# unknown_test_x = test_x
		# unknown_test_y = test_y

		return score_test_known*100, score_test_unknown*100
		# return known_test_x, known_test_y, unknown_test_x, unknown_test_y

	def classify_image(self,im):
		emb_im = embedder.embeddings(np.array([im]))

		pred = self.model.predict(emb_im)
		pred_proba = self.model.predict_proba(emb_im)

		result,_,_,_ = identify_unknown(probabilities=pred_proba[0], index=pred[0], slope_limit=self.slope_limit,intercept_limit=self.intercept_limit)

		if result == -1:
			return "Unknown"
		else:
			return self.index_to_subject[result]

	def single_image(self,filename):
		print("TESTING SINGLE IMAGE\n")
		im = extract_face(filename,self.size)
		print("Image was classified as",self.classify_image(im))

	def modify_image(self, frame):

		color = (246, 181, 100)

		img = frame

		preview = img

		detections = detector.detect_faces(img)

		if not detections:
			return preview
		for detection in detections:
			box = detection['box']

			face = img[abs(box[1]):abs(box[1])+abs(box[3]), abs(box[0]):abs(box[0])+abs(box[2])]

			if self.size == -1:
				face_array = face
			else:
				face_array = cv2.resize(face,(self.size,self.size),interpolation=cv2.INTER_NEAREST)

			identity = self.classify_image(face_array)

			if identity is not None:

				cv2.rectangle(preview,(abs(box[0]), abs(box[1])), (abs(box[0])+abs(box[2]), abs(box[1])+abs(box[3])),color, 1)

				cv2.putText(preview, identity, (abs(box[0]), abs(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,cv2.LINE_AA)

				print(identity)

		return preview

	def testing_webcam(self, video_path = 0):

		print("TESTING WEBCAM")

		cap = cv2.VideoCapture(video_path)

		while True:
			ok, frame = cap.read()
			if ok:

				preview = self.modify_image(frame)

			cv2.imshow("preview", preview)

			k = cv2.waitKey(0)
			if k == 27:
				break

def identify_unknown(probabilities,index,slope_limit,intercept_limit):
	new_x = []
	maximum = probabilities[index]
	for v in probabilities:
		aux = v/maximum
		new_x.append(aux)

	y = np.delete(new_x,index)
	x = np.array([i for i in range(len(y))]).reshape((-1,1))

	regressor = LinearRegression()
	model_regressor = regressor.fit(x,y)

	slope = model_regressor.coef_
	intercept = model_regressor.intercept_

	if abs(slope)>=slope_limit or intercept>=intercept_limit:
		ind = -1
	else:
		ind = index

	return ind, new_x, slope, intercept

def load_set(set,size,print_info):
	if print_info:
		print("\nLoading set...\n")
	images = []
	labels = []
	for person in set:
		for path in set[person]:
			a = extract_face(path,size)
			if a is None:
				print("No face detected in file:",path)
			else:
				images.append(a)
				labels.append(person)
		if print_info:
			print("Got",len(set[person]),"images for subject",person)
	if print_info:
		print("\n")
	return np.asarray(labels),np.asarray(images)


def extract_face(filename,size):
	image = cv2.imread(filename)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	results = detector.detect_faces(image)
	if not results:
		return None

	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

	face = image[y1:y2, x1:x2]

	if size == -1:
		return face
	else:
		face_array = cv2.resize(face,(size,size),interpolation=cv2.INTER_NEAREST)
		return face_array
