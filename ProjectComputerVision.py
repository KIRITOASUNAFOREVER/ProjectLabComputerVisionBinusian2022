import os
import cv2
import numpy as np
def get_all_train_image_labels(path):
    image_list = []
    image_index = []
  
    train_directory = os.listdir(path)
    for index, train in enumerate(train_directory):
        image_path_list = os.listdir(path + '/' + train)
        image_list.append(train)
        for image_path in image_path_list:
            if(image_path[-3:]=='jpg'):
                image_index.append(index)
    return image_list,image_index
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing all train images label
        list
            List containing all train images indexes
    '''

def get_all_train_images(path):
    image_data = []
  
  
    train_directory = os.listdir(path)
    for train in train_directory:
        image_path_list = os.listdir(path + '/' + train)
        for image_path in image_path_list:
                if(image_path[-3:]=='jpg'):
                    image = cv2.imread(path +'/' + train + '/' + image_path)
                    image_data.append(image)
    return image_data
    '''
        Get all Train Images & resize it using the given path

        Parameters
        ----------
        path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing all the resized train images
    '''

def detect_faces_and_filter(image_list, image_labels=None):
 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image_location = []
    image_penampung = []
    label_penampung = []
    for idx,image in enumerate(image_list):
        faces = face_cascade.detectMultiScale(image, 1.2, 3)
        if len(faces)==1:
            for x,y,w,h in faces:
                image_location.append((x,y,w,h))
                image_crop = image[y:y+h,x:x+w]
                width = 300
                height = int(image_crop.shape[0] * width / image_crop.shape[1])
                dim = (width,height)
                image_crop = cv2.resize(image_crop,dim)
                image_crop = cv2.cvtColor(image_crop,cv2.COLOR_BGR2GRAY)
                image_penampung.append(image_crop)
                if image_labels is not None:
                    label_penampung.append(image_labels[idx])
    return image_penampung,image_location,label_penampung
    '''
        To detect a face from given image list and filter it if the face on
        the given image is not equals to one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_labels : list
            List containing all image classes labels
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            list containing image gray face location
        list
            List containing all filtered image classes label
    '''

def train(gray_image_list, gray_labels):
    lbph = cv2.face.LBPHFaceRecognizer_create()
    lbph.train(gray_image_list,np.array(gray_labels))
    return lbph
    '''
        To create and train face recognizer object

        Parameters
        ----------
        gray_image_list : list
            List containing all filtered and cropped face images in grayscale
        gray_labels : list
            List containing all filtered image classes label
        
        Returns
        -------
        object
            Classifier object after being trained with cropped face images
    '''

def get_all_test_images(path):
    image_data = [] 
    image_path_list = os.listdir(path + '/')
    for image_path in image_path_list:
        if(image_path[-3:]=='jpg'):
            image = cv2.imread(path +'/'+ image_path)
            image_data.append(image)
    return image_data
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all image in the test directories
    '''

def predict(classifier, gray_test_image_list):
    prediction_list = []
    for image in gray_test_image_list:
        lists, _ = classifier.predict(image)
        prediction_list.append(lists)
    return prediction_list
    '''
        To predict the test image with the classifier

        Parameters
        ----------
        classifier : object
            Classifier object after being trained with cropped face images
        gray_test_image_list : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

def write_prediction(predict_results, test_image_list, test_faces_rects, train_labels):
    train_labels = np.unique(train_labels) 
    images = []
    for idx,image in enumerate(test_image_list):
        x, y, w, h = test_faces_rects[idx]
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        image = cv2.putText(image,train_labels[predict_results[idx]],(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        images.append(image)
    return images
    '''
        To draw prediction results on the given test images and resize the image

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''

def combine_and_show_result(image_list):
    scale_percent = 60 # percent of original size
    width = int(image_list[0].shape[1] * scale_percent / 100)
    height = int(image_list[0].shape[0] * scale_percent / 100)
    dim = (width, height)
    image_list[0] = cv2.resize(image_list[0], dim, interpolation = cv2.INTER_AREA)
    width = int(image_list[1].shape[1] * scale_percent / 100)
    height = int(image_list[1].shape[0] * scale_percent / 100)
    dim = (width, height)
    image_list[1] = cv2.resize(image_list[1], dim, interpolation = cv2.INTER_AREA)
    width = int(image_list[2].shape[1] * scale_percent / 100)
    height = int(image_list[2].shape[0] * scale_percent / 100)
    dim = (width, height)
    image_list[2] = cv2.resize(image_list[2], dim, interpolation = cv2.INTER_AREA)
    width = int(image_list[3].shape[1] * scale_percent / 100)
    height = int(image_list[3].shape[0] * scale_percent / 100)
    dim = (width, height)
    image_list[3] = cv2.resize(image_list[3], dim, interpolation = cv2.INTER_AREA)
    width = int(image_list[4].shape[1] * scale_percent / 100)
    height = int(image_list[4].shape[0] * scale_percent / 100)
    dim = (width, height)
    image_list[4] = cv2.resize(image_list[4], dim, interpolation = cv2.INTER_AREA)
    combined_image = np.hstack(image_list)
    cv2.imshow("Final Result",combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''


'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''

if __name__ == "__main__":

    '''
        Please modify train_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    train_image_labels, train_image_indexes = get_all_train_image_labels(train_path)
    train_image_list = get_all_train_images(train_path)
    gray_train_image_list, _, gray_train_labels = detect_faces_and_filter(train_image_list, train_image_indexes)
    
    classifier = train(gray_train_image_list, gray_train_labels)

    '''
        Please modify test_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_all_test_images(test_path)
    gray_test_image_list, gray_test_location, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(classifier, gray_test_image_list)
    predicted_test_image_list = write_prediction(predict_results, test_image_list, gray_test_location, train_image_labels)
    
    combine_and_show_result(predicted_test_image_list)