##Tayfun Pay, PhD 
from __future__ import division
import imageio
import os
import pickle
import numpy as np
import pandas as pd
from random import randint
from sklearn import svm, metrics
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.filters import prewitt

###
###STEP 1 - Processing the Images
###

def Process_Images(factor):
    df = pd.read_csv("find_phone/"+"labels.txt", sep=" ", header=None)
    df.columns = ["file_name", "x_coordinate", "y_coordinate"]
    yes  = 1
    no = 1

    for index, row in df.iterrows():
        image = imageio.imread("find_phone/"+row["file_name"])
        x = int(float(row["x_coordinate"]) * 490)
        y = int(float(row["y_coordinate"]) * 326)
        ymin = y - factor
        ymax = y + factor
        xmin = x - factor
        xmax = x + factor
    
        if xmin < 0:
            xmax = xmax+abs(xmin)
            xmin = 0
        if xmax > 490:
            xmin = xmin - (xmax-490)
            xmax = 490
        if ymin < 0:
            ymax = ymax+abs(ymin)
            ymin = 0
        if ymax > 326:
            ymin = ymin - (ymax-326)
            ymax = 326
        
        cropped = image[ymin:ymax, xmin:xmax]
        imageio.imwrite("yes_"+str(yes)+".jpg", cropped, quality=100)
        yes +=1
        i = 0
        while i < 32:
            xrandom = randint(33,457)
            yrandom = randint(33,293)
            if (not (((xrandom>=xmin) and (xrandom<=xmax)) or
                     ((yrandom>=ymin) and (yrandom<=ymax))) ):
                cropped_neg = image[yrandom-factor:yrandom+factor,
                                    xrandom-factor:xrandom+factor]
                imageio.imwrite("no_"+str(no)+".jpg", cropped_neg, quality=100)
                no += 1
                i += 1

#
# STEP2 - Deciding on Features, Training and Testing the Cellphone Classifier
#

def Train_And_Test_Image_Classifier(split):
    phone_images=[]
    for image_file in [img_f for img_f in os.listdir(".")if img_f.startswith("yes") and img_f.endswith("jpg")]:
        image = imageio.imread(image_file)
        image = img_as_float(image)
        image = rgb2gray(image)
        image_prewitt = prewitt(image)
        phone_images.append([image, image_prewitt])

    n_phone_images = len(phone_images)
    #split phone images into training and testing sets
    factor_pi = int(n_phone_images /3)
    training_phone_images = []
    testing_phone_images = []

    if split == 0 :
    ##The last 1/3
        training_phone_images = phone_images[:factor_pi*2]
        testing_phone_images = phone_images[factor_pi*2:]
    elif split == 1:
    ##The first 1/3
        training_phone_images = phone_images[factor_pi:]
        testing_phone_images = phone_images[:factor_pi]
    else:
    ##The middle 1/3
        training_phone_images = phone_images[:factor_pi] + phone_images[factor_pi*2:]
        testing_phone_images = phone_images[factor_pi:factor_pi*2]

    non_phone_images=[]
    for image_file in [img_f for img_f in os.listdir(".")if img_f.startswith("no") and img_f.endswith("jpg")]: 
        image = imageio.imread(image_file)
        image = img_as_float(image)
        image = rgb2gray(image)
        image_prewitt = prewitt(image)
        non_phone_images.append([image, image_prewitt])
        
    
    n_non_phone_images = len(non_phone_images)
    #split none phone images into training and testing sets
    factor_npi = int(n_non_phone_images/3)
    training_non_phone_images = []
    testing_non_phone_images = []

    if split == 0:
    ##The last 1/3
        training_non_phone_images = non_phone_images[:factor_npi*2]
        testing_non_phone_images = non_phone_images[factor_npi*2:]
    elif split == 1:
    ##The first 1/3
        training_non_phone_images = non_phone_images[factor_npi:]
        testing_non_phone_images = non_phone_images[:factor_npi]
    else:
    ##The middle 1/3
        training_non_phone_images = non_phone_images[:factor_npi] + non_phone_images[factor_npi*2:]
        testing_non_phone_images = non_phone_images[factor_npi:factor_npi*2]

    training_set = training_phone_images + training_non_phone_images
    training_set_output = [1] * len(training_phone_images) + [0] * len(training_non_phone_images)

    testing_set = testing_phone_images + testing_non_phone_images
    testing_set_output = [1] * len(testing_phone_images) + [0] * len(testing_non_phone_images)

    n_training_set = len(training_set)
    training_set = np.array(training_set)
    training_set = training_set.reshape(n_training_set, -1)

    n_testing_set = len(testing_set)
    testing_set = np.array(testing_set)
    testing_set = testing_set.reshape(n_testing_set, -1)

    classifier = svm.SVC(C=100, probability=True, random_state=0)
    classifier.fit(training_set, training_set_output)
    pickle.dump(classifier, open("cellphone_image_classifier.sav", "wb"))

    predicted = classifier.predict(testing_set)
    print(classifier.score(testing_set, testing_set_output))

    i = 0
    while i < len(testing_set_output):
        if predicted[i] != testing_set_output[i]:
            print (i, predicted[i], testing_set_output[i])
        i += 1
    
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(testing_set_output, predicted)))

    predict_prob = classifier.predict_proba(testing_set)
    i = 0
    while i < len(testing_set_output):
        if predict_prob[i][0] > predict_prob[i][1]:
            if testing_set_output[i] != 0:
                print(i, "0: ",predict_prob[i][0], "1: ",predict_prob[i][1], " shouldbe:1")
        else:
            if testing_set_output[i] != 1:
                print(i, "0: ", predict_prob[i][0], "1: ", predict_prob[i][1], " shouldbe:0")
        i += 1
    print(" ")


#
#STEP 3 - Detecting a Cellphone in an Image
#

def Detect_Cellphone_In_Image(split):
    classifier = pickle.load(open("cellphone_image_classifier.sav", 'rb')) 
    df = pd.read_csv("find_phone/"+"labels.txt", sep=" ", header=None)
    df.columns = ["file_name", "x_coordinate", "y_coordinate"]

    if split == 0:
    ##The last 1/3
        df = df[86:]
    elif split == 1:
    ##The first 1/3
        df = df[:43]
    else:
    ##The middle 1/3
        df = df[43:86]

    found = 0
    total = 0
    for index, row in df.iterrows():
        image = imageio.imread("find_phone/"+row["file_name"])
        image = img_as_float(image)
        image = rgb2gray(image)

        x_val = range(3,430,10) #43
        y_val = range(5,260,10) #26
        x_coord = [ (i,i+64) for i in x_val]
        y_coord = [ (i,i+64) for i in y_val]
        box_coord = [ [i,j] for i in x_coord for j in y_coord]
    
        box_images = []
        for box_k in box_coord:
            xmin = box_k[0][0]
            xmax = box_k[0][1]
            ymin = box_k[1][0]
            ymax = box_k[1][1]
            img = image[ymin:ymax, xmin:xmax]
            img_prewitt = prewitt(img)
            box_images.append([img, img_prewitt])
        
        n_box_images = len(box_images)
        box_images = np.array(box_images)
        box_images = box_images.reshape(n_box_images, -1)    
        box_images_prediction = classifier.predict_proba(box_images)

        index = 0
        prob = 0

        for idx, prob_m in enumerate(box_images_prediction):
            if prob_m[1] > prob:
                prob = prob_m[1]
                index =idx

        xmin = box_coord[index][0][0]
        xmax = box_coord[index][0][1]
        ymin = box_coord[index][1][0]
        ymax = box_coord[index][1][1]
        x_predicted = float(int(xmax + xmin)/2)/490
        y_predicted = float(int(ymax + ymin)/2)/326
        x = float(row["x_coordinate"])
        y = float(row["y_coordinate"])

        total+=1
        if np.sqrt(((x_predicted-x)**2) + ((y_predicted-y)**2)) < 0.05:
            found += 1
        else:
            print("NO  ", row["file_name"]) 
             
        print(row["file_name"], "x:",row["x_coordinate"], "y:",row["y_coordinate"], "p:", prob, x_predicted, y_predicted)
        print(" ")
        
    print (found)
    print (total)
    print (found/total)

if __name__ =='__main__':
    factor = 32 #should be >= 1
    split = 0 #0, 1 or 2
    Process_Images(factor) #Each call will create new set of no images
    Train_And_Test_Image_Classifier(split)
    Detect_Cellphone_In_Image(split)
