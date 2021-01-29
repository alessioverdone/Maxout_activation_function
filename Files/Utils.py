import numpy as np
import time
import os
from PIL import Image



def preprocess_images(path,debug):
    tm = time.time()
    dataset_lr = list()
    dataset_hr = list()
    crop_list=list()
    if debug:
        iterations=1
    else:
        iterations = 5
    '''Low resolution images'''
    for image in os.listdir(path +'/LR_img'):
        for i in range(iterations):
            img = Image.open(path +'/LR_img/'+ image).convert('YCbCr')
            rand_point=np.random.randint(24,530,1)#558x558 max size img(x2)
            rand_point=rand_point[0]
            crop_list.append(rand_point)
            cropDim=(rand_point-24,rand_point-24,rand_point,rand_point)
            img = img.crop(cropDim)#24x24  
            '''Rotate image for data augmentation'''
            if i % 4 == 0:
                img=img.rotate(30)
            imVect = np.array(img)
            '''Flipping image or data augmentation'''
            if i % 3 ==0:
                imVect = np.fliplr(imVect) 
            imVect = (imVect /255)-0.5#Normalize + zero mean
            imVect = imVect[:,:,0]#Take only the y channel
            dataset_lr.append(imVect)
    cont=-1
    '''High resolution images'''
    for image in os.listdir(path +'/HR_img'):
        for i in range(iterations):
            cont+=1
            img = Image.open(path +'/HR_img/'+ image).convert('YCbCr')
            rand_point = crop_list[cont]
            cropDim2=(rand_point-24,rand_point-24,rand_point+24,rand_point+24)
            img = img.crop(cropDim2)#24x24
            '''Rotate image for data augmentation'''
            if i % 4 == 0:
                img=img.rotate(30)
            imVect = np.array(img)
            '''Flipping image or data augmentation'''
            if i %3 ==0:
                imVect = np.fliplr(imVect)
            imVect = (imVect /255)-0.5#Normalize + zero mean
            imVect = imVect[:,:,0]#Take only the y channel and make the error based on this
            dataset_hr.append(imVect)
    print('Dataset uploaded ' + ' Time:'+ str(time.time()-tm) + ' sec')
    
    #Input set is composed by 100 images,after data augmentation they become 100*iterations
    X_train, X_val, X_test = dataset_lr[0:400], dataset_lr[400:450], dataset_lr[450:500]
    Y_train, Y_val, Y_test = dataset_hr[0:400], dataset_hr[400:450], dataset_hr[450:500]
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
