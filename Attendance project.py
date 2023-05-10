#1import libraries
import cv2
import numpy as np
import face_recognition
import os

#2convert images into RGB
#find image folder and the image no. and their encodings


#store images as list
path='images'
imageslist=[]#list of all images that we import
names=[]#names of all the images to print at time of output  did so not to enter the image name manually
mylist=os.listdir(path)#first we grab the list of images in the image folder
print(mylist)
#to remove extension
#to use the names and import the images one by one
for cls in mylist:
    currentimage=cv2.imread(f'{path}/{cls}')
    imageslist.append(currentimage)#adding images
    names.append(os.path.splitext(cls)[0])#adding names without extension
print(names)


#encoding process begin
#creating a function to find all the encodings
def findencodings(imgs):
    encodelist=[]
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#converting to RGB
        encode=face_recognition.face_encodings(img)[0]#getting the encoding of the image
        encodelist.append(encode)
    return encodelist

encodedlist=findencodings(imageslist)
print("encoding completed")

#find the match with the image
#getting the image to compare  from the camera
capture=cv2.VideoCapture(0)#to capture the video  ,given 0 as out ID
while True:
    success,img=capture.read()#getting our image
    #resize the image to reduce the time of processing in realtime
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)#resizing
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)#converting to RGB

    facescurrentframe=face_recognition.face_locations(imgs)#to get locations of the faces if there are more than one face
    encodecurrentframe = face_recognition.face_encodings(imgs,facescurrentframe)#encoding  the webcam image
    #finding the matches and comparing
    for encodeface,faceloc in zip(encodecurrentframe,facescurrentframe):#use zip to use both in same loop ,it grab face and encoding one by one from both side
        matches=face_recognition.compare_faces(encodedlist,encodeface)#comparing the every known faces in encoded list to  one  frame  from encodeface
        facedistance=face_recognition.face_distance(encodedlist,encodeface)
        #for each frame above code will give a list in return since we give in a list as input for the distance function above
        #the one with the lowest distance in the given list is exactly that person
        matchIndex=np.argmin(facedistance)#gives the index of the lowest

        #since we know who is that best match person using the index we just need to  print a box around that person and display their name
        if matches[matchIndex]:
            name=names[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceloc
            #since we cut downour size so the location will be changed  inorder to get the right
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)#coordinates,color,thickness
            cv2.rectangle(img, (x1, y2-35), (x2, y2),(0,255,0), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


        #to display the image
        cv2.imgshow('webcam',img)
        cv2.waitkey(1)






