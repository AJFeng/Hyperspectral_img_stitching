# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:27:51 2018

@author: Aijing Feng
"""
from __future__ import print_function
#from libtiff import TIFFfile, TIFFimage  #pylibtiff
#import matplotlib
#matplotlib.use('GTKAgg') #backends, need to run it first to force the plt.show() plot window appears before the code is done
import cv2
import numpy as np
#import matplotlib.pyplot as plt
import glob
import os
#import scipy.io
#import time
#import scipy.sparse as sparse
import math
import winsound





pp=0 #1620

font= cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale= 1
fontColor= (255,255,0)
lineType= 2

path = 'D:/.spyder-py3/tiff'
filename=glob.glob(os.path.join(path, '*.tiff'))
   
#f = open("matching M.txt","a") # w, r, a. if use r+, must read before write so that it can write after the original content. Or, it will replace!
#lines = f.readline()
#f.read() 
#f.write('hello boy')
#f.close()
  
def match_images(img1, img2,final_four_points_img2):
    """Given two images, returns the matches"""
    detector = cv2.xfeatures2d.SURF_create(4000, 3, 3,1,1)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    
    img3=img1[int(min(final_four_points_img2[1,:])):int(max(final_four_points_img2[1,:])),int(min(final_four_points_img2[0,:])):int(max(final_four_points_img2[0,:]))]
    kp1, desc1 = detector.detectAndCompute(img3, None)
    #cv2.imwrite("0005.jpg", img3)
    
    kp2, desc2 = detector.detectAndCompute(img2, None)
    #kp1[0].pt #the coodinate of the keypoint
    print(len(kp1))
    print(len(kp2))

    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) # k will be the dimension of each match 
    #dmatch.queryIdx: This attribute gives us the index of the descriptor in the list of query descriptors (in our case, it’s the list of descriptors in the img1).
    #dmatch.trainIdx: This attribute gives us the index of the descriptor in the list of train descriptors (in our case, it’s the list of descriptors in the img2).
    #dmatch.distance: This attribute gives us the distance between the descriptors. A lower distance indicates a better match.
    
    
    print(len(raw_matches))    
    
    kp_pairs= filter_matches(kp1, kp2, raw_matches,0.75)
    #print(len(list(kp_pairs)))
    M,r,sx,sy,error_min=RemoveBadMatching(kp_pairs,kp1,kp2,img3,img2,final_four_points_img2)
                  
    return M,r,sx,sy,kp1,kp2,error_min

def RemoveBadMatching(kp_pairs,kp1,kp2,img1,img2,final_four_points_img2):
    slope=[]
    distance=[]
    for f in range(len(kp_pairs)):
        x1,y1=np.array(kp1[kp_pairs[f][0].queryIdx].pt)
        x2,y2=np.array(kp2[kp_pairs[f][0].trainIdx].pt)
        slope.append((y2-y1)/(x2-x1))
        distance.append((x2 - x1)**2 + (y2 - y1)**2)
        
    slope=np.array(slope)
    good=kp_pairs.copy()
    
    for f in range(1,len(slope),1):
        
        #maxdiffSlop=max(slope)-min(slope)
        #maxdiffDiss=max(distence)-min(distence)
        #print("maxdiffSlop:"+str(maxdiffSlop))
        #print("maxdiffDiss:"+str(maxdiffDiss))
        MeanSlope=np.mean(slope)
        ratioSlope=slope/MeanSlope
        MeanDistance=np.mean(distance)
        ratioDistance=distance/MeanDistance
        
        distanceIdx=np.argsort(distance)
        sortIdx=np.argsort(slope) #smallest to largest
        #print(slope[sortIdx[len(sortIdx)//2]]-slope[sortIdx[len(sortIdx)//2-1]])
        #if (maxdiffSlop<((slope[sortIdx[len(sortIdx)//2]]-slope[sortIdx[len(sortIdx)//2-1]])*30)) and (maxdiffDiss<((distence[distenceIdx[len(distenceIdx)//2]]-distence[distenceIdx[len(distenceIdx)//2-1]])*100)) :
        
        if (((max(slope)-min(slope))<0.15) and ((max(distance)-min(distance))<30)) or (((max(ratioSlope)<1.1) and (min(ratioSlope)>0.9)) and ((max(ratioDistance)<1.09) and (min(ratioDistance)>0.9))) :
            M,r,sx,sy,error_min=getMatchMatric(kp1, kp2,good,final_four_points_img2)
            drawMyMatches(img1,kp1,img2,kp2,good)
            break
        
        elif (max(ratioDistance)>1.1) or (min(ratioDistance)<0.9):
            leftDissDiff=distance[distanceIdx[len(distanceIdx)//2]]-distance[distanceIdx[0]]
            rightDissDiff=distance[distanceIdx[-1]]-distance[distanceIdx[len(distanceIdx)//2]]
            if (rightDissDiff>leftDissDiff):
                del good[distanceIdx[-1]]
                slope=np.delete(slope, distanceIdx[-1], 0)
                distance=np.delete(distance, distanceIdx[-1], 0)
            elif (rightDissDiff<leftDissDiff):
                del good[distanceIdx[0]]
                slope=np.delete(slope, distanceIdx[0], 0)
                distance=np.delete(distance, distanceIdx[0], 0)
                
        elif ((max(ratioSlope)>1.1) or (min(ratioSlope)<0.9)):
            leftSlopDiff=slope[sortIdx[len(sortIdx)//2]]-slope[sortIdx[0]]
            rightSlopDiff=slope[sortIdx[-1]]-slope[sortIdx[len(sortIdx)//2]]
            #print(sortIdx[-1])
            if rightSlopDiff>leftSlopDiff:
                del good[sortIdx[-1]]
                slope=np.delete(slope, sortIdx[-1], 0)
                distance=np.delete(distance, sortIdx[-1], 0)
            else:
                del good[sortIdx[0]]
                slope=np.delete(slope, sortIdx[0], 0)
                distance=np.delete(distance, sortIdx[0], 0)
   
        
        if f>len(list(kp_pairs))*0.5:   # need to vilidate manully 
            #kp_pairs2=kp_pairs[0:3]
            #global pp
            #cv2.imwrite(str(pp)+"match2.jpg", img6)
            #pp=pp+1
            #plt.imshow(img6),plt.show()
            
            M,r,sx,sy,error_min=getMatchMatric(kp1, kp2,good, final_four_points_img2)
            
            if error_min>300:  #20
                winsound.Beep(800, 1500)
                drawMyMatches(img1,kp1,img2,kp2,good,1,final_four_points_img2)
            
                result = input("How is the result? 1.good 2.remove bad matches 3. select good matches")
                M,r,sx,sy,good,error_min=matchesResult(result, kp1, kp2,good,final_four_points_img2)
            
                drawMyMatches(img1,kp1,img2,kp2,good)
            
                break
            else:
                drawMyMatches(img1,kp1,img2,kp2,good)
                break 
        
    return M,r,sx,sy,error_min


def drawMyMatches(img1,kp1,img2,kp2,good,flag=0,final_four_points_img2=[0,0,0,0]):
    global pp
    img7 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    for f in range(len(list(good))):
         pt=np.array(kp1[good[f][0].queryIdx].pt)
         #print("kp1,"+str(f)+":"+str(pt))
         bottomLeftCornerOfText = (int(pt[0]),int(pt[1]))
         cv2.putText(img7,str(f), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
         pt=np.array(kp2[good[f][0].trainIdx].pt)
         bottomLeftCornerOfText = (int(pt[0])+img1.shape[1],int(pt[1]))
         #print("kp2,"+str(f)+":"+str(bottomLeftCornerOfText))
         cv2.putText(img7,str(f), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
    
    #plt.imshow(img7),plt.show()     
    
    if flag==0: #good result
        cv2.imwrite("D:/matching result/goodResult"+str(pp)+".jpg", img7)
        pp=pp+1
    elif flag==1:  #result needs to be validated
        img8=img7[max(int(min(final_four_points_img2[1,:])-50),0):min(int(max(final_four_points_img2[1,:])+50),img7.shape[0]),max(int(min(final_four_points_img2[0,:])-50),0):min(int(max(final_four_points_img2[0,:])+50),img7.shape[1])]
        cv2.imwrite("02.jpg", img7)

    else:
        print("Wrong flag value!")
    
    return

def filter_matches(kp1, kp2, matches, ratio = 1):

    good = []
    for m,n in matches:
       if m.distance <= ratio*n.distance:  #The main advantage using knnMatch is that you can perform a ratio test. So if the distances from one descriptor in descriptor1 to the two best descriptors in descriptor2 are similar it suggests that there are repetitive patterns in your images (e.g. the tips of a picket fence in front of grass). Thus, such matches aren't reliable and should be removed.
          good.append([m])
    
    print('good:'+str(len(list(good))))
    
    return good

def getDistance(elem):
    return elem[0].distance

def rotationFit(src_pts_0,dst_pts_0,x0,y0):
        
    M=np.zeros((2,3))
    src_pts=np.matrix.transpose(np.matrix(np.array(src_pts_0)))
    src_pts=np.vstack((src_pts,np.ones((1,src_pts.shape[1]))))
    error_min=math.inf
    best_fit=np.array([(0,0,0),(0,0,0)])
    best_r=0
    best_sx=0
    best_sy=0
    
    #test = cv2.getAffineTransform(src_pts_0[0:3,:],dst_pts_0[0:3,:])
    #print(test)
    
    r=[-0.0000873, -0.0001745, -0.0002618, -0.000349, -0.000436, -0.00524, -0.00061, -0.0007, -0.0007854,  -0.00087, -0.00096, -0.00105, -0.0011345, 0, 0.0000873, 0.0001745, 0.0002618, 0.000349, 0.000436, 0.00524, 0.00061, 0.0007, 0.0007854,  0.00087, 0.00096, 0.00105, 0.0011345] #degree of 0.005, 0.01, 0.015, 0.02,0.025, 0.03, 0.04, 0.05 to radian
    #r=[0]
    for kx0 in np.linspace(-10,10,41): # np.linspace(-10,10,101) cannot use range() here, range() is only for interger
        M[0,2]=-x0+kx0
        for ky0 in np.linspace(-10,10,41): #np.linspace(-10,10,101)
            M[1,2]=-y0+ky0
            for j in r:
                for sx in [1]:  #[0.9995,0.99975,1,1.00025,1.0005]
                    for sy in [1]:
                        cos_r=math.cos(j)
                        sin_r=math.sin(j)
                        M[0,0]=cos_r*sx
                        M[0,1]=-sin_r
                        M[1,0]=sin_r
                        M[1,1]=cos_r*sy
                
                        dst_pts_estimate=np.matmul(M,src_pts)
                        error=np.sum(np.absolute(np.subtract(dst_pts_0,np.transpose(dst_pts_estimate))))
                
                        if error<error_min:
                            error_min=error
                            best_fit=M.copy()
                            best_r=j
                            best_sx=sx
                            best_sy=sy
                            
    print(error_min)
    return best_fit,best_r,best_sx,best_sy,error_min
                
                
                

def getMatchMatric(kp1, kp2,kp_pairs,final_four_points_img2):
    
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in kp_pairs ])  #Be aware that these coodinates are after the cut from final_four_points_img2.
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in kp_pairs ])
     
    src_pts[:,0]=np.add(src_pts[:,0], int(min(final_four_points_img2[0,:])))
    src_pts[:,1]=np.add(src_pts[:,1], int(min(final_four_points_img2[1,:])))
    
    x0=int(round(np.mean(src_pts[:,0]-dst_pts[:,0]))) #just get the x and y trandform firstly, but the best fit would change in the rotationFit
    y0=int(round(np.mean(src_pts[:,1]-dst_pts[:,1])))
 
    #M=np.ones((1,2))
    #M[0,0]=x0 
    #M[0,1]=y0
    
    M,r,sx,sy,error_min=rotationFit(src_pts,dst_pts,x0,y0)
   
    return M,r,sx,sy,error_min


def matchesResult(result, kp1, kp2,kp_pairs):
    
    if result=='1':
        good=kp_pairs
        M,r,sx,sy,error_min=getMatchMatric(kp1, kp2,kp_pairs)
        
    elif result=='2':
        good=kp_pairs
        #num=input("How many matches do you want to remove?")
        
        r= [int(x) for x in input("Which do you want to remove? Enter lager number firstly").split()]
        for n in range(len(r)):
            del good[r[n]]           
        
        M,r,sx,sy,error_min=getMatchMatric(kp1, kp2, good)
        
    elif result=='3':
        good=[]
        
        r= [int(x) for x in input("Which matches do you want to choose?").split()]
        for n in range(len(r)):
            good.append(kp_pairs[r[n]])
        
        M,r,sx,sy,error_min=getMatchMatric(kp1, kp2,good)
    else:
        print("Wrong input! please check your matchesResult again!")
        
    return M,r,sx,sy,good,error_min
    
def dynamicPano(img,img2,M,startBand=1,endBand=103,cut=0):
    
    """
    ####################if no rotation, this code works better. Warp makes the images had low resolution.###################################
    
    M[0,0]=-int(M[0,2])
    M[0,1]=-int(M[1,2])
    
    imgTemp=np.zeros((abs(int(M[0,1])),img.shape[1]), dtype=int)
    #if M[0,1]>0: 
       #img=np.vstack((img,imgTemp)) # new image move down
    if M[0,1]<0:
       img=np.vstack((imgTemp,img))  # move up
       
    if (abs(int(M[0,0]))+2048-img.shape[1])>0:
        if M[0,0]>0:
            imgTemp=np.zeros((img.shape[0],(abs(int(M[0,0]))+2048-img.shape[1])), dtype=int)
            img=np.hstack((img,imgTemp))  # move right
    if M[0,0]<0:
       imgTemp=np.zeros((img.shape[0],(abs(int(M[0,0])))), dtype=int)
       img=np.hstack((imgTemp,img))  # fly left

    #plt.imshow(img),plt.show()
    img=img.astype(np.uint8)
    if (M[0,0]<0) and (M[0,1]<0):  # stitch in left and up
       img[(startBand-1)*8:(startBand-1)*8+img2.shape[0], 0:0+img2.shape[1]] = img2
    elif (M[0,0]<0) and (M[0,1]>0):  # stitch in left and down
       img[(startBand-1)*8+int(M[0,1]):(startBand-1)*8+int(M[0,1])+img2.shape[0], 0:0+img2.shape[1]] = img2
    elif (M[0,0]>0) and (M[0,1]<0):  # stitch in right and up
       #img[0:0+img2.shape[0], int(M[0,0]):int(M[0,0])+img2.shape[1]] = img2
       img[(startBand-1)*8:(startBand-1)*8+img2.shape[0], int(M[0,0]):int(M[0,0])+img2.shape[1]] = img2
    elif (M[0,0]>0) and (M[0,1]>0):  # stitch in right and down
       #img[int(M[0,1]):int(M[0,1])+img2.shape[0], int(M[0,0]):int(M[0,0])+img2.shape[1]] = img2
       img[(startBand-1)*8+int(M[0,1]):(startBand-1)*8+int(M[0,1])+img2.shape[0], int(M[0,0]):int(M[0,0])+img2.shape[1]] = img2
    
    return img 
    ################################################################################################################################
    """
    
    
    M[0,2]=-M[0,2]
    M[1,2]=-M[1,2]
    
    #four_points_img2=np.array(np.transpose([(0,0),(0,2048),(1088,0),(1088,2048)]))       
    #four_points_img2=np.vstack((four_points_img2,np.ones((1,four_points_img2.shape[1]))))
    four_points_img2=np.array([(0,2048,0,2048),(0,0,1088,1088),(1,1,1,1)])
    
    new_four_points_img2=np.matmul(M,four_points_img2)
    extend_y_down=int(max(new_four_points_img2[1,:])-img.shape[0])
    extend_y_up=int(-min(new_four_points_img2[1,:]))
    extend_x_right=int(max(new_four_points_img2[0,:])-img.shape[1])
    extend_x_left=int(-min(new_four_points_img2[0,:]))
    
    
    anchorX, anchorY = 0, 0
    
    if extend_y_up>0:            #need to know how the background map extend
        imgTemp=np.zeros((extend_y_up,img.shape[1]), dtype=int)
        img=np.vstack((imgTemp,img))  # extend up
        anchorY=extend_y_up
        
    if extend_y_down>0:
        imgTemp=np.zeros((extend_y_down,img.shape[1]), dtype=int)
        img=np.vstack((img,imgTemp))  # extend down
        
    if extend_x_right>0:
        imgTemp=np.zeros((img.shape[0],extend_x_right), dtype=int)
        img=np.hstack((img,imgTemp))  # extend right
        
        
    if extend_x_left>0:
        imgTemp=np.zeros((img.shape[0],extend_x_left), dtype=int)
        img=np.hstack((imgTemp,img))  # extend left
        anchorX=extend_x_left
        
    dst_pad = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
   
    
    #cv2.imwrite("D:/.spyder-py3/0004.jpg", img2)
    
    if cut==0:
        dst_pad[anchorY+(startBand-1)*8:anchorY+(startBand-1)*8+img2.shape[0], anchorX:anchorX+2048] = img2
        final_four_points_img2=np.matmul(M,np.array([(anchorX,anchorX+2048,anchorX,anchorX+2048),(anchorY+(startBand-1)*8,anchorY+(startBand-1)*8,anchorY+(startBand-1)*8+img2.shape[0],anchorY+(startBand-1)*8+img2.shape[0]),(1,1,1,1)]))
        
    if cut==1:
        dst_pad[anchorY+(startBand-1)*8+132:anchorY+(startBand-1)*8+132+img2.shape[0], anchorX:anchorX+2048] = img2
        final_four_points_img2=[0,0,0,0]
     
    #cv2.imwrite("D:/.spyder-py3/000.jpg",dst_pad)
    
    
    img2_rotation = cv2.warpAffine(dst_pad,M,(img.shape[1],img.shape[0]))  
    #cv2.imwrite("D:/.spyder-py3/001.jpg",img2_rotation)
    
        #######################################the warp creat some black line in the bondary of the images. The following is trying to remove the black lines##############
    flag1=math.inf
    
    if cut==0:
        
        for v in range(max(anchorY+(startBand-1)*8+int(M[1,2])-5,0),min(anchorY+(startBand-1)*8+int(M[1,2])+5,dst_pad.shape[0]),1):
            for w in [anchorX+int(M[0,2])-1,anchorX+int(M[0,2]),anchorX+int(M[0,2])+1,anchorX+int(M[0,2])+2048-2,anchorX+int(M[0,2])+2048-1,anchorX+int(M[0,2])+2048]:
                if (w<img2_rotation.shape[1]-1) and (img2_rotation[v,w]>10) and (v<flag1):
                    flag1=v
                    continue
    if cut==1:
        for v in range(max(anchorY+(startBand-1)*8+132+int(M[1,2])-5,0),min(anchorY+(startBand-1)*8+132+int(M[1,2])+5,dst_pad.shape[0]),1):
            for w in [anchorX+int(M[0,2])-1,anchorX+int(M[0,2]),anchorX+int(M[0,2])+1,anchorX+int(M[0,2])+2048-2,anchorX+int(M[0,2])+2048-1,anchorX+int(M[0,2])+2048]:
                if (w<img2_rotation.shape[1]-1) and (img2_rotation[v,w]>10) and (v<flag1):
                    flag1=v
                    continue
        
    
    
    img2_rotation[0:flag1+2,:]=np.zeros((flag1+2,img2_rotation.shape[1]))
    img2_rotation[flag1+img2.shape[0]-2:img2_rotation.shape[0]-1,:]=np.zeros((img2_rotation.shape[0]-flag1-img2.shape[0]+1,img2_rotation.shape[1]))           
    
    #cv2.imwrite("D:/.spyder-py3/002.jpg",img2_rotation)
    
    flag2=math.inf
    for v in range(max(anchorX+int(M[0,2])-10,0),min(anchorX++int(M[0,2])+10,dst_pad.shape[1]),1):
        for w in [flag1-1,flag1,flag1+1,flag1+2,flag1+3,flag1+img2.shape[0]-3, flag1+img2.shape[0]-2,flag1+img2.shape[0]-1,flag1+img2.shape[0],flag1+img2.shape[0]+1]:
            if (w<img2_rotation.shape[0]-1) and (img2_rotation[w,v]>10) and (v<flag2):
                flag2=v
                continue
            
    img2_rotation=np.asmatrix(img2_rotation)
    
    if flag2!=math.inf:
       img2_rotation[:,0:flag2+1]=np.zeros((img2_rotation.shape[0],flag2+1))
       img2_rotation[:,flag2+img2.shape[1]-2:img2_rotation.shape[1]]=np.zeros((img2_rotation.shape[0],1))
    
    #cv2.imwrite("D:/.spyder-py3/003.jpg",img2_rotation)
    ###################################################################################################################################################################
    
    
    img2_rotation[np.where(img2_rotation <10)] = img[np.where(img2_rotation <10)]
    
    #cv2.imwrite("F:/.spyder-py3/001.jpg",img2_rotation)
            
    
    return img2_rotation,final_four_points_img2



def SticthCutting(startBand,endBand):
    f = open("matching M.txt","r")
    pos = f.tell()
    flag=0;
    while True:
        
        lines = f.readline() # read the whole line
        #print(lines)
        
        
        newpos = f.tell()
        if newpos == pos:  # stream position hasn't changed -> EOF
            f.close()
            return
        else:
            pos = newpos
                
        M=np.ones((2,3))         
        imgID,M[0,0],M[0,1],M[1,0],M[1,1],M[0,2],M[1,2],r,sx,sy,error_min = [float(i) for i in lines.split()]
        imgID=int(imgID)
        
        if flag==0:
           img=cv2.imread(filename[imgID-1], 0)
           #img=img[(startBand-1)*8:endBand*8,:]
           f.seek(0,0)
           pos = f.tell()
           flag=1;
           continue
           
        img2=cv2.imread(filename[imgID], 0)
        #img2=img2[(startBand-1)*8+132:endBand*8+132,:]
        img2=img2[(startBand-1)*8:endBand*8,:]
        
        img,final_four_points_img2=dynamicPano(img,img2,M,startBand,endBand,1)
        
        if imgID==2138:
            cv2.imwrite("D:/.spyder-py3/matching result/band"+str(startBand)+"-"+str(endBand)+" "+str(imgID)+".png", img,[cv2.IMWRITE_PNG_COMPRESSION,0])
    
        
        
        if not lines:
           break
        pass
    

############### Test ###############



img=cv2.imread(filename[0], 0)


#t0 = time.time()

f = open("D:/.spyder-py3/matching M.txt","a")

final_four_points_img2=np.array([(0,2048,0,2048),(0,0,1088,1088)])


for i in range(1,len(filename)-1,1): #range(len(filename)-1):  range(1620,1750,1)
    print(i)
    img2 = cv2.imread(filename[i], 0)
    M,r,sx,sy,kp1,kp2,error_min= match_images(img, img2,final_four_points_img2)
    f.write(str(i)+" "+str(M[0,0])+" "+str(M[0,1])+" "+str(M[1,0])+" "+str(M[1,1])+" "+str(M[0,2])+" "+str(M[1,2])+" "+str(r)+" "+str(sx)+" "+str(sy)+" "+str(error_min)+"\n")
    
    img,final_four_points_img2=dynamicPano(img,img2,M);      
    cv2.imwrite("D:/.spyder-py3/matching result/"+str(i)+".jpg", img)

#t1=time.time() - t0
#print(t1)
f.close()


"""
SticthCutting(25,29)  #red-edge
SticthCutting(14,17)  #red
SticthCutting(56,60)   #nir
"""
"""
SticthCutting(1,5)  
SticthCutting(6,10)
SticthCutting(11,15)
SticthCutting(16,20)
SticthCutting(21,25)
SticthCutting(26,30)
SticthCutting(31,35)
SticthCutting(36,40)
SticthCutting(41,45)
SticthCutting(46,50)
SticthCutting(51,55)
SticthCutting(56,60)
SticthCutting(61,65)
SticthCutting(66,70)
SticthCutting(71,75)
SticthCutting(76,80)
SticthCutting(81,85)
SticthCutting(86,90)
SticthCutting(91,95)
SticthCutting(96,100)
SticthCutting(99,103)

"""
#SticthCutting(1,136)

#winsound.Beep(600, 500)
