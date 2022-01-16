
import numpy as np
import cv2 as cv
import Person

cnt_in   = 0
cnt_out = 0

cap = cv.VideoCapture('Test Files/testvideo.mp4')

h = 540
w = 724
frameArea = h*w
areaTH = frameArea/10

line_in = int(2*(h/5))
line_out   = int(3*(h/5))

in_limit =   int(1*(h/5))
out_limit = int(4*(h/5))

fgbg = cv.createBackgroundSubtractorMOG2(detectShadows = True)

#Structuring elements for morphological filters
kernelOp = np.ones((10,10),np.uint8)
kernelCl = np.ones((25,25),np.uint8)

#Variables

persons = []
pid = 1

while(cap.isOpened()):
    ret, frame = cap.read()
    
    fgmask = fgbg.apply(frame)

    try:
        ret,imBin= cv.threshold(fgmask,200,255,cv.THRESH_BINARY)
        #Opening (erode->dilate) to remove noise.
        mask = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernelOp)
        #Closing (dilate -> erode) to join white regions.
        mask =  cv.morphologyEx(mask , cv.MORPH_CLOSE, kernelCl)
    except:
        print('EOF')
        print( 'UP:',cnt_in)
        print ('DOWN:',cnt_out)
        break

    contours0, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv.contourArea(cnt)
        if area > areaTH:
            
            # tracking

            M = cv.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv.boundingRect(cnt)

            new = True
            if cy in range(in_limit,out_limit):
                for i in persons:
                    if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h:
                        new = False
                        i.updateCoords(cx,cy) 
                        if i.going_UP(line_out,line_in) == True:
                            cnt_in += 1
                        elif i.going_DOWN(line_out,line_in) == True:
                            cnt_out += 1
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > out_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < in_limit:
                            i.setDone()
                    if i.timedOut():
                        index = persons.index(i)
                        persons.pop(index)
                        del i
                if new == True:
                    p = Person.MyPerson(pid, cx, cy)
                    persons.append(p)
                    pid += 1     

            cv.circle(frame,(cx,cy), 5, (0,0,255), -1)
            img = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    str_in = 'IN: '+ str(cnt_in)
    str_out = 'OUT: '+ str(cnt_out)


    # Display

    cv.line(frame, (0, line_in), (1024, line_in), (0, 0, 255), 2)
    cv.line(frame, (0, line_out), (1024, line_out), (255, 0, 0), 2)
    cv.line(frame, (0, in_limit), (1024, in_limit), (255, 255, 255), 2)
    cv.line(frame, (0, out_limit), (1024, out_limit), (255, 255, 255), 2)

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, str_in ,(10,40),font,0.5,(255,255,255),2,cv.LINE_AA)
    cv.putText(frame, str_in ,(10,40),font,0.5,(0,0,255),1,cv.LINE_AA)
    cv.putText(frame, str_out ,(10,90),font,0.5,(255,255,255),2,cv.LINE_AA)
    cv.putText(frame, str_out ,(10,90),font,0.5,(255,0,0),1,cv.LINE_AA)

    cv.imshow('Mask',mask)   
    cv.imshow('Frame',frame)
     

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()

cv.destroyAllWindows()
