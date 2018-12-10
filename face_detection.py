import cv2

#insatncié un calassifier qui détecte le visage
faceDetect=cv2.CascadeClassifier('data\haarcascades\haarcascade_frontalface_default.xml')
#utiliser la cam pour récuperer des images et on peux mettre 1 ou 2 a la place de zéro si on a plus d'une cam
cam=cv2.VideoCapture(0)
#la boucle est pour récupérer une vidéo
while (True):
    #lire une image à partire de la cam qui sera stocké dans la variable img et status est un bolléan qui indique si on a pu lire l'image donc il est à true si non à flase
    status,img=cam.read()
    #Détecte des objets de différentes tailles dans l'image d'entrée. Les objets détectés sont renvoyés sous forme de liste de rectangles.
    faces=faceDetect.detectMultiScale(img,1.3,5)
    #tracer le rectangle qui englobe le visage
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),1)

    cv2.imshow("firstLive", img)
    #chaque 1ms un nouveau frame doit apparaitre
    key=cv2.waitKey(1)
    # quand l'utilisateur click sur la touche q on va devoir fermer la frame qui souvre
    if key==ord("q"):
        break
#supprime le frame cree
cv2.destroyWindow("firstLive")
#libirer la cam
cam.release()