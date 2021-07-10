import cv2
import numpy as np

def get_Points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        P.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), thickness = -1)
        cv2.imshow(window_name, img)


def getPerspectiveTransform(src, dst):

    A = []
    b = []

    for i in range(4):
        x = src[i, 0]
        y = src[i, 1]
        Px = dst[i, 0]
        Py = dst[i, 1]
        # Para as coordenadas x: Px
        aux_A = [-x, -y, -1, 0, 0, 0, Px*x, Px*y] 
        aux_b = -Px

        A.append(aux_A)
        b.append(aux_b)

        # Para as coordenadas y: Py
        aux_A = [0, 0, 0, -x, -y, -1, Py*x, Py*y]
        aux_b = -Py

        A.append(aux_A)
        b.append(aux_b)

    A = np.asarray(A)
    b = np.asarray(b)

    M = np.linalg.solve(A, b)
    M = np.append(M, [1])
    M = np.reshape(M, (3, 3))
   
    return M


def perspectiveWarp(img, M, new_shape):
    
    shape = np.asarray(new_shape)
    h = shape[0]
    w = shape[1]
    M_inv = np.linalg.inv(M)

    warped = np.zeros(new_shape, dtype = np.uint8)

    for Px in range(w):
        for Py in range(h):
            w_ = 1 / np.sum((M_inv[2, :] * np.array([Px, Py, 1])))
            P_warped = np.array([[Px * w_], 
                                 [Py * w_],
                                 [w_]])
            
            
            p = M_inv @ P_warped
            x = int(p[0,0])
            y = int(p[1,0])
            try:
                warped[Py, Px, :] = img[y, x, :]
            except:
                pass

    return warped


name = "img_examples/2.jpeg"
img = cv2.imread(name)

# Resize
i = 0.5
(h, w, c) = img.shape
img = cv2.resize(img, (int(i*w), int(i*h)))

# Obter pixels
P = []
window_name = "original"

while True:
    # Se ja pegou todos os 4 pixels
    if len(P) == 4:
        # Selecao dos pixels
        A = P[0]
        B = P[1]
        C = P[2]
        D = P[3]


        src = np.asarray([A,B,C,D], dtype = "float32")

        # Tamanhos
        widthA = np.sqrt(((C[0] - D[0]) ** 2) + ((C[1] - D[1]) ** 2))
        widthB = np.sqrt(((B[0] - A[0]) ** 2) + ((B[1] - A[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((B[0] - C[0]) ** 2) + ((B[1] - C[1]) ** 2))
        heightB = np.sqrt(((A[0] - D[0]) ** 2) + ((A[1] - D[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([[0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype = "float32")


        M = getPerspectiveTransform(src, dst)
        warped = perspectiveWarp(img, M, (maxHeight, maxWidth, 3))


        M_cv2 = cv2.getPerspectiveTransform(src, dst)
        warped_cv2 = cv2.warpPerspective(img, M_cv2, (maxWidth, maxHeight))
        
        cv2.imshow("warp", warped)
        cv2.imshow("warped_cv2", warped_cv2)
        break
    
    cv2.imshow(window_name, img)
    cv2.setMouseCallback(window_name, get_Points)
    cv2.waitKey(1)

cv2.waitKey(0)
cv2.destroyAllWindows()