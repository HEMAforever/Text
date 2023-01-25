# Text
#Recognising a photo text
import cv2
import numpy as np

def detect(img):
    percent_white_pix = 0
    letter = -1
    for i, d in enumerate(letters):# проходимся по массиву букв и ищем в нём наиболее подходящий элемент
        scaled_img = cv2.resize(img, d.shape[:2][::-1])
        #d AND (scaled_img XOR d)
        bitwise = cv2.bitwise_and(d, cv2.bitwise_xor(scaled_img, d))
        # результат определяется наибольшей потерей белых пикселей
        before = np.sum(d == 255)
        matching = 100 - (np.sum(bitwise == 255) / before * 100)
        #cv2.imshow('digit_%d' % (9-i), bitwise)
        if percent_white_pix < matching:
            percent_white_pix = matching
            letter = i
    return letter

SCALE = 3
THICK = 5
WHITE = (255, 255, 255)
letters = []
dit = dict()
ar = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
i = 0
for letter in ar: #запоминаем, как будут выглядеть буквы
    (width, height), bline = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX,
                                             SCALE, THICK) #нашли размер текста
    letters.append(np.zeros((height + bline, width), np.uint8))
    cv2.putText(letters[-1], letter, (0, height), cv2.FONT_HERSHEY_SIMPLEX,
                SCALE, WHITE, THICK)
    x0, y0, w, h = cv2.boundingRect(letters[-1])
    letters[-1] = letters[-1][y0:y0+h, x0:x0+w]
    dit[i] = letter
    i += 1
    #cv2.imshow("Letter", letters[-1])
    # Задерживаем программу до нажатия на клавишу
    #cv2.waitKey(0)

txt = cv2.imread("numpy.jpg") #берём файл, пишем в кавычках название файла
cl_gray = cv2.cvtColor(txt, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(cl_gray, 170, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
ans = []

for cnt in contours:
    # проверяется размер контура, чтобы избежать обработки "дефекта".
    if cv2.contourArea(cnt) > 30:
        # получаем прямоугольник, окружающий число
        brect = cv2.boundingRect(cnt)
        x, y, w, h = brect
        roi = thresh[y:y+h, x:x+w]
        # определение
        numb = detect(roi)
        if numb == -1:
            ans.append(" ")
        else:
            letter = dit[numb]
            ans.append(letter)
            cv2.rectangle(txt, brect, (0, 255, 0), 2)
            cv2.putText(txt, letter, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (190, 123, 68), 2)
cv2.imshow('resultat', txt)
cv2.waitKey(0)
print(*ans)
