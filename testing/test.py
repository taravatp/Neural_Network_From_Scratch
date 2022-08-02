from cv2 import cv2
import pickle
from hand_pipe import hands
import numpy as np
import argparse
import sys
import time

labels = 'ABCDEFGHIJ'

with open('parameters.pickle', 'rb') as handle:
    parameters = pickle.load(handle)

layers = [42, 100, 10]
neuron_values = {}
activations = ['None', 'sigmoid', 'softmax']


def linear_layer(X, W, b):
    z = np.dot(X, W) + b
    return z


def ReLU(x):
    a = np.max(x, 0)
    return a


def sigmoid(x):
    a = 1 / (1 + np.exp(-x))
    return a


def softmax(x):
    temp = np.sum(np.exp(x), axis=1)
    temp = temp[..., np.newaxis]  # i did this for removing the broadcasting error
    a = np.exp(x) / temp  # dar rastaye satr jam mikonad
    return a


def feedforward(X, parameters):
    global neuron_values
    neuron_values['linear' + str(0)] = X
    neuron_values['activation' + str(0)] = X

    for i in range(1, len(layers)):
        A_previous = X

        X = linear_layer(A_previous, parameters['w' + str(i)],
                         parameters['b' + str(i)])  # dar khuruji har satr baraye yek data ast
        neuron_values['linear' + str(i)] = X

        if (activations[i] == 'sigmoid'):
            X = sigmoid(X)
            neuron_values['activation' + str(i)] = X

        elif (activations[i] == 'softmax'):
            X = softmax(X)
            neuron_values['activation' + str(i)] = X

        else:
            pass

    return X


def get_model_pred(X):
    # ------------------------------------------------
    # you should replace this code with your own code:
    # you should use X which is a 1*42 Matrix as input
    # and compute output.
    # your output should be a 1*10 matrix or 10 element vector
    # containing the softmax probabilty outputs.
    # remove the code bellow and put your own code here
    pred = feedforward(X, parameters)
    # you should replace this code with your own code
    # ------------------------------------------------
    return pred


def get_frame_label(x_marks, y_marks):
    y_marks = (y_marks - y_marks.mean()) / y_marks.std()
    x_marks = (x_marks - x_marks.mean()) / x_marks.std()

    X = np.hstack((x_marks, y_marks)).reshape(1, -1)

    pred = get_model_pred(X)

    pred, prob = pred.argmax(), pred.max()

    pred_char = labels[pred]
    if prob < 0.5:
        return 'nothing', 0

    return pred_char, prob


def get_hand_image(img, x_marks, y_marks):
    x_min = int(min(x_marks) * 0.9)
    x_max = int(max(x_marks) * 1.1)
    y_min = int(min(y_marks) * 0.9)
    y_max = int(max(y_marks) * 1.1)

    w_hand = x_max - x_min
    h_hand = y_max - y_min
    if w_hand > h_hand:
        offset = int((w_hand - h_hand) / 2)
        y_min -= offset
        y_max += offset
    else:
        offset = int((h_hand - w_hand) / 2)
        x_min -= offset
        x_max += offset

    x_min = max(x_min, 0)
    x_max = min(x_max, img.shape[1])
    y_min = max(y_min, 0)
    y_max = min(y_max, img.shape[0])

    return img[y_min:y_max, x_min:x_max]


def write_on_image(img, pred_char, prob, x_marks, y_marks):
    x_min = int(min(x_marks) * 0.9)
    x_max = int(max(x_marks) * 1.1)
    y_min = int(min(y_marks) * 0.9)
    y_max = int(max(y_marks) * 1.1)

    w_hand = x_max - x_min
    h_hand = y_max - y_min
    if w_hand > h_hand:
        offset = int((w_hand - h_hand) / 2)
        y_min -= offset
        y_max += offset
    else:
        offset = int((h_hand - w_hand) / 2)
        x_min -= offset
        x_max += offset

    x_min = max(x_min, 0)
    x_max = min(x_max, img.shape[1])
    y_min = max(y_min, 0)
    y_max = min(y_max, img.shape[0])

    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 255), 3)
    img = cv2.putText(img, pred_char, (x_min + 5, y_min + 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    img = cv2.putText(img, str((int(prob * 10000) / 100)), (x_min + 5, y_min + 80), cv2.FONT_HERSHEY_PLAIN, 2,
                      (0, 0, 255), 2)
    return img


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret is not True:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    h, w, c = img.shape
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        x_marks = np.array([lm.x * w for lm in handLms.landmark])
        y_marks = np.array([lm.y * h for lm in handLms.landmark])
        pred_char, prob = get_frame_label(x_marks, y_marks)

        if pred_char != 'nothing':
            img = write_on_image(img, pred_char, prob, x_marks, y_marks)

    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
