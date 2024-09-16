import cv2
import mediapipe as mp

# Initialisation de Mediapipe pour la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Ouvre la caméra
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur: Impossible de lire le frame.")
        break

    # Convertir l'image en RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des mains
    results = hands.process(rgb_frame)

    # Dessiner les contours des mains détectées
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Entourer la main d'un cadre rouge
            x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
            y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # Affiche le frame dans une fenêtre
    cv2.imshow('Camera', frame)

    # Quitte la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libère la caméra et ferme les fenêtres
cap.release()
cv2.destroyAllWindows()
