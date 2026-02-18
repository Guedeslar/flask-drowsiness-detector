from flask import Flask,jsonify, request
from threading import Thread
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import pygame  # Para reproduzir som em segundo plano
import base64 # Для работы с Base64

app = Flask(__name__)

#Глобальные переменные
camera_active = False
status_message = "Камера неактивна."
EAR_THRESHOLD = 0.3
EAR_CONSEC_FRAMES = 48
COUNTER = 0
ALARM_ON = False
vs = None

# Inicializa o detector de rosto e o predictor de pontos faciais
print("[INFO] Carregando o detector de pontos faciais...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Obtém os índices dos pontos faciais para os olhos
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Inicializa o mixer do pygame
pygame.mixer.init()
pygame.mixer.music.load("sound_alarm/security-alarm-63578.mp3")  # Carrega o ficheiro de som

# Função para calcular o EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def monitor_driver():    # Функция для мониторинга сонливости водителя.
    global COUNTER, ALARM_ON, camera_active, vs, status_message
    while camera_active:
        ret, frame = vs.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        # Obtém os pontos faciais e converte para um array NumPy
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # Extrai as coordenadas dos olhos e calcula o EAR
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0 # Média do EAR para ambos os olhos

            # Verifica se o EAR está abaixo do limiar
            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        status_message = "Сонливость обнаружена!"
                        pygame.mixer.music.play(-1) # Reproduz o som em loop
            else:
                COUNTER = 0
                if ALARM_ON:
                    ALARM_ON = False
                    status_message = "Все в порядке."
                    pygame.mixer.music.stop() # Para o som quando os olhos abrem

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): # Sai do loop se a tecla 'q' for pressionada
            break

    vs.release()
    cv2.destroyAllWindows()

# Маршрут для запуска камеры
@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active, vs, status_message
    if not camera_active:
        vs = cv2.VideoCapture(0)
        camera_active = True
        status_message = "Камера активна."
        thread = Thread(target=monitor_driver)
        thread.start()
        return jsonify({"message": "Камера включена"}), 200
    else:
        return jsonify({"message": "Камера уже активна."}), 400

# Маршрут для остановки камеры
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active, status_message
    if camera_active:
        camera_active = False
        status_message = "Камера неактивна."
        return jsonify({"message": "Камера остановлена."}), 200
    else:
        return jsonify({"message": "Камера уже неактивна."}), 400

# Маршрут для получения статуса камеры и состояния сонливости.
@app.route('/status', methods=['GET'])
def get_status():
    global status_message
    return jsonify({"status": status_message}), 200


# маршрут для получения данных от фронтенда
@app.route('/get_data', methods=['POST'])
def get_data():
    """Получение данных от фронтенда и их обработка."""
    global EAR_THRESHOLD, EAR_CONSEC_FRAMES, detector, predictor, lStart, lEnd, rStart, rEnd

    # Получение данных из POST-запроса
    data = request.json
    if "base64_data" not in data:
        return jsonify({"error": "Отсутствует параметр 'base64_data'"}), 400

    # Раскодирование Base64 в изображение
    try:
        image_data = base64.b64decode(data["base64_data"])
        np_image = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Ошибка при декодировании изображения: {str(e)}"}), 400

    # Конвертация в черно-белое изображение
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    rects = detector(gray, 0)
    if len(rects) == 0:
        return jsonify({"message": "Лицо не обнаружено"}), 200

    # Анализируем каждое лицо
    for rect in rects:
        # Получение координат лица
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Извлечение координат глаз
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Вычисление EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Проверка на сонливость
        if ear < EAR_THRESHOLD:
            return jsonify({"message": "Сонливость обнаружена!", "ear": ear}), 200

    return jsonify({"message": "Все в порядке", "ear": ear}), 200


if __name__ == '__main__':
    app.run(debug=True)