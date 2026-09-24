import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Caminho do modelo
MODEL_PATH = "hand_landmarker.task"


# Configuração do modelo
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = vision.HandLandmarker.create_from_options(options)


# Webcam
cap = cv2.VideoCapture(0)

timestamp = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("Erro ao acessar a webcam.")
        break

    # Espelhar webcam
    frame = cv2.flip(frame, 1)

    # Converter BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Criar imagem para o MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 1

    # Detectar mão
    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    h, w, _ = frame.shape


    # Se encontrou uma mão
    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Ponta do indicador = landmark 8
        dedo = hand[8]

        x = int(dedo.x * w)
        y = int(dedo.y * h)


        # Quadrado
        tamanho = 30

        cv2.rectangle(
            frame,
            (x - tamanho, y - tamanho),
            (x + tamanho, y + tamanho),
            (0, 255, 0),
            3
        )


        # Ponto central
        cv2.circle(
            frame,
            (x, y),
            6,
            (0, 255, 0),
            -1
        )


        # Texto
        cv2.putText(
            frame,
            "DEDO DETECTADO",
            (x + 40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Desenhar todos os pontos da mão
        for ponto in hand:

            px = int(ponto.x * w)
            py = int(ponto.y * h)

            cv2.circle(
                frame,
                (px, py),
                4,
                (255, 0, 0),
                -1
            )


    else:

        cv2.putText(
            frame,
            "MAO NAO DETECTADA",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    cv2.imshow(
        "HzSee - Detector de Dedo",
        frame
    )


    # Q para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
detector.close()
cv2.destroyAllWindows()