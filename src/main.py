import platform
import sys

import cv2 as cv
from ultralytics import YOLO

cur_os = platform.system()

# 1. Carrega o modelo de IA (Baixará o arquivo yolov8n.pt automaticamente na 1ª vez)
# Deve ser o modelo que tu escolheu que n ta dando o resultado esperado
# 1. Carrega o modelo focado em Pose (Esqueleto)
model = YOLO("yolov8n-pose.pt")

indicadorSize = 10

if cur_os == "Windows":
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
elif cur_os == "Linux":
    cap = cv.VideoCapture(0)


cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("Cannot open camera")
    sys.exit()

alpha = 0.2
suave = {"esq": None, "dir": None}


def suavizar(anterior, atual):
    if anterior is None:
        return atual
    return (
        anterior[0] + alpha * (atual[0] - anterior[0]),
        anterior[1] + alpha * (atual[1] - anterior[1]),
    )


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)

    resultados = model(frame, stream=True, verbose=False)

    for r in resultados:
        # Verifica se detectou alguém e se os pontos existem
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            # Pega as coordenadas da primeira pessoa detectada
            pontos = r.keypoints.xy[0]

            # Verifica se mapeou o esqueleto até os braços (tamanho > 10)
            if len(pontos) > 10:
                # --- PULSO ESQUERDO (Índice 9) ---
                pulso_e_x = int(pontos[9][0])
                pulso_e_y = int(pontos[9][1])

                # --- PULSO DIREITO (Índice 10) ---
                pulso_d_x = int(pontos[10][0])
                pulso_d_y = int(pontos[10][1])

                # Desenha o Pulso Esquerdo em Azul se estiver visível
                if pulso_e_x != 0 and pulso_e_y != 0:
                    suave["esq"] = suavizar(suave["esq"], (pulso_e_x, pulso_e_y))
                    pulso_e_x = int(suave["esq"][0])
                    pulso_e_y = int(suave["esq"][1])
                    cv.circle(
                        frame, (pulso_e_x, pulso_e_y), indicadorSize, (255, 0, 0), -1
                    )  # Círculo Azul
                    cv.putText(
                        frame,
                        "Esq",
                        (pulso_e_x - 15, pulso_e_y - 20),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

                # Desenha o Pulso Direito em Verde se estiver visível
                if pulso_d_x != 0 and pulso_d_y != 0:
                    suave["dir"] = suavizar(suave["dir"], (pulso_d_x, pulso_d_y))
                    pulso_d_x = int(suave["dir"][0])
                    pulso_d_y = int(suave["dir"][1])
                    cv.circle(
                        frame, (pulso_d_x, pulso_d_y), indicadorSize, (0, 255, 0), -1
                    )  # Círculo Verde
                    cv.putText(
                        frame,
                        "Dir",
                        (pulso_d_x - 15, pulso_d_y - 20),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                # Exibe no terminal para debug
                # print(f"Esq(X:{pulso_e_x}, Y:{pulso_e_y}) | Dir(X:{pulso_d_x}, Y:{pulso_d_y})")

    cv.imshow("Bateria Virtual - Dois Punhos", frame)

    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
