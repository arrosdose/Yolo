"""
Lógica do alarme por câmera para o estoque.

Detecta pessoas com YOLOv4-tiny (lido pelo módulo cv2.dnn do OpenCV) e verifica
se há azul (cor do uniforme) no tronco de cada pessoa. Se alguém ficar
FRAMES_PARA_ALERTA quadros seguidos sem uniforme, o alarme é disparado.

É o mesmo código da Seção 2 do notebook notebooks/Alarme_estoque_YOLO.ipynb,
separado em módulo para ser usado também pelo alarme_local.py.
"""
import cv2
import numpy as np

# ---------------- Configurações ----------------
CONF_MIN = 0.5           # confiança mínima para aceitar uma "pessoa"
NMS_MIN = 0.4            # supressão de caixas sobrepostas
AZUL_MIN_PCT = 0.12      # % mínima de pixels azuis no tronco para considerar uniforme
FRAMES_PARA_ALERTA = 25  # frames seguidos sem uniforme antes de disparar (evita falso alarme)

# Faixas de azul em HSV (OpenCV: H 0-179). Cobrem azul escuro, médio e claro.
FAIXAS_AZUL = [
    (np.array([100, 120, 40]), np.array([130, 255, 255])),  # azul forte / escuro
    (np.array([95, 60, 60]), np.array([135, 255, 255])),    # azul médio
    (np.array([90, 35, 120]), np.array([135, 140, 255])),   # azul claro
]


def carrega_rede(cfg, pesos):
    """Carrega o YOLOv4-tiny (arquivos Darknet .cfg e .weights) pelo OpenCV."""
    net = cv2.dnn.readNetFromDarknet(cfg, pesos)
    modelo = cv2.dnn_DetectionModel(net)
    modelo.setInputParams(size=(416, 416), scale=1 / 255.0, swapRB=True)
    return modelo


def detecta_pessoas(modelo, frame):
    """Retorna lista de caixas (x, y, w, h) de pessoas (classe 0 do COCO)."""
    classes, confs, caixas = modelo.detect(frame, CONF_MIN, NMS_MIN)
    pessoas = []
    for c, caixa in zip(np.array(classes).flatten(), caixas):
        if int(c) == 0:
            pessoas.append(tuple(int(v) for v in caixa))
    return pessoas


def mascara_azul(hsv):
    """Máscara binária: branco onde o pixel está em alguma faixa de azul."""
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in FAIXAS_AZUL:
        m |= cv2.inRange(hsv, lo, hi)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def tem_uniforme(frame, caixa):
    """Analisa só o tronco (miolo da parte superior da caixa) e mede o % de azul."""
    x, y, w, h = caixa
    x0, x1 = x + int(0.2 * w), x + int(0.8 * w)
    y0, y1 = y + int(0.15 * h), y + int(0.55 * h)
    H, W = frame.shape[:2]
    x0, x1, y0, y1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return False, 0.0
    pct = float(np.count_nonzero(mascara_azul(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)))) / (roi.shape[0] * roi.shape[1])
    return pct >= AZUL_MIN_PCT, pct


class Monitor:
    """Processa frame a frame, mantém o contador de frames 'suspeitos' e decide o alerta."""

    def __init__(self, modelo):
        self.modelo = modelo
        self.seq_suspeito = 0
        self.alarme = False

    def processa(self, frame):
        """Desenha o resultado no frame. Retorna (frame, número de pessoas, alarme ligado?)."""
        pessoas = detecta_pessoas(self.modelo, frame)
        algum_suspeito = False
        for caixa in pessoas:
            ok, pct = tem_uniforme(frame, caixa)
            x, y, w, h = caixa
            cor = (0, 200, 0) if ok else (0, 0, 255)
            rotulo = f"Funcionario {pct:.0%}" if ok else f"SEM UNIFORME {pct:.0%}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 3)
            cv2.putText(frame, rotulo, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
            algum_suspeito |= not ok
        self.seq_suspeito = self.seq_suspeito + 1 if algum_suspeito else 0
        self.alarme = self.seq_suspeito >= FRAMES_PARA_ALERTA
        if self.alarme:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 12)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(frame, "ALERTA: POSSIVEL ENTRADA NAO AUTORIZADA", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return frame, len(pessoas), self.alarme
