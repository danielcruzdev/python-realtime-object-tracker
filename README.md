# 🎯 python-realtime-object-tracker

Detecção e rastreamento multi-objeto em tempo real via webcam, combinando **YOLOv8** (detecção) com o algoritmo **SORT** (rastreamento) e Filtro de Kalman.

---

## 📸 Demonstração

> Cada objeto recebe um **ID único e persistente**, bounding box colorida, label com classe, centróide e **trilha de trajetória** em tempo real.

```
[Webcam]  →  [YOLOv8: detecção]  →  [SORT + Kalman: rastreamento]  →  [Visualização OpenCV]
```

---

## 🚀 Funcionalidades

| Funcionalidade | Descrição |
|---|---|
| 🔍 Detecção YOLOv8 | Detecta 80 classes (pessoas, carros, animais, etc.) |
| 🏷️ Rastreamento SORT | IDs únicos e persistentes por objeto via Filtro de Kalman |
| 🌈 IDs coloridos | Cada track_id possui uma cor única e reprodutível |
| 〰️ Trilha de trajetória | Linha que mostra o caminho percorrido por cada objeto |
| 📊 HUD em tempo real | FPS, total de tracks ativos e contagem por classe |
| ⚙️ Ajuste de confiança | Teclas `+` / `-` alteram o threshold YOLO em tempo real |
| 👁️ Toggles de exibição | Liga/desliga centróides (`C`) e labels (`L`) |
| 📷 Screenshot | Salva frame atual em PNG com a tecla `S` |

---

## 🛠️ Pré-requisitos

### Python

Versão recomendada: **Python 3.10 ou 3.11**

> Python 3.12+ pode ter incompatibilidades com algumas dependências. Use 3.10 ou 3.11 para garantir compatibilidade.

Baixe em: https://www.python.org/downloads/

Durante a instalação no Windows, marque a opção **"Add Python to PATH"**.

Verifique a instalação:
```bash
python --version
```

---

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/python-realtime-object-tracker.git
cd python-realtime-object-tracker
```

### 2. (Recomendado) Crie um ambiente virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

> Na primeira execução, o YOLOv8 fará o download automático do modelo `yolov8n.pt` (~6 MB) caso não exista na pasta.

---

## ▶️ Executando o Projeto

```bash
# Padrão (yolov8n, câmera 0)
python main.py

# Escolher modelo
python main.py --model yolov8s.pt
python main.py --model yolov8m.pt

# Escolher câmera e modelo
python main.py --model yolov8s.pt --camera 1

# Ajustar confiança inicial
python main.py --model yolov8n.pt --conf 0.6

# Ver todas as opções
python main.py --help
```

Uma janela será aberta com o feed da webcam em tempo real com os objetos detectados e rastreados.

---

## ⌨️ Controles

| Tecla | Ação |
|---|---|
| `Q` ou `ESC` | Encerrar o programa |
| `C` | Mostrar / esconder centróides |
| `L` | Mostrar / esconder labels (classe + ID) |
| `+` ou `=` | Aumentar confiança mínima (+5%) |
| `-` | Diminuir confiança mínima (-5%) |
| `S` | Salvar screenshot da janela atual |

---

## ⚙️ Configurações

Edite as constantes no início do `main.py` para personalizar o comportamento:

```python
CAMERA_INDEX   = 0      # Índice da webcam (0 = padrão do sistema)
YOLO_MODEL     = "yolov8n.pt"  # Modelo: n (rápido) → s → m → l → x (preciso)
CONF_THRESHOLD = 0.4    # Confiança mínima inicial (0.0 a 1.0)
IOU_THRESHOLD  = 0.45   # IoU para NMS interno do YOLO
MAX_AGE        = 5      # Frames máximos sem atualização antes de remover um track
MIN_HITS       = 3      # Detecções mínimas para confirmar um novo track
SORT_IOU       = 0.3    # IoU mínimo para associar detecção a um track existente
TRAIL_LENGTH   = 40     # Número de posições armazenadas na trilha de trajetória
```

### Escolha do modelo YOLO

| Modelo | Velocidade | Precisão | Uso recomendado |
|---|---|---|---|
| `yolov8n.pt` | ⚡⚡⚡ | ★★☆ | Webcam / CPU |
| `yolov8s.pt` | ⚡⚡ | ★★★ | CPU moderno / GPU |
| `yolov8m.pt` | ⚡ | ★★★★ | GPU |
| `yolov8l.pt` | 🐢 | ★★★★★ | GPU dedicada |

---

## 📂 Estrutura do Projeto

```
yolo-sort-webcam/
├── tracker/
│   ├── __init__.py   # Exporta a classe Sort
│   └── sort.py       # Algoritmo SORT com Filtro de Kalman
├── models/           # Pesos .pt (ignorados pelo git, baixados automaticamente)
│   └── yolov8n.pt
├── screenshots/      # Screenshots salvos com a tecla S (ignorados pelo git)
├── main.py           # Loop principal: captura, detecção, rastreamento e HUD
├── requirements.txt  # Dependências do projeto
├── .gitignore
└── README.md
```

---

## 🧠 Como Funciona

### 1. Detecção — YOLOv8
O frame da webcam é processado pelo **YOLOv8** que retorna bounding boxes `[x1, y1, x2, y2]` com a classe e a confiança de cada objeto detectado.

### 2. Rastreamento — SORT + Filtro de Kalman
O **SORT** (*Simple Online and Realtime Tracking*) recebe as detecções do YOLO e:
- Usa o **Algoritmo Húngaro** para associar detecções anteriores com as novas (baseado em IoU)
- Mantém um **Filtro de Kalman** por objeto para prever sua posição mesmo quando não detectado
- Atribui **IDs únicos e estáveis** por objeto entre os frames

### 3. Visualização — OpenCV
Cada track renderizado inclui:
- Bounding box colorida por ID
- Label `ID X | classe`
- Centróide (ponto central)
- Trilha de trajetória (últimas N posições)
- HUD com FPS, total de tracks ativos e contagem por classe

---

## 📚 Referências

- [YOLOv8 — Ultralytics](https://github.com/ultralytics/ultralytics)
- [SORT — Bewley et al., 2016](https://arxiv.org/abs/1602.00763)
- [FilterPy — Filtro de Kalman em Python](https://github.com/rlabbe/filterpy)
- [OpenCV](https://opencv.org/)

---

## 📋 Requisitos de Sistema

- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.10 ou 3.11
- **RAM**: mínimo 4 GB (8 GB recomendado)
- **Webcam**: qualquer câmera compatível com OpenCV
- **GPU** *(opcional)*: CUDA compatível para aceleração do YOLO
