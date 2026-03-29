"""
YOLO + SORT Webcam Tracker
--------------------------
Detecta e rastreia objetos em tempo real usando:
  - YOLO26 (Ultralytics) para detecção de objetos
  - SORT para rastreamento multi-objeto

Controles:
  Q / ESC  → Encerrar
  C        → Alternar entre mostrar/esconder centróides
  L        → Alternar labels (classe + ID)
  +/-      → Aumentar/diminuir confiança mínima do YOLO

Uso:
  python main.py                        # padrão: yolo26n, câmera 0
  python main.py --model yolo26s.pt
  python main.py --model yolo26m.pt --camera 1
"""

import argparse
import cv2
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
from ultralytics import YOLO
from tracker.sort import Sort

MODELS_DIR  = Path("models")
SCREEN_DIR  = Path("screenshots")
MODEL_NAMES = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]


# ──────────────────────────────────────────────
# Configurações
# ──────────────────────────────────────────────
CAMERA_INDEX   = 0              # Índice da webcam (0 = padrão)
YOLO_MODEL     = "yolo26n.pt"   # Modelo YOLO: yolo26n/s/m/l/x
CONF_THRESHOLD = 0.4      # Confiança mínima inicial para detecção
IOU_THRESHOLD  = 0.45     # IoU para NMS do YOLO
MAX_AGE        = 5        # SORT: frames máximos sem atualização
MIN_HITS       = 3        # SORT: detecções mínimas para confirmar track
SORT_IOU       = 0.3      # SORT: IoU mínimo para associação
TRAIL_LENGTH   = 40       # Número de posições armazenadas na trilha de trajetória

# ──────────────────────────────────────────────
# Paleta de cores para IDs de rastreamento
# ──────────────────────────────────────────────
def get_color(track_id: int) -> tuple[int, int, int]:
    """Retorna uma cor BGR reproducível para um dado track_id."""
    np.random.seed(int(track_id) * 37 % 256)
    return tuple(int(x) for x in np.random.randint(80, 255, 3))


def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_length=10):
    """Desenha um retângulo com bordas tracejadas."""
    x1, y1 = pt1
    x2, y2 = pt2
    pts = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in pts:
        dist = int(np.hypot(ex - sx, ey - sy))
        for i in range(0, dist, dash_length * 2):
            t0 = i / dist
            t1 = min((i + dash_length) / dist, 1.0)
            p0 = (int(sx + (ex - sx) * t0), int(sy + (ey - sy) * t0))
            p1 = (int(sx + (ex - sx) * t1), int(sy + (ey - sy) * t1))
            cv2.line(img, p0, p1, color, thickness)


def draw_label(img, text: str, pos: tuple[int, int], color: tuple[int, int, int]):
    """Desenha uma label com fundo colorido."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Fundo
    cv2.rectangle(img, (x, y - th - bl - 4), (x + tw + 4, y + 2), color, -1)
    # Texto branco
    cv2.putText(img, text, (x + 2, y - bl - 2), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def overlay_info(frame, conf_threshold: float, n_tracks: int, fps: float,
                 class_counts: dict, model_name: str = ""):
    """HUD com informações gerais no canto superior esquerdo."""
    counts_str = "  ".join(f"{cls}:{cnt}" for cls, cnt in sorted(class_counts.items()))
    lines = [
        f"Modelo:    {model_name}",
        f"FPS:       {fps:.1f}",
        f"Tracks:    {n_tracks}",
        f"Confianca: {conf_threshold:.2f}  (+/- para ajustar)",
        f"Classes:   {counts_str}" if counts_str else "Classes:   —",
        "C: centroide | L: labels | S: screenshot | Q/ESC: sair",
    ]
    x, y0 = 10, 25
    font  = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(frame, line, (x, y), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), font, 0.5, (200, 255, 200), 1, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + SORT Webcam Tracker")
    parser.add_argument(
        "--model", "-m",
        default=YOLO_MODEL,
        choices=MODEL_NAMES,
        help=f"Modelo YOLO26 a usar. Opções: {', '.join(MODEL_NAMES)} (padrão: {YOLO_MODEL})",
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=CAMERA_INDEX,
        help=f"Índice da câmera (padrão: {CAMERA_INDEX})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help=f"Confiança mínima inicial (padrão: {CONF_THRESHOLD})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Inicializa YOLO ──────────────────────────────────────────────────────
    model_path = MODELS_DIR / args.model
    print(f"[INFO] Carregando modelo {model_path}...")
    idx = MODEL_NAMES.index(args.model)
    print(f"[INFO] Velocidade/precisão: {' < '.join(MODEL_NAMES)} (você escolheu: #{idx + 1})")
    model = YOLO(str(model_path))
    class_names = model.names  # dict {id: nome}
    print(f"[INFO] Modelo carregado. Classes: {len(class_names)}")

    # ── Inicializa SORT ──────────────────────────────────────────────────────
    tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=SORT_IOU)

    # ── Abre Webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera (índice {args.camera}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Estado da UI ─────────────────────────────────────────────────────────
    conf_threshold  = CONF_THRESHOLD
    show_centroids  = True
    show_labels     = True

    # Mapeamento track_id → classe detectada (mantém classe estável)
    id_class_map: dict[int, str] = {}

    # Trilha de trajetória: track_id → deque de centroides (x, y)
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

    # Contador de screenshots
    SCREEN_DIR.mkdir(exist_ok=True)
    screenshot_count = 0

    # Cálculo de FPS
    fps_counter = 0
    fps_display = 0.0
    tick_freq   = cv2.getTickFrequency()
    t_start     = cv2.getTickCount()

    print("[INFO] Pressione Q/ESC para sair, C/L para alternar exibição, +/- para confiança.")
    print("[INFO] Iniciando captura...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Falha ao capturar frame.")
            break

        frame = cv2.flip(frame, 1)  # espelho para webcam frontal

        # ── Detecção YOLO ────────────────────────────────────────────────────
        results = model(
            frame,
            conf=conf_threshold,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        # Converte detecções para array [x1, y1, x2, y2, conf, class_id]
        detections_full = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                detections_full.append([x1, y1, x2, y2, conf, cls])

        detections_full = np.array(detections_full) if detections_full else np.empty((0, 6))

        # Prepara array para SORT: [x1, y1, x2, y2, score]
        dets_for_sort = (
            detections_full[:, :5]
            if len(detections_full) > 0
            else np.empty((0, 5))
        )

        # ── Atualização SORT ─────────────────────────────────────────────────
        tracks = tracker.update(dets_for_sort)
        # tracks: [[x1, y1, x2, y2, track_id], ...]

        # Associa track_id → classe (baseado no IoU com detecções atuais)
        for track in tracks:
            tx1, ty1, tx2, ty2, tid = track
            tid = int(tid)
            if len(detections_full) > 0:
                # Encontra a detecção com maior IoU para este track
                best_iou = 0.0
                best_cls = None
                for det in detections_full:
                    dx1, dy1, dx2, dy2, _, cls_id = det
                    inter_x1 = max(tx1, dx1)
                    inter_y1 = max(ty1, dy1)
                    inter_x2 = min(tx2, dx2)
                    inter_y2 = min(ty2, dy2)
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    inter_area = inter_w * inter_h
                    area_t = (tx2 - tx1) * (ty2 - ty1)
                    area_d = (dx2 - dx1) * (dy2 - dy1)
                    union  = area_t + area_d - inter_area
                    iou_val = inter_area / union if union > 0 else 0.0
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_cls = int(cls_id)
                if best_cls is not None and best_iou > 0.2:
                    id_class_map[tid] = class_names.get(best_cls, f"cls{best_cls}")

        # ── Renderização ─────────────────────────────────────────────────────
        active_ids = {int(t[4]) for t in tracks}
        # Remove trilhas de IDs expirados
        for old_id in list(trails.keys()):
            if old_id not in active_ids:
                del trails[old_id]

        # Contagem por classe (tracks ativos)
        class_counts: dict[str, int] = defaultdict(int)

        for track in tracks:
            x1, y1, x2, y2, tid = [int(v) for v in track]
            color      = get_color(tid)
            class_name = id_class_map.get(tid, "?")
            class_counts[class_name] += 1

            # Atualiza trilha com centróide atual
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            trails[tid].append((cx, cy))

            # Desenha trilha de trajetória
            pts = list(trails[tid])
            for i in range(1, len(pts)):
                alpha = i / len(pts)         # fade: mais fraco no início
                thickness = max(1, int(alpha * 3))
                c = tuple(int(v * alpha) for v in color)
                cv2.line(frame, pts[i - 1], pts[i], c, thickness)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            if show_labels:
                label = f"ID {tid} | {class_name}"
                draw_label(frame, label, (x1, y1), color)

            # Centróide
            if show_centroids:
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.circle(frame, (cx, cy), 8, color, 2)

        # ── FPS ──────────────────────────────────────────────────────────────
        fps_counter += 1
        elapsed = (cv2.getTickCount() - t_start) / tick_freq
        if elapsed >= 0.5:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            t_start     = cv2.getTickCount()

        # ── HUD ──────────────────────────────────────────────────────────────
        overlay_info(frame, conf_threshold, len(tracks), fps_display, class_counts, args.model)

        cv2.imshow("YOLO + SORT Tracker  |  Q/ESC para sair", frame)

        # ── Teclado ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):       # Q ou ESC
            break
        elif key == ord("c"):
            show_centroids = not show_centroids
        elif key == ord("l"):
            show_labels = not show_labels
        elif key in (ord("+"), ord("=")):
            conf_threshold = min(0.95, conf_threshold + 0.05)
            print(f"[INFO] Confiança ajustada para {conf_threshold:.2f}")
        elif key == ord("-"):
            conf_threshold = max(0.05, conf_threshold - 0.05)
            print(f"[INFO] Confiança ajustada para {conf_threshold:.2f}")
        elif key == ord("s"):
            screenshot_count += 1
            filepath = SCREEN_DIR / f"screenshot_{screenshot_count:03d}.png"
            cv2.imwrite(str(filepath), frame)
            print(f"[INFO] Screenshot salvo: {filepath}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Encerrando.")


if __name__ == "__main__":
    main()
