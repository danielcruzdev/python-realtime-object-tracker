"""
SORT: Simple Online and Realtime Tracking
Implementação do algoritmo SORT para rastreamento de objetos.

Referência: Bewley et al., 2016 - "Simple Online and Realtime Tracking"
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


def iou(bb_test, bb_gt):
    """
    Calcula o Intersection over Union (IoU) entre duas bounding boxes.
    
    Args:
        bb_test: [x1, y1, x2, y2]
        bb_gt:   [x1, y1, x2, y2]
    
    Returns:
        float: valor IoU entre 0 e 1
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2]   - bb_gt[0])   * (bb_gt[3]   - bb_gt[1])
    union = area_test + area_gt - intersection

    return intersection / union if union > 0 else 0.0


def convert_bbox_to_z(bbox):
    """
    Converte bounding box [x1,y1,x2,y2] para formato de estado Kalman
    [cx, cy, area, aspect_ratio].
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    area = w * h
    ratio = w / float(h) if h != 0 else 1.0
    return np.array([[cx], [cy], [area], [ratio]], dtype=np.float32)


def convert_x_to_bbox(x, score=None):
    """
    Converte estado Kalman [cx, cy, area, ratio, ...] de volta para [x1,y1,x2,y2].
    """
    w = np.sqrt(abs(x[2] * x[3]))
    h = x[2] / w if w != 0 else 0
    box = [
        x[0] - w / 2.0,
        x[1] - h / 2.0,
        x[0] + w / 2.0,
        x[1] + h / 2.0,
    ]
    if score is None:
        return np.array(box).reshape(1, 4)
    return np.array(box + [score]).reshape(1, 5)


class KalmanBoxTracker:
    """
    Rastreador individual baseado em Filtro de Kalman para uma única bounding box.
    Estado: [cx, cy, area, ratio, vx, vy, v_area]
    """

    count = 0

    def __init__(self, bbox):
        """
        Inicializa o rastreador com uma detecção inicial.
        
        Args:
            bbox: [x1, y1, x2, y2] ou [x1, y1, x2, y2, score]
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # Matriz de transição de estado (modelo de velocidade constante)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # Matriz de observação (medimos apenas posição/tamanho, não velocidade)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        # Ruído de medição
        self.kf.R[2:, 2:] *= 10.0

        # Covariância inicial (alta incerteza nas velocidades)
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Ruído do processo
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """Atualiza o estado usando uma nova detecção."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """Avança o estado e retorna a bounding box prevista."""
        # Evitar área negativa
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """Retorna a bounding box atual estimada."""
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Associa detecções a rastreadores usando o algoritmo Húngaro (IoU como métrica).

    Args:
        detections: array Nx4 com [x1, y1, x2, y2]
        trackers:   array Mx4 com [x1, y1, x2, y2]
        iou_threshold: IoU mínimo para considerar um match válido

    Returns:
        matches, unmatched_detections, unmatched_trackers
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int),
        )

    # Monta matriz de IoU (negativa para minimização)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    # Algoritmo Húngaro (minimização → maximizamos IoU com negativo)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.stack((row_ind, col_ind), axis=1)

    unmatched_detections = [
        d for d in range(len(detections))
        if d not in matched_indices[:, 0]
    ]
    unmatched_trackers = [
        t for t in range(len(trackers))
        if t not in matched_indices[:, 1]
    ]

    # Filtra matches com IoU abaixo do limiar
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """
    SORT: Simple Online and Realtime Tracker.

    Usage:
        tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
        # Para cada frame:
        tracks = tracker.update(detections)   # detections: Nx4 ou Nx5 array
        # tracks: Mx5 array [x1, y1, x2, y2, track_id]
    """

    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        """
        Args:
            max_age:       Número máximo de frames que um rastreador pode existir
                           sem ser atualizado antes de ser removido.
            min_hits:      Número mínimo de detecções consecutivas antes de um
                           rastreador ser confirmado e exibido.
            iou_threshold: IoU mínimo para associar detecção a rastreador.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Atualiza o rastreador com novas detecções.

        Args:
            dets: np.array de shape (N, 4) ou (N, 5) — [x1, y1, x2, y2] ou
                  [x1, y1, x2, y2, score]. Passar array vazio quando não há
                  detecções no frame.

        Returns:
            np.array de shape (M, 5): [x1, y1, x2, y2, track_id] para cada
            objeto rastreado e confirmado no frame atual.
        """
        self.frame_count += 1

        # Predição para todos os rastreadores ativos
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = pos[:4]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Remove rastreadores com NaN
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Associação detecções ↔ rastreadores
        dets_bbox = dets[:, :4] if len(dets) > 0 else np.empty((0, 4))
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets_bbox, trks, self.iou_threshold
        )

        # Atualiza rastreadores com detecções correspondentes
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # Cria novos rastreadores para detecções sem correspondência
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        # Coleta resultados e remove rastreadores mortos
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            d = trk.get_state()[0]
            # Exibe rastreador apenas se foi confirmado (min_hits) ou ainda jovem
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            # Remove rastreadores antigos sem atualização
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.concatenate(ret) if ret else np.empty((0, 5))
