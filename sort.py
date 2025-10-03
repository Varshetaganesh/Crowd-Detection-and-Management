import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        # Initialize KF
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        self.kf.P[4:,4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.R[2:,2:] *= 10.0

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def update(self, bbox):
        self.time_since_update = 0
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        self.kf.predict()
        return self.kf.x

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        x = bbox[0] + w/2.0
        y = bbox[1] + h/2.0
        s = w * h
        r = w / float(h)
        return np.array([x,y,s,r]).reshape((4,1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        try:
            w = np.sqrt(max(0.0, x[2] * x[3]))
            if w < 1e-6:  # avoid division by zero
                w = 1.0
            h = max(1.0, x[2] / w)
        except Exception:
            w, h = 1.0, 1.0

        return np.array([
            float(x[0] - w / 2.0),
            float(x[1] - h / 2.0),
            float(x[0] + w / 2.0),
            float(x[1] + h / 2.0)
        ])

class Sort:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        """
        detections: [[x1,y1,x2,y2,score], ...]
        returns: [[x1,y1,x2,y2,id,score], ...]
        """
        updated_tracks = []
        to_del = []

        # Predict existing trackers
        for i, t in enumerate(self.trackers):
            pred = t.predict()
            if np.any(np.isnan(pred)):
                to_del.append(i)
        for i in reversed(to_del):
            self.trackers.pop(i)

        if len(detections) == 0:
            return []

        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections)

        # Update matched trackers
        for t_idx, d_idx in matched:
            if t_idx < len(self.trackers) and d_idx < len(detections):   # safeguard
                self.trackers[t_idx].update(detections[d_idx][:4])
                bbox = self.trackers[t_idx].get_state()
                updated_tracks.append([*bbox, self.trackers[t_idx].id, detections[d_idx][4]])

        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            trk = KalmanBoxTracker(detections[d_idx][:4])
            self.trackers.append(trk)
            bbox = trk.get_state()
            updated_tracks.append([*bbox, trk.id, detections[d_idx][4]])

        return updated_tracks

    def associate_detections_to_trackers(self, detections):
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []

        iou_matrix = np.zeros((len(detections), len(self.trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = self.iou(det, trk.get_state())

        # Sanitize matrix (no NaN/inf)
        iou_matrix = np.nan_to_num(iou_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if iou_matrix.size == 0:
            return [], list(range(len(detections))), list(range(len(self.trackers)))

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))

        unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:,0]]
        unmatched_trks = [t for t in range(len(self.trackers)) if t not in matched_indices[:,1]]

        matches = []
        for m in matched_indices:
            d_idx, t_idx = m
            if d_idx >= len(detections) or t_idx >= len(self.trackers):
                continue
            if iou_matrix[d_idx, t_idx] < self.iou_threshold:
                unmatched_dets.append(d_idx)
                unmatched_trks.append(t_idx)
            else:
                matches.append(m)

        return matches, unmatched_dets, unmatched_trks

    @staticmethod
    def iou(bb_det, bb_trk):
        xx1 = max(bb_det[0], bb_trk[0])
        yy1 = max(bb_det[1], bb_trk[1])
        xx2 = min(bb_det[2], bb_trk[2])
        yy2 = min(bb_det[3], bb_trk[3])
        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        wh = w * h
        det_area = max(1e-6, (bb_det[2]-bb_det[0])*(bb_det[3]-bb_det[1]))
        trk_area = max(1e-6, (bb_trk[2]-bb_trk[0])*(bb_trk[3]-bb_trk[1]))
        o = wh / (det_area + trk_area - wh)
        return max(0.0, min(1.0, o))
