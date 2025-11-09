from typing import List, Optional, Tuple
import numpy as np
import cv2

from .detectors.roboflow_detector import LocalYOLODetector
from .core.pose import PoseEstimator
from .core.tracker import SimpleTracker


class CubeVision:
    """Vision pipeline: YOLO-seg facelets → cluster → 3x3 quads → PnP pose."""

    COLOR_TO_LETTER = {
        'White': 'U',   'Yellow': 'D',  'Red': 'R',
        'Orange': 'L',  'Green': 'F',   'Blue': 'B'
    }

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 model_path: str = "models/rubik_cube_yolo11_seg.pt",
                 device: str = 'cpu', half: bool = False,
                 allowed_names: Optional[List[str]] = None,
                 default_conf: float = 0.5,
                 ema_alpha: float = 0.25,
                 refine_lm: bool = False,
                 imgsz: int = 640,
                 iou: float = 0.5,
                 max_det: int = 100):
        self.detector = LocalYOLODetector(model_path=model_path, device=device, half=half,
                                          allowed_names=set(allowed_names) if allowed_names else None,
                                          default_conf=default_conf,
                                          imgsz=imgsz, iou=iou, max_det=max_det)
        self.pose = PoseEstimator(camera_matrix, dist_coeffs, refine_lm=refine_lm)
        self.tracker = SimpleTracker()

        self.current_facelets: List[dict] = []
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_rvec = None
        self.last_tvec = None
        self.rvec_smooth = None
        self.rvec_smooth_alpha = float(ema_alpha)

    def detect_facelets(self, frame) -> List[dict]:
        detections = self.detector.detect(frame, conf_threshold=self.default_conf)
        self.current_facelets = detections
        return detections

    def select_face_cluster(self, detections: List[dict]) -> Optional[List[dict]]:
        if not detections:
            return None
        items = []
        for det in detections:
            x1,y1,x2,y2 = det['x1'],det['y1'],det['x2'],det['y2']
            cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
            size = 0.5*((x2-x1)+(y2-y1))
            items.append((cx,cy,size))
        if not items:
            return None
        n = len(items)
        visited = [False]*n
        components = []
        for i in range(n):
            if visited[i]:
                continue
            q=[i]; visited[i]=True; comp=[i]
            while q:
                u=q.pop(0); ux,uy,us=items[u]
                for v in range(n):
                    if visited[v]:
                        continue
                    vx,vy,vs=items[v]
                    th=1.8*max(us,vs)
                    if (ux-vx)**2+(uy-vy)**2<=th**2:
                        visited[v]=True; q.append(v); comp.append(v)
            components.append(comp)
        best_comp=None; best_score=-1e9
        for comp in components:
            if len(comp)<5: continue
            xs=[detections[i]['x1'] for i in comp]+[detections[i]['x2'] for i in comp]
            ys=[detections[i]['y1'] for i in comp]+[detections[i]['y2'] for i in comp]
            w=max(xs)-min(xs); h=max(ys)-min(ys)
            aspect=(w+1e-6)/(h+1e-6)
            size_std=float(np.std([items[i][2] for i in comp]))
            score=len(comp)*2.0-0.6*abs(aspect-1.0)-0.02*size_std
            if score>best_score:
                best_score=score; best_comp=comp
        if not best_comp:
            return None
        cx_all=float(np.mean([items[i][0] for i in best_comp]))
        cy_all=float(np.mean([items[i][1] for i in best_comp]))
        best_sorted=sorted(best_comp, key=lambda i:(items[i][0]-cx_all)**2+(items[i][1]-cy_all)**2)
        keep_idx=best_sorted[:min(9,len(best_sorted))]
        return [detections[i] for i in keep_idx]

    def build_grid_from_segmentation(self, detections: Optional[List[dict]] = None) -> Optional[list]:
        dets = detections if detections is not None else self.current_facelets
        if not dets:
            return None
        cluster = self.select_face_cluster(dets)
        if not cluster or len(cluster) < 5:
            return None
        centers=[]
        for d in cluster:
            pts=d.get('points')
            if pts and len(pts)>=3:
                arr=np.asarray(pts,dtype=np.float32)
                M=cv2.moments(arr)
                if abs(M.get('m00',0))>1e-6:
                    cx=float(M['m10']/M['m00']); cy=float(M['m01']/M['m00'])
                else:
                    cx=0.5*(d['x1']+d['x2']); cy=0.5*(d['y1']+d['y2'])
            else:
                cx=0.5*(d['x1']+d['x2']); cy=0.5*(d['y1']+d['y2'])
            centers.append([cx,cy])
        centers=np.asarray(centers,dtype=np.float32)
        if centers.shape[0]<5:
            return None
        mu=centers.mean(0)
        _,_,Vt=np.linalg.svd(centers-mu,full_matrices=False)
        u_axis=Vt[0]; v_axis=Vt[1]
        if np.cross(np.append(u_axis,0),np.append(v_axis,0))[2]<0:
            v_axis=-v_axis
        u=(centers-mu)@u_axis; v=(centers-mu)@v_axis
        u_min,u_max=float(u.min()),float(u.max())
        v_min,v_max=float(v.min()),float(v.max())
        if u_max-u_min<1e-3 or v_max-v_min<1e-3:
            return None
        shrink=0.98
        u_span=0.5*(u_max-u_min)*shrink; v_span=0.5*(v_max-v_min)*shrink
        u_mid=0.5*(u_max+u_min); v_mid=0.5*(v_max+v_min)
        u0,u1=u_mid-u_span,u_mid+u_span
        v0,v1=v_mid-v_span,v_mid+v_span
        u_lines=np.linspace(u0,u1,4); v_lines=np.linspace(v0,v1,4)
        def uv2xy(uu,vv):
            return (mu + uu*u_axis + vv*v_axis).astype(np.float32)
        quads=[]
        for r in range(3):
            for c in range(3):
                p00=uv2xy(u_lines[c],  v_lines[r])
                p10=uv2xy(u_lines[c+1],v_lines[r])
                p11=uv2xy(u_lines[c+1],v_lines[r+1])
                p01=uv2xy(u_lines[c],  v_lines[r+1])
                quads.append(np.stack([p00,p10,p11,p01],axis=0))
        return quads

    # ---- Fallbacks: outer quad + warp-to-square + uniform 3x3 ----
    @staticmethod
    def _order_tl_tr_br_bl(pts: np.ndarray) -> Optional[np.ndarray]:
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 2:
            return None
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.float32)
        s = box.sum(1)
        d = np.diff(box, axis=1).ravel()
        tl = box[np.argmin(s)]; br = box[np.argmax(s)]
        tr = box[np.argmin(d)]; bl = box[np.argmax(d)]
        return np.stack([tl, tr, br, bl], axis=0)

    def _outer_quad_from_dets(self, dets: List[dict]) -> Optional[np.ndarray]:
        if not dets:
            return None
        corners = []
        for d in dets:
            x1,y1,x2,y2 = d['x1'], d['y1'], d['x2'], d['y2']
            corners.extend([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
            pts = d.get('points')
            if pts and len(pts) >= 3:
                corners.extend(pts)
        return self._order_tl_tr_br_bl(np.array(corners, dtype=np.float32))

    @staticmethod
    def _warp_to_square(img, quad_xy, out_size=300):
        quad = np.array(quad_xy, dtype=np.float32)
        dst = np.float32([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]])
        H = cv2.getPerspectiveTransform(quad, dst)
        face = cv2.warpPerspective(img, H, (out_size, out_size), flags=cv2.INTER_LINEAR)
        return face, H

    @staticmethod
    def _uniform_quads(out_size=300, margin=20):
        cs = out_size // 3
        quads = []
        for i in range(3):
            for j in range(3):
                x1 = j * cs + margin
                y1 = i * cs + margin
                x2 = (j + 1) * cs - margin
                y2 = (i + 1) * cs - margin
                quad = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float32)
                quads.append(quad)
        return quads

    def fallback_quads_via_contour(self, frame, detections: Optional[List[dict]] = None) -> Optional[list]:
        dets = detections if detections is not None else self.current_facelets
        if not dets:
            return None
        cluster = self.select_face_cluster(dets)
        src = cluster if cluster else dets
        quad = self._outer_quad_from_dets(src)
        if quad is None:
            return None
        face, H = self._warp_to_square(frame, quad, out_size=300)
        # build uniform grid in square coords
        quads_sq = self._uniform_quads(out_size=300, margin=20)
        # map back to original image via H^-1
        Hinv = np.linalg.inv(H)
        quads_img = []
        for q in quads_sq:
            pts = q.reshape(-1,1,2).astype(np.float32)
            mapped = cv2.perspectiveTransform(pts, Hinv).reshape(-1,2)
            quads_img.append(mapped)
        return quads_img

    # ---- Rotation from rvec to k×90° and apply to labels ----
    @staticmethod
    def rotation_k_from_rvec(rvec: np.ndarray) -> int:
        if rvec is None:
            return 0
        R, _ = cv2.Rodrigues(rvec)
        # Project cube local x-axis to image plane (approx using camera X-Y)
        face_x = R @ np.array([1.0, 0.0, 0.0])
        angle = np.arctan2(face_x[1], face_x[0])  # against image x-axis
        k = int(np.round(angle / (np.pi/2))) % 4
        return k

    @staticmethod
    def rotate_grid_labels(labels9: List[str], k: int) -> List[str]:
        g = np.array(labels9).reshape(3,3)
        return np.rot90(g, -k).reshape(-1).tolist()

    def estimate_pose_from_quads(self, frame, quads) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if quads is None or len(quads)!=9:
            return None
        pose = self.pose.estimate_pose_from_grid(frame, quads, roi_offset=(0,0))
        if pose is None:
            return None
        rvec_raw, tvec_raw = pose
        if self.last_rvec is not None and self.last_tvec is not None:
            alpha=0.3
            rvec_raw = (1-alpha)*self.last_rvec + alpha*rvec_raw
            tvec_raw = (1-alpha)*self.last_tvec + alpha*tvec_raw
        self.last_rvec = rvec_raw
        self.last_tvec = tvec_raw
        if self.rvec_smooth is None:
            self.rvec_smooth = rvec_raw.copy()
        else:
            self.rvec_smooth = (1.0-self.rvec_smooth_alpha)*self.rvec_smooth + self.rvec_smooth_alpha*rvec_raw
        return rvec_raw, tvec_raw

    @staticmethod
    def visible_face_from_rvec(rvec) -> Optional[str]:
        if rvec is None:
            return None
        R,_=cv2.Rodrigues(rvec)
        normals={
            'F': np.array([0,0, 1.0]),
            'B': np.array([0,0,-1.0]),
            'U': np.array([0,1.0,0]),
            'D': np.array([0,-1.0,0]),
            'R': np.array([1.0,0,0]),
            'L': np.array([-1.0,0,0]),
        }
        best=None; bestz=-1e9
        for k,n in normals.items():
            z=(R@n)[2]
            if z>bestz:
                best=k; bestz=z
        return best

    def order_facelets_3x3(self, dets: List[dict]) -> Optional[List[dict]]:
        if len(dets)<9:
            return None
        centers=np.array([[(d['x1']+d['x2'])/2., (d['y1']+d['y2'])/2.] for d in dets[:9]], np.float32)
        mu=centers.mean(0)
        _,_,Vt=np.linalg.svd(centers-mu)
        ax=Vt[0]; ay=Vt[1]
        if np.cross(np.append(ax,0),np.append(ay,0))[2]<0:
            ay=-ay
        u=(centers-mu)@ax; v=(centers-mu)@ay
        row_bounds=np.percentile(v,[33.33,66.67])
        col_bounds=np.percentile(u,[33.33,66.67])
        rows=np.digitize(v,row_bounds); cols=np.digitize(u,col_bounds)
        grid=[[None]*3 for _ in range(3)]
        for i,(r,c) in enumerate(zip(rows,cols)):
            r=int(r); c=int(c)
            if 0<=r<3 and 0<=c<3 and grid[r][c] is None:
                grid[r][c]=dets[i]
        if any(grid[r][c] is None for r in range(3) for c in range(3)):
            return None
        return [grid[r][c] for r in range(3) for c in range(3)]
