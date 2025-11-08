"""
3D 迷你魔方 HUD 渲染器
在右上角显示跟随视角的3D魔方状态
"""
import cv2
import numpy as np
from typing import Optional

class MiniCubeHUD:
    def __init__(self, size=220):
        self.size = size
        f = size * 1.1
        self.Kp = np.array([[f,0,size/2],[0,f,size/2],[0,0,1]], np.float32)
        self.dist = np.zeros(5, np.float32)
        
        # 立方体(边长=1)的8顶点（物体坐标，需与FACE_NORMALS_OBJ一致）
        a = 0.5
        self.verts = np.float32([
            [-a,-a,-a],[ a,-a,-a],[ a, a,-a],[-a, a,-a],  # B面四角 z=-a
            [-a,-a, a],[ a,-a, a],[ a, a, a],[-a, a, a],  # F面四角 z=+a
        ])
        
        # 每个面的四角（按TL,TR,BR,BL）
        self.face_quads = {
            'F': [4,5,6,7], 'B': [1,0,3,2],
            'U': [3,2,6,7], 'D': [0,1,5,4],
            'R': [2,1,5,6], 'L': [0,3,7,4],
        }
        
        # BGR颜色（未知=灰）
        self.colormap = {
            'U':(255,255,255),'D':(0,255,255),
            'R':(0,0,255),'L':(0,165,255),
            'F':(0,200,0),'B':(255,0,0),'?':(120,120,120),
        }

    def _draw_face_grid(self, canvas, poly, face_colors_3x3):
        """绘制3x3网格并填色"""
        if face_colors_3x3 is None:
            return
        
        tl,tr,br,bl = [p.astype(np.float32) for p in poly]
        
        for i in range(3):
            for j in range(3):
                # 双线性插值求4角
                u0, u1 = j/3, (j+1)/3
                v0, v1 = i/3, (i+1)/3
                
                p00 = tl*(1-u0)*(1-v0) + tr*u0*(1-v0) + br*u0*v0 + bl*(1-u0)*v0
                p10 = tl*(1-u1)*(1-v0) + tr*u1*(1-v0) + br*u1*v0 + bl*(1-u1)*v0
                p11 = tl*(1-u1)*(1-v1) + tr*u1*(1-v1) + br*u1*v1 + bl*(1-u1)*v1
                p01 = tl*(1-u0)*(1-v1) + tr*u0*(1-v1) + br*u0*v1 + bl*(1-u0)*v1
                
                quad = np.int32([p00, p10, p11, p01])
                ch = face_colors_3x3[i][j] if i < len(face_colors_3x3) and j < len(face_colors_3x3[i]) else '?'
                cv2.fillConvexPoly(canvas, quad, self.colormap.get(ch, '?'))
                cv2.polylines(canvas, [quad], True, (40,40,40), 1, cv2.LINE_AA)

    def render(self, frame, rvec, state_manager):
        """渲染3D迷你魔方到右上角（总是渲染，即使rvec=None）"""
        H, W = frame.shape[:2]
        x0, y0 = W - self.size - 10, 10
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        
        # 即使rvec为None也渲染（使用单位旋转）
        if rvec is None:
            rvec = np.array([[0.0],[0.0],[0.0]], np.float32)  # 身份旋转
        
        # 虚拟tvec（固定距离）
        tvecp = np.array([[0.0],[0.0],[1.8]], np.float32)
        
        # 按深度排序（远的先画）
        R, _ = cv2.Rodrigues(rvec)
        order = []
        for f, idxs in self.face_quads.items():
            center = self.verts[idxs].mean(0)
            z = (R @ center.reshape(3,1))[2,0]
            order.append((z, f))
        order.sort()
        
        # 绘制每个面
        for _, f in order:
            idxs = self.face_quads[f]
            pts2, _ = cv2.projectPoints(self.verts[idxs], rvec, tvecp, self.Kp, self.dist)
            poly = pts2.reshape(-1,2)
            
            # 获取该面的颜色
            face_colors = state_manager.get_face_colors(f)
            self._draw_face_grid(canvas, poly, face_colors)
            cv2.polylines(canvas, [np.int32(poly)], True, (200,200,200), 2, cv2.LINE_AA)
        
        # 边框和标题
        cv2.rectangle(canvas, (0,0), (self.size-1,self.size-1), (100,100,100), 1)
        cv2.putText(canvas, "Mini Cube", (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)
        
        frame[y0:y0+self.size, x0:x0+self.size] = canvas
        return frame

