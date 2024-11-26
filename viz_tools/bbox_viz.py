import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from enum import Enum
from scipy.spatial.transform import Rotation
import os
import cv2

class BoxVisibility(Enum):
    ALL = 0
    ANY = 1
    NONE = 2

def load_pkl_file(pkl_path):
    """加载pickle文件"""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

class Box3D:
    def __init__(self, box, intrinsic: np.ndarray, extrinsic: np.ndarray, imsize: Tuple[float, float], category: str):
        """
        初始化3D边界框
        
        :param box: 边界框参数 [x, y, z, w, l, h, yaw]
        :param intrinsic: 相机内参矩阵
        :param extrinsic: 相机外参矩阵
        :param imsize: 图像尺寸 (宽, 高)
        :param category: 类别
        """
        self.box = box
        self.wlh = np.array(box[3:6])
        self.loc = np.array(box[:3])
        self.yaw = box[6]
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.imsize = imsize
        self.category = category

    @staticmethod
    def create_extrinsic_matrix(translation, rotation_quat):
        """
        创建4x4的外参矩阵
        
        :param translation: 平移向量
        :param rotation_quat: 旋转四元数
        :return: 4x4外参矩阵
        """
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = Rotation.from_quat(np.array([rotation_quat[1], rotation_quat[2], rotation_quat[3], rotation_quat[0]])).as_matrix()
        extrinsic[:3, 3] = translation
        return extrinsic
        
    def box_in_image(self, box, vis_level: BoxVisibility = BoxVisibility.ANY) -> bool:
        """
        检查边界框是否在图像中可见（不考虑遮挡）
        
        :param box: 边界框
        :param vis_level: 可见性级别
        :return: 是否可见
        """
        corners_3d = self.corners()
        corners_img = self.view_points(corners_3d)[:2, :]

        visible = np.logical_and.reduce([
            corners_img[0, :] > 0,
            corners_img[0, :] < self.imsize[0],
            corners_img[1, :] < self.imsize[1],
            corners_img[1, :] > 0
        ])

        if vis_level == BoxVisibility.ALL:
            return np.all(visible)
        elif vis_level == BoxVisibility.ANY:
            return np.any(visible)
        elif vis_level == BoxVisibility.NONE:
            return True
        else:
            raise ValueError(f"vis_level: {vis_level} 不是有效值")
        
    def view_points(self, points_3d) -> np.ndarray:
        """
        将3D点投影到2D图像平面上
        
        :param points_3d: 3xN的numpy数组，表示3D空间中的点
        :return: 2xN的numpy数组，表示投影后的2D点坐标
        """
        assert points_3d.shape[0] == 3, "points_3d应该是3xN的矩阵"
        
        points_3d_homogeneous = np.vstack([points_3d, np.ones((1, points_3d.shape[1]))])
        points_camera = np.linalg.inv(self.extrinsic) @ points_3d_homogeneous
        points_image = self.intrinsic @ points_camera[:3, :]

        points_image[0, :] /= points_image[2, :]
        points_image[1, :] /= points_image[2, :]

        if np.any(points_image[2, :] < 0):
            return np.array([[-1] * 8, [-1] * 8])

        return points_image[:2, :]
        
    @property
    def rotation_matrix(self):
        """返回边界框的旋转矩阵"""
        return np.array([
            [np.cos(-self.yaw), -np.sin(-self.yaw), 0],
            [np.sin(-self.yaw), np.cos(-self.yaw), 0],
            [0, 0, 1]
        ])
        
    def corners(self, wlh_factor: float = 1.0):
        """
        计算边界框的角点
        
        :param wlh_factor: 宽度、长度和高度的缩放因子
        :return: 3x8的numpy数组，表示8个角点的3D坐标
        """
        w, l, h = self.wlh * wlh_factor
        center = tuple(self.loc)
        rotation_matrix = self.rotation_matrix
        return self._calc_corners(w, l, h, center, rotation_matrix)
        
    def _calc_corners(self, w, l, h, center, rotation_matrix):
        """
        辅助方法，计算边界框的角点
        
        :param w: 宽度
        :param l: 长度
        :param h: 高度
        :param center: 中心点
        :param rotation_matrix: 旋转矩阵
        :return: 3x8的numpy数组，表示8个角点的3D坐标
        """
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]

        corners_3d = np.dot(rotation_matrix, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]

        return corners_3d
        
    def viz(self, ax, color_map):
        """
        在给定的轴上可视化边界框
        
        :param ax: matplotlib轴对象
        :param color: 边界框的颜色
        """
        corners_3d = self.corners()
        corners_img = self.view_points(corners_3d)[:2, :]

        color = color_map.get(self.category, 'r')  # 默认为红色
        
        for i in range(4):
            ax.plot([corners_img[0, i], corners_img[0, (i + 1) % 4]], 
                    [corners_img[1, i], corners_img[1, (i + 1) % 4]], color=color)
            ax.plot([corners_img[0, i + 4], corners_img[0, ((i + 1) % 4) + 4]], 
                    [corners_img[1, i + 4], corners_img[1, ((i + 1) % 4) + 4]], color=color)

        for i in range(4):
            ax.plot([corners_img[0, i], corners_img[0, i + 4]], 
                    [corners_img[1, i], corners_img[1, i + 4]], color=color)

    def viz_cv2(self, img, color=(0, 0, 255), thickness=3):
        """
        使用OpenCV在图像上可视化边界框
        
        :param img: OpenCV图像对象
        :param color: 边界框的颜色
        :param thickness: 线条粗细
        :return: 绘制了边界框的图像
        """
        corners_3d = self.corners()
        corners_img = self.view_points(corners_3d)[:2, :]
        corners_img = corners_img.astype(np.int32)

        # 绘制底部矩形
        for i in range(4):
            cv2.line(img, tuple(corners_img[:, i]), tuple(corners_img[:, (i+1)%4]), color, thickness)
        
        # 绘制顶部矩形
        for i in range(4):
            cv2.line(img, tuple(corners_img[:, i+4]), tuple(corners_img[:, ((i+1)%4)+4]), color, thickness)
        
        # 连接顶部和底部
        for i in range(4):
            cv2.line(img, tuple(corners_img[:, i]), tuple(corners_img[:, i + 4]), color, thickness)
        
        return img

class VizBoxes:
    def __init__(self, boxes, categories, camera, camera_name):
        """
        初始化可视化对象
        
        :param boxes: 边界框列表
        :param categories: 类别列表
        :param camera: 相机参数
        :param camera_name: 相机名称
        """
        self.boxes = boxes
        self.categories = categories
        self.camera = camera
        self.camera_name = camera_name
        self.pil_img = Image.open(camera['data_path'])
        self.distortion = np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231])

        translation = camera['sensor2lidar_translation']
        rotation_quat = camera['sensor2ego_rotation']
        extrinsic = Box3D.create_extrinsic_matrix(translation, rotation_quat)

        self.boxes_3d = [Box3D(box, camera['cam_intrinsic'], extrinsic, self.pil_img.size, category) for box, category in zip(boxes, categories)]

        self.color_map = {
            'vehicle': 'r',             # red
            'bicycle': 'g',             # green
            'traffic_cone': 'b',        # blue
            'barrier': 'm',             # magenta
            'czone_sign': 'y',          # yellow
            'generic_object': 'c',      # cyan
            'pedestrian': 'k'           # black
        }
        
    
    def undistort_image(self):
        """对图像进行去畸变处理"""
        img_array = np.array(self.pil_img)
        undistorted_img = cv2.undistort(img_array, self.camera['cam_intrinsic'], self.distortion)
        return Image.fromarray(undistorted_img)

    def viz(self, save_path: Optional[str] = None, return_fig: bool = False):
        fig, ax = plt.subplots(figsize=(12, 9))
        undistorted_img = self.undistort_image()
        ax.imshow(undistorted_img)
        # ax.set_title(self.camera_name)

        for box_3d in self.boxes_3d:
            if box_3d.box_in_image(box_3d.box, BoxVisibility.ALL):
                box_3d.viz(ax, self.color_map)

        # 移除坐标轴和边框
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        if return_fig:
            return fig
        elif save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"已保存图像: {save_path}")
        else:
            plt.show()

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化3D边界框')
    parser.add_argument('--pkl_path', type=str, default="vis_tools/demo/gt_sampled_300.pkl", help='pickle文件路径')
    parser.add_argument('--output_dir', type=str, default="vis_out", help='输出目录')
    parser.add_argument('--num_samples', type=int, default=300, help='要处理的样本数量')
    args = parser.parse_args()

    data = load_pkl_file(args.pkl_path)
    camera_names = [
        "CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2",
        "CAM_B0", "CAM_R0", "CAM_R1", "CAM_R2"
    ]

    for i in range(min(len(data['infos']), args.num_samples)):
        token = data['infos'][i]['token']
        boxes = data['infos'][i]['gt_boxes']
        categories = data['infos'][i]['gt_names']
        for camera_name in camera_names:
            camera = data['infos'][i]['cams'][camera_name]
            viz_boxes = VizBoxes(boxes, categories, camera, camera_name)
            viz_boxes.viz(f"{args.output_dir}/{token}/{camera_name}.png")
    
    print(f"已处理 {min(len(data['infos']), args.num_samples)} 个样本")
