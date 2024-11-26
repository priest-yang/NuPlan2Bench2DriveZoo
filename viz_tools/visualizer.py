import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from map_viz import VisualizationManager
from bbox_viz import VizBoxes
from tqdm import tqdm
import imageio
from datetime import datetime
import cv2

from datetime import datetime
from time import time

class ComprehensiveVisualizer:
    def __init__(self, data_root):
        self.visualization_manager = VisualizationManager(data_root)
        self.camera_names = [
            "CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2",
            "CAM_B0", "CAM_R0", "CAM_R1", "CAM_R2"
        ]

    def visualize_frame(self, frame, pred_result=None, pred_plan_result=None, pred_map_result=None):
        # 1. 可视化BEV视角
        bev_fig, bev_ax = self.visualization_manager.visualize_frame(frame, pred_result, pred_plan_result, pred_map_result)
        
        # 2. 可视化八个摄像头的bbox
        camera_images = []
        for camera_name in self.camera_names:
            camera = frame['cams'][camera_name]
            viz_boxes = VizBoxes(frame['gt_boxes'], frame['gt_names'], camera, camera_name)
            camera_fig = viz_boxes.viz(return_fig=True)
            camera_images.append(self._fig2img(camera_fig))
            plt.close(camera_fig)

        # 3. 拼接所有图像
        combined_image = self._combine_images(self._fig2img(bev_fig), camera_images)
        plt.close(bev_fig)

        return combined_image

    def _fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def _combine_images(self, bev_image, camera_images):
        """将BEV图像与相机图像组合"""
        # 调整相机图像大小
        cam_size = (400, 300)
        camera_images = [img.resize(cam_size) for img in camera_images]

        # 调整BEV图像大小
        bev_image = bev_image.resize(cam_size)

        # 创建画布
        canvas_width = cam_size[0] * 3
        canvas_height = cam_size[1] * 3
        canvas = Image.new('RGB', (canvas_width, canvas_height))

        # 放置BEV图像在中间
        bev_pos = (cam_size[0], cam_size[1])
        canvas.paste(bev_image, bev_pos)

        # 放置相机图像
        positions = [
            (cam_size[0], 0),                           # 上中
            (0, 0),                                     # 左上
            (0, cam_size[1]),                           # 左中
            (0, cam_size[1] * 2),                       # 左下
            (cam_size[0], cam_size[1] * 2),             # 下中
            (cam_size[0] * 2, 0),                       # 右上
            (cam_size[0] * 2, cam_size[1]),             # 右中
            (cam_size[0] * 2, cam_size[1] * 2)          # 右下
        ]

        for img, pos in zip(camera_images, positions):
            canvas.paste(img, pos)

        return canvas

    def save_visualization(self, frames, output_dir, pred_data, save_mode='gif', duration=0.5, start_index=0, num_frames=None):
        """
        将多个帧的可视化结果保存为图像或GIF动画。

        :param frames: 包含多个帧信息的列表
        :param output_dir: 输出目录
        :param pred_data: 预测结果数据
        :param save_mode: 保存模式，'gif'或'image'
        :param duration: 每帧的持续时间（秒），仅在GIF模式下使用
        :param start_index: 开始处理的帧索引
        :param num_frames: 要处理的帧数量，如果为None则处理所有剩余帧
        """
        if num_frames is None:
            end_index = len(frames)
        else:
            end_index = min(start_index + num_frames, len(frames))

        images = []
        for i, frame in enumerate(tqdm(frames[start_index:end_index], desc="处理帧")):
            gt_token = frame['token']
            pred_result = pred_data['results'].get(gt_token)
            pred_plan_result = pred_data['plan_results'].get(gt_token)
            pred_map_result = None

            combined_image = self.visualize_frame(frame, pred_result, pred_plan_result, pred_map_result)

            if save_mode == 'image':
                # 保存为单独的图像
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"frame_{start_index + i:04d}.png")
                combined_image.save(output_path)
            else:
                # 为GIF模式收集图像
                img_array = np.array(combined_image)
                images.append(img_array)

        if save_mode == 'gif':
            # 保存为GIF
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.fromtimestamp(time()).strftime("%Y-%m-%d_%H-%M-%S")
            output_path = os.path.join(output_dir, f"vis_{timestamp}.gif")
            imageio.mimsave(output_path, images, duration=duration)
            print(f"保存GIF动画到 {output_path}")
        else:
            print(f"保存 {len(images)} 张图像到 {output_dir}")

    def save_visualization_as_gif(self, frames, output_dir, pred_data, duration=0.5, start_index=0, num_frames=None):
        """
        将多个帧的可视化结果保存为带有时间戳的GIF动画。

        :param frames: 包含多个帧信息的列表
        :param output_dir: 输出目录
        :param pred_data: 预测结果数据
        :param duration: 每帧的持续时间（秒）
        :param start_index: 开始处理的帧索引
        :param num_frames: 要处理的帧数量，如果为None则处理所有剩余帧
        """
        if num_frames is None:
            end_index = len(frames)
        else:
            end_index = min(start_index + num_frames, len(frames))

        images = []
        for frame in tqdm(frames[start_index:end_index], desc="处理帧"):
            gt_token = frame['token']
            pred_result = pred_data['results'].get(gt_token)
            pred_plan_result = pred_data['plan_results'].get(gt_token)
            pred_map_result = None

            combined_image = self.visualize_frame(frame, pred_result, pred_plan_result, pred_map_result)

            img_array = np.array(combined_image)
            images.append(img_array)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 保存为GIF
        timestamp = datetime.fromtimestamp(frames[start_index]['timestamp'] / 1e6)  # 使用第一帧的时间戳
        current_time = time()
        timestamp_str = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d_%H-%M-%S")  # 使用当前时间
        output_path = os.path.join(output_dir, f"vis_{timestamp_str}.gif")
        imageio.mimsave(output_path, images, duration=duration)
        print(f"保存GIF动画到 {output_path}")

# 使用示例
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成综合可视化GIF或图像')
    parser.add_argument('--data_root', type=str, default='/data/nuplan/dataset/', help='数据根目录')
    parser.add_argument('--ann_data_path', type=str, default="vis_tools/demo/gt_sampled_300.pkl", help='注释数据路径')
    parser.add_argument('--pred_pickle_path', type=str, default="vis_tools/demo/pred_sampled_300.pkl", help='预测数据路径')
    parser.add_argument('--output_dir', type=str, default="vis_tools/demo", help='输出目录')
    parser.add_argument('--start_frame', type=int, default=0, help='开始处理的帧索引')
    parser.add_argument('--num_frames', type=int, default=None, help='要处理的帧数量')
    parser.add_argument('--duration', type=float, default=0.5, help='GIF中每帧的持续时间（秒）')
    parser.add_argument('--save_mode', type=str, choices=['gif', 'image'], default='gif', help='保存模式：gif或image')

    args = parser.parse_args()

    # 加载数据
    with open(args.ann_data_path, 'rb') as f:
        data = pickle.load(f)
    infos, metadata = data['infos'], data['metadata']

    with open(args.pred_pickle_path, 'rb') as f:
        pred_data = pickle.load(f)

    visualizer = ComprehensiveVisualizer(args.data_root)

    # 可视化并保存
    visualizer.save_visualization(infos, args.output_dir, pred_data, 
                                  save_mode=args.save_mode,
                                  duration=args.duration, 
                                  start_index=args.start_frame, 
                                  num_frames=args.num_frames)
