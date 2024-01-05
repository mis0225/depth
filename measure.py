"""
指定座標の深度を推測して出力するプログラム
"""

#!/usr/bin/env python
import time
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import tkinter as tk

def mouse_callback(event, x, y, flags, param):
    # マウス座標 保持用のグローバル変数
    global mouse_point
    mouse_point = [x, y]

def run_inference(model, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # リサイズ
    resize_image = cv2.resize(image, (256, 256))
    
    # 正規化
    resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB) / 255.0
    
    # 形状変更
    resize_image = resize_image.transpose(2, 0, 1)
    resize_image = resize_image.reshape(1, 3, 256, 256)
    
    # tensor形式へ変換
    tensor = tf.convert_to_tensor(resize_image, dtype=tf.float32)
    
    # 推論
    result = model(tensor)
    
    # 余分な次元を削除し、入力画像のサイズへリサイズ
    result = result['default'].numpy()
    result = np.squeeze(result)
    result_depth_map = cv2.resize(result, (image_width, image_height))
    
    return result_depth_map

if __name__ == "__main__":
    # マウス座標保持用のグローバル変数
    global mouse_point
    mouse_point = None

    # モデルロード
    module = hub.load("https://tfhub.dev/intel/midas/v2_1_small/1", tags=['serve'])
    model = module.signatures['serving_default']

    ## カメラ準備(webカメラの場合はここのコメントを外す) ##
    #cap = cv2.VideoCapture(0)
    
    # 動画ファイルを開く
    video_path = r"C:\Users\mikus\Videos\Captures\motion.mp4"
    cap = cv2.VideoCapture(video_path)

    # Tkinter初期化
    root = tk.Tk()
    root.withdraw()

    # OpenCVウィンドウ初期化、マウス操作用のコールバックを登録
    rgb_window_name = 'rgb'
    cv2.namedWindow(rgb_window_name)
    cv2.setMouseCallback(rgb_window_name, mouse_callback)

    while True:
        start_time = time.time()

        # カメラキャプチャ
        ret, frame = cap.read()
        if not ret:
            continue

        # Depth推定
        depth_map = run_inference(model, frame)

        elapsed_time = time.time() - start_time

        # 情報描画
        rgb_frame = frame  # RGB画像はそのまま使用

        if mouse_point is not None:
            point_x, point_y = mouse_point
            depth_value = depth_map[point_y][point_x]

            # Depth画像
            cv2.circle(rgb_frame, (point_x, point_y),
                       3, (0, 255, 0), thickness=1)
            cv2.putText(rgb_frame, f"Depth: {depth_value:.2f}", (point_x, point_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                        cv2.LINE_AA)

        cv2.imshow(rgb_window_name, rgb_frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

