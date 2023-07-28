import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_threshold_model(depth_distances, approaching_status):
    # ロジスティック回帰モデルを初期化
    model = LogisticRegression()

    # 特徴量として壁からの距離を使用
    X = depth_distances.reshape(-1, 1)

    # ラベルとして接近状態を使用
    y = approaching_status

    # データを使ってモデルを学習
    model.fit(X, y)

    return model

def is_approaching_wall_prob(depth_distance, model):
    # ロジスティック回帰モデルから確率を予測
    prob_approaching = model.predict_proba(np.array([[depth_distance]]))[:, 1]
    return prob_approaching[0]

# 動画ファイルを開く（深度動画を表す動画）
cap_depth = cv2.VideoCapture(r'C:\Users\mikus\lab\detection\depth 2023-07-05 17-43-39.mp4')

# 動画のフレームサイズとFPSの取得
frame_width = int(cap_depth.get(3))
frame_height = int(cap_depth.get(4))
fps = int(cap_depth.get(5))

# 出力動画の設定
output_filename = 'output_video1.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, fps, (2*frame_width, frame_height))

# 学習用のダミーデータ（仮想的なデータです。実際のデータに置き換えてください）
depth_distances = np.random.randint(50, 300, size=100)
approaching_status = np.random.randint(0, 2, size=100)  # 0: 非接近, 1: 接近

# ロジスティック回帰モデルを学習
model = train_threshold_model(depth_distances, approaching_status)

while cap_depth.isOpened():
    ret, depth_frame = cap_depth.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)

    # 学習済みモデルを使って接近確率を取得
    test_depth_distance = np.mean(gray_frame)  # 仮想的に平均値を使う
    approaching_prob = is_approaching_wall_prob(test_depth_distance, model)

    # 閾値を動的に設定
    initial_threshold = 150
    threshold_distance = initial_threshold * (1 - approaching_prob)

    # 壁の検出
    wall_mask = gray_frame < threshold_distance

    # 可視化
    visualized_frame = depth_frame.copy()
    visualized_frame[wall_mask] = [0, 0, 255]  # 壁に接近している部分を赤色に設定

    # 入力動画と出力動画を横に連結して表示
    combined_frame = np.hstack((depth_frame, visualized_frame))

    # 出力動画にフレームを書き込み
    out.write(combined_frame)

    cv2.imshow('Depth and Output', combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_depth.release()
out.release()
cv2.destroyAllWindows()
