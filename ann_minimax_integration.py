"""
ann_minimax_integration.py
============================
Đây là phần "lõi" của Phần 3 đề bài: thay hàm heuristic tay bằng ANN
đã huấn luyện, theo đúng boilerplate trong đề:

    def get_heuristic_value(board_state):
        input_vector = preprocess(board_state)
        score = model.predict(input_vector)
        return score

LƯU Ý KỸ THUẬT QUAN TRỌNG: minimax gọi eval_fn() rất nhiều lần (mỗi nút
lá trong cây tìm kiếm). Nếu gọi `model.predict()` của Keras trực tiếp
cho từng bàn cờ một, overhead khởi tạo của Keras/TensorFlow cho MỖI lần
gọi sẽ rất lớn (thực tế đo được ~2-3 giây / 1 nước đi với depth=2 -> không
thể chạy hàng trăm trận để đánh giá).

=> Giải pháp: lấy thẳng các ma trận trọng số (weights) đã huấn luyện ra,
   rồi tự viết forward pass bằng numpy thuần. Vì kiến trúc chỉ là vài
   lớp Dense (Linear + activation), việc này rất đơn giản và nhanh hơn
   gọi model.predict() hàng trăm lần. Đây vẫn là CÙNG MỘT MẠNG ANN đã
   train bằng Keras -- chỉ là cách "thực thi" nhanh hơn cho việc tìm kiếm.
"""

import numpy as np
from tensorflow import keras
from gomoku_env import GomokuBoard
from train_ann import inverse_scale_label


def relu(x):
    return np.maximum(0, x)


class ANNHeuristic:
    """Bọc model Keras thành 1 eval_fn(board) -> float, dùng được trực tiếp
    trong minimax_ai.minimax(..., eval_fn=...) giống như evaluate_heuristic.

    Forward pass được viết lại bằng numpy thuần (đọc weights từ model Keras)
    để đủ nhanh cho việc tìm kiếm minimax gọi hàng ngàn lần."""

    def __init__(self, model_path: str = "gomoku_ann_model.keras"):
        model = keras.models.load_model(model_path)
        # Trích các lớp Dense: list of (W, b, activation_name)
        self.layers_params = []
        for layer in model.layers:
            W, b = layer.get_weights()
            activation = layer.get_config()["activation"]
            self.layers_params.append((W, b, activation))

    def _forward(self, x: np.ndarray) -> float:
        h = x
        for (W, b, activation) in self.layers_params:
            h = h @ W + b
            if activation == "relu":
                h = relu(h)
            elif activation == "tanh":
                h = np.tanh(h)
            # "linear" -> không làm gì thêm
        return float(h.flatten()[0])

    def get_heuristic_value(self, board: GomokuBoard) -> float:
        # 1. Chuyển bàn cờ về vector (input của mạng) -- giống preprocess() trong đề
        input_vector = board.to_vector()
        # 2. ANN dự đoán điểm số thế trận (forward pass numpy, trên thang đã nén)
        score_scaled = self._forward(input_vector)
        # 3. Giải nén về thang điểm thật để so sánh công bằng với heuristic tay
        return float(inverse_scale_label(np.array([score_scaled]))[0])

    def __call__(self, board: GomokuBoard) -> float:
        return self.get_heuristic_value(board)


if __name__ == "__main__":
    import time
    from heuristic import evaluate_heuristic

    ann_eval = ANNHeuristic()

    b = GomokuBoard()
    for (r, c) in [(2, 2), (2, 3), (2, 4)]:
        b.grid[r, c] = 1  # X có 3 quân sống

    print("Heuristic tay :", evaluate_heuristic(b))
    print("ANN dự đoán   :", ann_eval(b))

    # Đo tốc độ: forward pass numpy phải nhanh hơn rất nhiều so với model.predict()
    t0 = time.time()
    for _ in range(1000):
        ann_eval(b)
    t1 = time.time()
    print(f"\n1000 lần gọi ANN heuristic (numpy forward): {t1 - t0:.3f}s "
          f"({(t1 - t0):.5f}s / lần)")

