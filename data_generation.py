"""
data_generation.py
====================
Sinh dữ liệu huấn luyện cho ANN bằng cách cho Minimax (dùng heuristic tay)
tự đấu với chính nó nhiều trận. Tại MỖI trạng thái bàn cờ đi qua trong trận,
ta tính sẵn điểm heuristic THẬT (bằng hàm tay) và lưu lại làm "nhãn" (label).

  -> Input  (X) : vector 36 chiều của bàn cờ
  -> Output (y) : điểm heuristic tương ứng (do hàm tay tính ra)

Mục tiêu: ANN sẽ học để BẮT CHƯỚC hàm heuristic tay, để sau đó ta có thể
dùng ANN thay cho hàm tay trong Minimax (đúng yêu cầu Phần 3 đề bài).

Có thêm yếu tố ngẫu nhiên (epsilon) khi chọn nước đi để bàn cờ đi qua
nhiều trạng thái đa dạng hơn, tránh data chỉ tập trung vào 1-2 lối chơi.
"""

import random
import numpy as np
from gomoku_env import GomokuBoard
from heuristic import evaluate_heuristic
from minimax_ai import get_best_move


def play_one_self_play_game(depth: int = 2, epsilon: float = 0.3, max_moves: int = 36):
    """
    Chơi 1 trận Minimax-vs-Minimax (cùng dùng heuristic tay), có epsilon%
    khả năng đi ngẫu nhiên thay vì đi nước tốt nhất -> tăng đa dạng trạng thái.

    Trả về list các tuple (board_vector, heuristic_score).
    """
    board = GomokuBoard()
    samples = []

    while not board.is_terminal() and len(board.move_history) < max_moves:
        # Lưu lại trạng thái TRƯỚC khi đi (kèm nhãn = heuristic của chính trạng thái này)
        samples.append((board.to_vector(), evaluate_heuristic(board)))

        moves = board.candidate_moves(radius=1)
        if not moves:
            break

        if random.random() < epsilon:
            move = random.choice(moves)
        else:
            move = get_best_move(board, depth=depth, eval_fn=evaluate_heuristic)
            if move is None:
                move = random.choice(moves)

        board.make_move(*move)

    # Lưu luôn trạng thái cuối (thắng/thua/hòa) để ANN cũng học được các thế kết thúc
    samples.append((board.to_vector(), evaluate_heuristic(board)))
    return samples


def generate_dataset(n_games: int = 400, depth: int = 2, epsilon: float = 0.3, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    all_X, all_y = [], []
    for i in range(n_games):
        samples = play_one_self_play_game(depth=depth, epsilon=epsilon)
        for vec, score in samples:
            all_X.append(vec)
            all_y.append(score)
        if (i + 1) % 50 == 0:
            print(f"  Đã chơi {i + 1}/{n_games} trận, tổng {len(all_X)} mẫu...")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    return X, y


if __name__ == "__main__":
    import time

    t0 = time.time()
    X, y = generate_dataset(n_games=400, depth=2, epsilon=0.3)
    t1 = time.time()

    print(f"\nTổng số mẫu: {X.shape[0]}, thời gian sinh data: {t1 - t0:.1f}s")
    print("X shape:", X.shape, "| y shape:", y.shape)
    print("y min/max/mean:", y.min(), y.max(), y.mean())

    np.savez_compressed("gomoku_dataset.npz", X=X, y=y)
    print("Đã lưu dataset vào gomoku_dataset.npz")
