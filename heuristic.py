"""
heuristic.py
=============
Hàm lượng giá (heuristic) THỦ CÔNG cho Gomoku — đây chính là phần mà đề bài
yêu cầu thay thế bằng ANN ở Phần 3. Phải hiểu rõ hàm này để:
  1. Biết ANN "đang học" để bắt chước cái gì.
  2. Trả lời phản biện vì sao chọn cách tính điểm này.

Ý tưởng: quét tất cả các "cửa sổ" độ dài WIN_LEN (4 ô) theo 4 hướng
(ngang, dọc, 2 chéo). Với mỗi cửa sổ chỉ có quân của 1 bên (bên kia chưa
chặn), tính điểm theo số quân đã có trong cửa sổ đó — cửa sổ có 3 quân
sống thì nguy hiểm hơn cửa sổ có 1 quân.

Điểm trả về luôn được tính theo góc nhìn của X (số dương = lợi cho X,
số âm = lợi cho O), giống quy ước minimax maximizing=X, minimizing=O.
"""

import numpy as np
from gomoku_env import GomokuBoard, BOARD_SIZE, WIN_LEN

# Trọng số cho từng số quân trong 1 cửa sổ 4 ô còn khả năng thắng.
# Tăng phi tuyến để nhấn mạnh thế cờ gần thắng nguy hiểm hơn nhiều.
WINDOW_SCORE = {0: 0, 1: 1, 2: 8, 3: 40, 4: 100000}


def _all_windows(size: int, win_len: int):
    """Sinh danh sách các cửa sổ (list toạ độ) theo 4 hướng trên lưới size x size."""
    windows = []
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        for r in range(size):
            for c in range(size):
                cells = []
                for k in range(win_len):
                    rr, cc = r + dr * k, c + dc * k
                    if 0 <= rr < size and 0 <= cc < size:
                        cells.append((rr, cc))
                    else:
                        break
                if len(cells) == win_len:
                    windows.append(cells)
    return windows


# Tính sẵn 1 lần (không đổi giữa các bàn cờ vì size/win_len cố định) để tăng tốc.
_WINDOWS_CACHE = _all_windows(BOARD_SIZE, WIN_LEN)


def evaluate_heuristic(board: GomokuBoard) -> float:
    """
    Trả về điểm heuristic của bàn cờ theo góc nhìn X.
    Nếu đã có người thắng/hòa thì trả điểm "tuyệt đối" luôn.
    """
    if board.winner == 1:
        return 100000.0
    if board.winner == -1:
        return -100000.0
    if board.winner == 0:
        return 0.0

    grid = board.grid
    score = 0.0
    for window in _WINDOWS_CACHE:
        values = [grid[r, c] for (r, c) in window]
        x_count = values.count(1)
        o_count = values.count(-1)
        if x_count > 0 and o_count > 0:
            continue  # cửa sổ đã bị chặn 2 bên -> không ai thắng được ở đây nữa
        if x_count > 0:
            score += WINDOW_SCORE[x_count]
        elif o_count > 0:
            score -= WINDOW_SCORE[o_count]
    return float(score)


if __name__ == "__main__":
    b = GomokuBoard()
    print("Bàn trống, điểm:", evaluate_heuristic(b))  # kỳ vọng 0

    # X có 3 quân sống liên tiếp (mối nguy lớn) -> điểm phải dương rõ rệt
    for (r, c) in [(2, 2), (2, 3), (2, 4)]:
        b.grid[r, c] = 1
    print("X có 3 quân sống liên tiếp, điểm:", evaluate_heuristic(b))

    # Thêm O chặn 1 đầu -> điểm phải giảm so với trên
    b2 = b.clone()
    b2.grid[2, 5] = -1
    print("X có 3 quân nhưng bị O chặn 1 đầu, điểm:", evaluate_heuristic(b2))
