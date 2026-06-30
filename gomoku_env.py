"""
gomoku_env.py
==============
Bàn cờ Gomoku thu nhỏ: lưới 6x6, thắng khi có 4 quân liên tiếp
(ngang / dọc / chéo).

Quy ước giá trị ô:
     1  -> quân X (người/AI đi trước)
    -1  -> quân O (người/AI đi sau)
     0  -> ô trống

Đây là phần các bạn PHẢI tự hiểu rõ (không phải phần "xin AI sinh code rồi để đó"),
vì toàn bộ Phần 3 của đồ án phụ thuộc vào cách biểu diễn trạng thái này.
"""

import numpy as np

BOARD_SIZE = 6
WIN_LEN = 4


class GomokuBoard:
    def __init__(self, size: int = BOARD_SIZE, win_len: int = WIN_LEN):
        self.size = size
        self.win_len = win_len
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.current_player = 1   # X đi trước
        self.winner = None        # None = chưa kết thúc, 0 = hòa, 1/-1 = ai thắng
        self.move_history = []

    # ---------- Tiện ích cơ bản ----------
    def clone(self):
        new_board = GomokuBoard(self.size, self.win_len)
        new_board.grid = self.grid.copy()
        new_board.current_player = self.current_player
        new_board.winner = self.winner
        new_board.move_history = self.move_history.copy()
        return new_board

    def is_full(self) -> bool:
        return not np.any(self.grid == 0)

    def to_vector(self) -> np.ndarray:
        """Chuyển bàn cờ thành vector 1D (36,) làm input cho ANN."""
        return self.grid.flatten().astype(np.float32)

    # ---------- Sinh nước đi ----------
    def legal_moves(self):
        """Trả về list (row, col) các ô còn trống."""
        rows, cols = np.where(self.grid == 0)
        return list(zip([int(r) for r in rows], [int(c) for c in cols]))

    def candidate_moves(self, radius: int = 1):
        """
        Giới hạn không gian tìm kiếm: chỉ xét các ô trống nằm trong bán kính
        `radius` quanh các quân đã đặt. Nếu bàn cờ trống -> chỉ xét ô trung tâm.
        Đây là kỹ thuật cắt giảm nhánh thực tế (move ordering / pruning thô),
        bắt buộc phải có để Minimax chạy đủ nhanh trên lưới 6x6.
        """
        if np.all(self.grid == 0):
            c = self.size // 2
            return [(c, c)]

        occupied = np.argwhere(self.grid != 0)
        candidates = set()
        for (r, c) in occupied:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if self.grid[nr, nc] == 0:
                            candidates.add((nr, nc))
        if not candidates:  # bàn đầy nhưng chưa ai thắng (hòa)
            return []
        return list(candidates)

    # ---------- Thực hiện nước đi ----------
    def make_move(self, row: int, col: int):
        if self.grid[row, col] != 0:
            raise ValueError(f"Ô ({row},{col}) đã có quân.")
        self.grid[row, col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        self._check_winner_at(row, col)
        self.current_player *= -1  # đổi lượt

    def undo_move(self):
        if not self.move_history:
            return
        r, c, _ = self.move_history.pop()
        self.grid[r, c] = 0
        self.winner = None
        self.current_player *= -1

    # ---------- Kiểm tra thắng ----------
    def _check_winner_at(self, row: int, col: int):
        """Chỉ cần kiểm tra 4 hướng đi qua ô vừa đánh (nhanh hơn quét toàn bàn)."""
        player = self.grid[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # đi theo chiều dương
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.grid[r, c] == player:
                count += 1
                r += dr
                c += dc
            # đi theo chiều âm
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.grid[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= self.win_len:
                self.winner = int(player)
                return
        if self.is_full():
            self.winner = 0  # hòa

    def is_terminal(self) -> bool:
        return self.winner is not None

    # ---------- Hiển thị ----------
    def render(self):
        symbols = {1: "X", -1: "O", 0: "."}
        for row in self.grid:
            print(" ".join(symbols[v] for v in row))
        print()


if __name__ == "__main__":
    # Test nhanh: tạo bàn cờ và đánh thử 1 đường thắng ngang
    b = GomokuBoard()
    b.render()
    moves = [(2, 2), (3, 2), (2, 3), (3, 3), (2, 4), (3, 4), (2, 5)]
    for (r, c) in moves:
        b.make_move(r, c)
    b.render()
    print("Người thắng:", b.winner)  # kỳ vọng: 1 (X thắng vì 2,2-2,3-2,4-2,5)
    print("Candidate moves count:", len(b.candidate_moves()))
