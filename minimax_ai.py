"""
minimax_ai.py
==============
Minimax + Alpha-Beta Pruning cho Gomoku.

Điểm quan trọng nhất của file này: hàm `minimax()` nhận vào một
`eval_fn` (hàm lượng giá) làm tham số -> đây chính là "khe cắm" để
Phần 3 của đồ án thay heuristic tay bằng ANN, đúng như boilerplate
trong đề bài:

    def get_heuristic_value(board_state):
        input_vector = preprocess(board_state)
        score = model.predict(input_vector)
        return score

Ở đây `eval_fn` đóng vai trò của `get_heuristic_value`.
"""

import math
from gomoku_env import GomokuBoard


def minimax(board: GomokuBoard, depth: int, alpha: float, beta: float,
            maximizing: bool, eval_fn):
    """
    board       : GomokuBoard hiện tại
    depth       : độ sâu còn lại được phép tìm
    alpha, beta : ngưỡng cắt nhánh Alpha-Beta
    maximizing  : True nếu lượt này là X (maximizing), False nếu là O
    eval_fn     : hàm (board) -> float, dùng khi hết depth hoặc bàn kết thúc
    """
    if board.is_terminal() or depth == 0:
        return eval_fn(board), None

    moves = board.candidate_moves(radius=1)
    if not moves:
        return eval_fn(board), None

    best_move = None

    if maximizing:
        best_value = -math.inf
        for (r, c) in moves:
            board.make_move(r, c)
            value, _ = minimax(board, depth - 1, alpha, beta, False, eval_fn)
            board.undo_move()
            if value > best_value:
                best_value = value
                best_move = (r, c)
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break  # cắt nhánh (beta cut-off)
        return best_value, best_move
    else:
        best_value = math.inf
        for (r, c) in moves:
            board.make_move(r, c)
            value, _ = minimax(board, depth - 1, alpha, beta, True, eval_fn)
            board.undo_move()
            if value < best_value:
                best_value = value
                best_move = (r, c)
            beta = min(beta, best_value)
            if alpha >= beta:
                break  # cắt nhánh (alpha cut-off)
        return best_value, best_move


def get_best_move(board: GomokuBoard, depth: int, eval_fn):
    """Trả về nước đi tốt nhất cho người chơi hiện tại (board.current_player)."""
    maximizing = (board.current_player == 1)
    _, move = minimax(board, depth, -math.inf, math.inf, maximizing, eval_fn)
    if move is None:
        # fallback an toàn: không tìm được nước (hiếm khi xảy ra) -> chọn ô trống đầu tiên
        legal = board.legal_moves()
        move = legal[0] if legal else None
    return move


if __name__ == "__main__":
    from heuristic import evaluate_heuristic
    import time

    b = GomokuBoard()
    # Test 1: X đã có 3 quân sống ngang, O chưa chặn -> Minimax phải chọn đi tiếp
    # để thắng (hoặc chặn nếu là O), không đi lung tung.
    b.make_move(2, 2)  # X
    b.make_move(0, 0)  # O (nước đi ngẫu nhiên, không liên quan)
    b.make_move(2, 3)  # X
    b.make_move(0, 1)  # O
    b.make_move(2, 4)  # X -> X đang có 3 quân sống (2,2)-(2,3)-(2,4)
    b.render()
    print("Đến lượt:", "X" if b.current_player == 1 else "O")

    t0 = time.time()
    move = get_best_move(b, depth=3, eval_fn=evaluate_heuristic)
    t1 = time.time()
    print("O nên chặn ở:", move, f"(thời gian: {t1 - t0:.3f}s)")
    # Kỳ vọng: O phải chặn ở (2,1) hoặc (2,5) để không thua ngay
