"""
evaluate_models.py
====================
Đánh giá xem ANN-Minimax có "chơi thông minh hơn ngẫu nhiên" hay không
-- đây chính là tiêu chí rubric của đề bài cho Phần 3 (không cần thắng
tuyệt đối Minimax-tay, chỉ cần hơn random rõ rệt là đạt mục tiêu).

3 cặp đấu được đánh giá:
  1. ANN-Minimax      vs Random           (kiểm tra ANN có học được gì không)
  2. Minimax-tay      vs Random           (baseline để so sánh)
  3. ANN-Minimax      vs Minimax-tay      (xem ANN gần heuristic tay tới đâu)

Mỗi cặp đấu N trận, đổi vai trò X/O luân phiên để công bằng (loại bỏ
lợi thế đi trước cố định).
"""

import random
import numpy as np
from gomoku_env import GomokuBoard
from heuristic import evaluate_heuristic
from minimax_ai import get_best_move
from ann_minimax_integration import ANNHeuristic


def random_move(board: GomokuBoard):
    moves = board.candidate_moves(radius=1)
    return random.choice(moves) if moves else None


def play_match(agent_x, agent_o, max_moves: int = 36):
    """
    agent_x, agent_o: hàm (board) -> (row, col) trả về nước đi cho từng bên.
    Trả về: 1 nếu X thắng, -1 nếu O thắng, 0 nếu hòa.
    """
    board = GomokuBoard()
    while not board.is_terminal() and len(board.move_history) < max_moves:
        agent = agent_x if board.current_player == 1 else agent_o
        move = agent(board)
        if move is None:
            break
        board.make_move(*move)
    return board.winner if board.winner is not None else 0


def make_minimax_agent(eval_fn, depth: int):
    return lambda board: get_best_move(board, depth=depth, eval_fn=eval_fn)


def run_evaluation(n_games_per_side: int, ann_eval, depth_ann: int = 2, depth_handcrafted: int = 2):
    """
    Mỗi cặp đối thủ chơi `n_games_per_side` trận với agent A đi X,
    và `n_games_per_side` trận với agent A đi O -> tổng 2*n_games_per_side trận.
    """
    random.seed(123)

    ann_agent = make_minimax_agent(ann_eval, depth_ann)
    handcrafted_agent = make_minimax_agent(evaluate_heuristic, depth_handcrafted)

    def run_pair(name_a, agent_a, name_b, agent_b, n):
        a_wins, b_wins, draws = 0, 0, 0
        for _ in range(n):
            result = play_match(agent_a, agent_b)
            if result == 1:
                a_wins += 1
            elif result == -1:
                b_wins += 1
            else:
                draws += 1
        for _ in range(n):
            result = play_match(agent_b, agent_a)
            if result == 1:
                b_wins += 1
            elif result == -1:
                a_wins += 1
            else:
                draws += 1
        total = 2 * n
        print(f"\n{name_a} vs {name_b}  ({total} trận, đổi vai X/O luân phiên)")
        print(f"  {name_a} thắng: {a_wins} ({a_wins/total:.0%})")
        print(f"  {name_b} thắng: {b_wins} ({b_wins/total:.0%})")
        print(f"  Hòa          : {draws} ({draws/total:.0%})")
        return a_wins, b_wins, draws

    print("=" * 60)
    print("ĐÁNH GIÁ 1: ANN-Minimax vs Random")
    run_pair("ANN-Minimax", ann_agent, "Random", random_move, n_games_per_side)

    print("=" * 60)
    print("ĐÁNH GIÁ 2 (baseline): Minimax-tay vs Random")
    run_pair("Minimax-tay", handcrafted_agent, "Random", random_move, n_games_per_side)

    print("=" * 60)
    print("ĐÁNH GIÁ 3: ANN-Minimax vs Minimax-tay")
    run_pair("ANN-Minimax", ann_agent, "Minimax-tay", handcrafted_agent, n_games_per_side)


if __name__ == "__main__":
    import time
    ann_eval = ANNHeuristic()
    t0 = time.time()
    run_evaluation(n_games_per_side=30, ann_eval=ann_eval, depth_ann=2, depth_handcrafted=2)
    print(f"\nTổng thời gian đánh giá: {time.time() - t0:.1f}s")
