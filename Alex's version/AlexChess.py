import chess
import chess.engine
import torch
import torch.nn as nn
import torch.functional as func
from rich.live import Live
import numpy as np
stockfih = r"C:\Users\Smith_928114\Downloads\stockfish-windows-x86-64-universal\stockfish\stockfish-windows-x86-64-universal.exe"
PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,   (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,  (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,   (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}
class gloobyswiglu(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return func.silu(gate) * x
class dugismartthechessthebot(nn.Module):
    def __init__(self):
        super().__init__()
        nn.Conv2d(18,1000)
class game():
    def __init__(self):
        self.board = chess.Board()
        self.movehist = []
        self.score = 0
    def move(self,move):
        self.board.push_san(move)
        self.movehist.append(move)
        self.score = self.eval()
        if self.board.outcome() is not None:
            print(self.board.outcome())
    def eval(self):
        with chess.engine.SimpleEngine.popen_uci(stockfih) as engine:
            info = engine.analyse(self.board, chess.engine.Limit(depth=20))
            score = info["score"]
            score = score.white() if self.board.turn else score.black()
            if score.is_mate():
                mate = score.mate()
                if mate > 0:
                     return 100000 - (mate * 3000)
                else:
                    # if getting mated go like oh shi yk frfr
                    return -100000 + (abs(mate) * 3000)
            else:
                # convert centipawns into like the real pawn like the stuff in the real eval bar yk
                eval_pawns = score.score() / 100.0
                return eval_pawns
    def asatorchtensor(self):
        # the planes
        planes = torch.zeros((18, 8, 8), dtype=torch.float32)
        for square, piece in self.board.piece_map().items():
            plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            row, col = divmod(square, 8) 
            planes[plane, row, col] = 1.0

        # side to move its all 1 if white else its zero because black and that like does thing
        if self.board.turn == chess.WHITE:
            planes[12, :, :] = 1.0
        # castle rights
        planes[13, :, :] = float(self.board.has_kingside_castling_rights(chess.WHITE))
        planes[14, :, :] = float(self.board.has_queenside_castling_rights(chess.WHITE))
        planes[15, :, :] = float(self.board.has_kingside_castling_rights(chess.BLACK))
        planes[16, :, :] = float(self.board.has_queenside_castling_rights(chess.BLACK))
        # en passant dimension
        if self.board.ep_square is not None:
            row, col = divmod(self.board.ep_square, 8)
            planes[17, row, col] = 1.0
        return planes

larpchess = game()
while True:
    while True:
        print(larpchess.board)
        try:
            larpchess.move(input("White:"))
            break
        except chess.IllegalMoveError:
            print("Illegal Move!")
        except chess.InvalidMoveError:
            print("Invalid Move.")
    while True:
        print("\n".join("".join(reversed(line)) for line in reversed(str(larpchess.board).splitlines())))
        try:
            larpchess.move(input("Black:"))
            break
        except chess.IllegalMoveError:
            print("Illegal Move!")
        except chess.InvalidMoveError:
            print("Invalid Move.")
