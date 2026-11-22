import io
import time
import zstandard
import chess.pgn
import chess
import chess.engine
import json

GAMES = 15000
DEPTH = 8           
THREADS = 4          
OUT = "data.json"

path = r"C:\Users\princ\OneDrive\Documents\gpuCode\lichess_db_standard_rated_2014-12.pgn.zst"
StockFishPath = r"C:\Users\princ\OneDrive\Documents\gpuCode\stockfish\stockfish-windows-x86-64-universal.exe"

engine = chess.engine.SimpleEngine.popen_uci(StockFishPath)
engine.configure({"Threads": THREADS, "Hash": 256})   # el speed boost

data = []
start = time.time()

with open(path, "rb") as f:
    # decompres, turns to text
    stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8")

    for x in range(GAMES):
        game = chess.pgn.read_game(stream)
        if game is None:      # ran out of games in the file
            print("NO MORE GAMES LEFT LOL")
        print(x, game.headers["White"], "vs", game.headers["Black"], len(list(game.mainline_moves())), "moves")

        board = game.board()

        for move in game.mainline_moves():
            board.push(move)
            if board.is_game_over():
                continue
            turn = ""
            if board.turn == chess.WHITE:
                turn = "white"
            else:
                turn = "black"
            info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            fen = board.fen()
            best = info["pv"][0].uci()

            data.append({"fen" : fen,"best" : best,"turn" : turn,})
        #every 250 game
        if x % 250 == 249:
            with open(OUT, "w") as out:
                json.dump(data, out)
            elapsed = time.time() - start
            left = elapsed / (x + 1) * (GAMES - x - 1)
            print(f"saved {len(data)} positions | {elapsed/60:.0f} min so far | ~{left/60:.0f} min left")

engine.quit()
with open(OUT, "w") as f:
    json.dump(data, f)

print(f"Done: {len(data)} positions") 