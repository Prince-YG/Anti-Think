import json
import chess
import torch
import torch.nn as nn


width = 1028
learningRate =3e-4
ephocs = 100              
BATCH = 512      
VAL_SPLIT = 0.1     

data = json.load(open("data.json"))
gpu = torch.device("cuda")

inputs = []
outputs =[]



peice2plane = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

def TensorAsBoard(board):

    t = torch.zeros(773)
    for square, piece in board.piece_map().items():

        #gets plane fro the peice and color
        plane = peice2plane[(piece.piece_type, piece.color)]

        #activates square
        t[plane * 64 + square] = 1.0

    #thanks g for this neat formating here
    t[768] = 1.0 if board.turn == chess.WHITE else 0.0
    t[769] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    t[770] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    t[771] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    t[772] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    return t

for x in data:
    board = chess.Board(x["fen"])
    inputs.append(TensorAsBoard(board))

    best = chess.Move.from_uci(x["best"])

    #getTheIndex/slot? of the move (* 64 for its layer, then add for its index on the layer)
    outputs.append((best.from_square *64)+ best.to_square)
inputs = torch.stack(inputs).to(gpu)              
outputs = torch.tensor(outputs, dtype=torch.long).to(gpu)  

# NEW: shuffle, then split into train / val
perm = torch.randperm(len(inputs), device=gpu)
inputs, outputs = inputs[perm], outputs[perm]
split = int(len(inputs) * (1 - VAL_SPLIT))
train_x, train_y = inputs[:split], outputs[:split]
val_x, val_y = inputs[split:], outputs[split:]

print(train_x.shape, val_x.shape)




class AntiThink(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(773, width),  
            nn.ReLU(), 
            nn.Dropout(0.3),       
            nn.Linear(width, 4096),  
        )

    def forward(self, x):
        return self.model(x)
    
model = AntiThink().to(gpu)

lossFunc = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate, weight_decay=0.05)

for epoch in range(ephocs):


    model.train()  
    
    order = torch.randperm(len(train_x), device=gpu)
    total_loss = 0.0

    for i in range(0, len(train_x), BATCH):
        idx = order[i:i + BATCH]              
        tests = model(train_x[idx])
        loss = lossFunc(tests, train_y[idx])
        optimizer.zero_grad()            
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)
    model.eval()
    with torch.no_grad():
        val_scores = model(val_x)
        val_loss = lossFunc(val_scores, val_y).item()
        val_acc = (val_scores.argmax(dim=1) == val_y).float().mean().item()

    # hastage w printing lol
    print(f"epoch {epoch:02d} | train loss {total_loss/len(train_x):.4f} | val loss {val_loss:.4f} | val acc {val_acc*100:.2f}%")
torch.save(model.state_dict(), "antithink.pt")