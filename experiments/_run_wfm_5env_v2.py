"""5-Atari WFM — downsample 64x64, AE 16x16, CPU AE, MPS WFM."""
import ale_py; import gymnasium; gymnasium.register_envs(ale_py)
import numpy as np, torch, torch.nn as nn, time, os
from wmca.modules.hybrid import CML2DMultiR

ACE = torch.device('cpu'); WFM_DEV = torch.device('mps')
GRID = 16  # no 32x32 — 16x16 latents

def play_game(name, na, eps=200):
    env = gymnasium.make(f'ALE/{name}-v5')
    rng = np.random.default_rng(42)
    fs, nfs, acts = [], [], []
    for ep in range(eps):
        obs, _ = env.reset(seed=ep)
        for _ in range(50):
            a = rng.integers(0, na)
            nobs, _, t, tr, _ = env.step(a)
            g = np.dot(obs[...,:3],[0.299,0.587,0.114]).astype(np.float32)/255.
            ng = np.dot(nobs[...,:3],[0.299,0.587,0.114]).astype(np.float32)/255.
            # Downsample to 64x64 via simple averaging
            g = g.reshape(105,2,80,2).mean(axis=(1,3))  # 210x160 -> 105x80, not 64
            ng = ng.reshape(105,2,80,2).mean(axis=(1,3))
            # Better: use proper resize
            H,W = g.shape
            g = g[::H//64 or 1, ::W//64 or 1][:64,:64]  # simple stride
            ng = ng[::H//64 or 1, ::W//64 or 1][:64,:64]
            fs.append(g); nfs.append(ng); acts.append(a); obs = nobs
            if t or tr: break
        if ep%50==0: print(f'  {name}: {ep}/{eps}')
    env.close(); del env
    return np.stack(fs)[:,np.newaxis], np.stack(nfs)[:,np.newaxis], np.array(acts, dtype=np.int64)

class GAE(nn.Module):
    def __init__(self, in_ch=1, hid=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(in_ch,hid,3,1,1),nn.ReLU(),nn.Conv2d(hid,1,3,1,1),nn.Sigmoid())
        self.dec = nn.Sequential(nn.Conv2d(1,hid,3,1,1),nn.ReLU(),nn.Conv2d(hid,in_ch,3,1,1),nn.Sigmoid())
    def encode(self,x):
        return self.enc(x)
    def forward(self,x):
        return self.dec(self.enc(x))

def make_af(acts, na, H, W):
    v = (acts.astype(np.float32)+1.0)/na
    return np.tile(v[:,None,None,None],(1,1,H,W))

def pad_to(X, th, tw):
    ph = max(0, (th-X.shape[2])//2); pw = max(0, (tw-X.shape[3])//2)
    return np.pad(X,((0,0),(0,0),(ph,max(0,th-X.shape[2]-ph)),(pw,max(0,tw-X.shape[3]-pw))))

class WFM(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        self.cml = CML2DMultiR(in_channels=in_ch,K=32,steps=15,seed=42,gate_mode='uniform')
        self.nca = nn.Sequential(nn.Conv2d(in_ch*2,64,3,1,1),nn.ReLU(),nn.Conv2d(64,64,3,1,1),nn.ReLU(),nn.Conv2d(64,1,1))
    def forward(self,x):
        with torch.no_grad(): co=self.cml(x)
        return co[:,:1]+self.nca(torch.cat([x,co],dim=1))

# 3 ALE games
ALE_GAMES = [('SpaceInvaders',6),('Freeway',3),('BeamRider',9)]
HOLDOUT = 'Breakout'

data = {}

# Load existing Pong + Breakout
print("Loading existing Pong + Breakout latents...")
for game in ['pong', 'breakout']:
    lats = np.load(f'experiments/atari_data/{game}_latents.npy')
    nlats = np.load(f'experiments/atari_data/{game}_next_latents.npy')
    acts = np.load(f'experiments/atari_data/{game}_actions.npy')[:len(lats)]
    na = 3
    H,W = lats.shape[2:]
    X = pad_to(np.concatenate([lats, make_af(acts, na, H, W)], axis=1), GRID, GRID)
    Y = pad_to(nlats, GRID, GRID)
    name = game.capitalize()
    data[name] = {'X': X, 'Y': Y}
    print(f'  {name}: {len(lats)} samples, {H}x{W}')

# Generate 3 ALE games
for name, na in ALE_GAMES:
    print(f'\n--- {name} ---')
    f, nf, acts = play_game(name, na)
    print(f'  {len(acts)} frames, shape={f.shape[2]}x{f.shape[3]}')
    
    nt = int(len(f)*0.8)
    ae = GAE().to(ACE); opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    Xa = torch.from_numpy(f[:nt]).float().to(ACE)
    for _ in range(10):
        p = torch.randperm(len(Xa))
        for i in range(0, len(p), 8):
            ix = p[i:i+8]; opt.zero_grad()
            nn.functional.mse_loss(ae(Xa[ix]), Xa[ix]).backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        lats = ae.encode(torch.from_numpy(f).float().to(ACE)).numpy()
        nlats = ae.encode(torch.from_numpy(nf).float().to(ACE)).numpy()
    mse = nn.functional.mse_loss(ae(torch.from_numpy(f[:64]).float().to(ACE)), torch.from_numpy(f[:64]).float()).item()
    H,W = lats.shape[2:]
    X = pad_to(np.concatenate([lats, make_af(acts, na, H, W)], axis=1), GRID, GRID)
    Y = pad_to(nlats, GRID, GRID)
    data[name] = {'X': X, 'Y': Y}
    del ae, Xa, f, nf, lats, nlats
    print(f'  PSNR={10*np.log10(1/mse):.1f}dB, grid={H}x{W}')

# WFM pre-training
train = [g for g in data if g != HOLDOUT]
print(f'\n=== WFM: {train} -> {HOLDOUT} (16x16 grid, 100 epochs) ===')
model = WFM(in_ch=2).to(WFM_DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()
for ep in range(100):
    model.train()
    for g in train:
        Xt, Yt = data[g]['X'], data[g]['Y']
        n = min(len(Xt), 4000)
        xt = torch.from_numpy(Xt[:n]).float().to(WFM_DEV)
        yt = torch.from_numpy(Yt[:n]).float().to(WFM_DEV)
        p = torch.randperm(n, device=WFM_DEV)
        for i in range(0, n, 8):
            ix = p[i:i+8]; opt.zero_grad()
            nn.functional.mse_loss(model(xt[ix]), yt[ix]).backward(); opt.step()
    if ep%20==0 or ep==99:
        print(f'  ep {ep+1:3d}/100 [{((time.time()-t0)/60):.0f}m]')

os.makedirs('experiments/results', exist_ok=True)
torch.save(model.state_dict(), 'experiments/results/wfm_5env.pt')
print(f'Saved {(time.time()-t0)/60:.0f}m')

# Transfer to Breakout
Xb, Yb = data[HOLDOUT]['X'], data[HOLDOUT]['Y']
nft = int(len(Xb)*0.7)
Xft = torch.from_numpy(Xb[:nft]).float().to(WFM_DEV); Yft = torch.from_numpy(Yb[:nft]).float().to(WFM_DEV)
Xfv = torch.from_numpy(Xb[nft:nft+256]).float().to(WFM_DEV); Yfv = torch.from_numpy(Yb[nft:nft+256]).float().to(WFM_DEV)

model.eval()
with torch.no_grad(): z = nn.functional.mse_loss(model(Xfv), Yfv).item()
print(f'Zero-shot: {z:.2e}')

mf = WFM(in_ch=2).to(WFM_DEV); mf.load_state_dict(torch.load('experiments/results/wfm_5env.pt'))
ot = torch.optim.Adam(mf.parameters(), lr=1e-4)
for ep in range(30):
    mf.train(); p = torch.randperm(nft, device=WFM_DEV)
    for i in range(0, nft, 8):
        ix = p[i:i+8]; ot.zero_grad()
        nn.functional.mse_loss(mf(Xft[ix]), Yft[ix]).backward(); ot.step()
mf.eval()
with torch.no_grad(): ft = nn.functional.mse_loss(mf(Xfv), Yfv).item()
print(f'FT 30ep: {ft:.2e}')

ms = WFM(in_ch=2).to(WFM_DEV); os_ = torch.optim.Adam(ms.parameters(), lr=1e-3)
for ep in range(100):
    ms.train(); p = torch.randperm(nft, device=WFM_DEV)
    for i in range(0, nft, 8):
        ix = p[i:i+8]; os_.zero_grad()
        nn.functional.mse_loss(ms(Xft[ix]), Yft[ix]).backward(); os_.step()
ms.eval()
with torch.no_grad(): s_ = nn.functional.mse_loss(ms(Xfv), Yfv).item()
print(f'Scratch 100ep: {s_:.2e}')
print(f'\nRESULTS ({len(train)} envs -> {HOLDOUT}):')
print(f'  Zero: {z:.2e}  FT: {ft:.2e}  Scratch: {s_:.2e}  FT/Scratch: {s_/ft:.1f}x {"WIN!" if ft<s_ else "LOSS"}')
