"""5-Atari WFM — CPU for AE, MPS for WFM, batch=4, safe padding."""
import ale_py; import gymnasium; gymnasium.register_envs(ale_py)
import numpy as np, torch, torch.nn as nn, time, os
from wmca.modules.hybrid import CML2DMultiR

ACE = torch.device('cpu'); WFM_DEV = torch.device('mps')
GAMES = [('Pong',6),('Breakout',4),('SpaceInvaders',6),('Freeway',3),('Asterix',9)]
HOLDOUT = 'Breakout'

def play_game(name, na, eps=150):
    env = gymnasium.make(f'ALE/{name}-v5')
    rng = np.random.default_rng(42)
    fs, nfs, acts = [], [], []
    for ep in range(eps):
        obs, _ = env.reset(seed=ep)
        for _ in range(50):
            a = rng.integers(0, na)
            nobs, _, t, tr, _ = env.step(a)
            g = np.dot(obs[...,:3],[0.299,0.587,0.114]).astype(np.float32)/255.0
            ng = np.dot(nobs[...,:3],[0.299,0.587,0.114]).astype(np.float32)/255.0
            fs.append(g); nfs.append(ng); acts.append(a); obs = nobs
            if t or tr: break
        if ep%40==0: print(f'  {name}: {ep}/{eps}')
    env.close(); del env
    return np.stack(fs)[:,np.newaxis], np.stack(nfs)[:,np.newaxis], np.array(acts, dtype=np.int64)

class GAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1,16,3,1,1),nn.ReLU(),nn.Conv2d(16,1,3,1,1),nn.Sigmoid())
        self.dec = nn.Sequential(nn.Conv2d(1,16,3,1,1),nn.ReLU(),nn.Conv2d(16,1,3,1,1),nn.Sigmoid())
    def encode(self,x):
        return self.enc(x)
    def forward(self,x):
        return self.dec(self.enc(x))

def make_af(acts, na, H, W):
    v = (acts.astype(np.float32)+1.0)/na
    return np.tile(v[:,None,None,None],(1,1,H,W))

def pad32(X):
    h,w = X.shape[2], X.shape[3]
    ph = max(0, (32-h)//2); pw = max(0, (32-w)//2)
    return np.pad(X,((0,0),(0,0),(ph,max(0,32-h-ph)),(pw,max(0,32-w-pw))))

class WFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cml = CML2DMultiR(in_channels=2,K=32,steps=15,seed=42,gate_mode='uniform')
        self.nca = nn.Sequential(nn.Conv2d(4,64,3,1,1),nn.ReLU(),nn.Conv2d(64,64,3,1,1),nn.ReLU(),nn.Conv2d(64,1,1))
    def forward(self,x):
        with torch.no_grad(): co=self.cml(x)
        return co[:,:1]+self.nca(torch.cat([x,co],dim=1))

# STEP 1: Generate + encode
data = {}
for name, na in GAMES:
    print(f'\n--- {name} ---')
    f, nf, acts = play_game(name, na)
    nt = int(len(f)*0.8)
    ae = GAE().to(ACE); opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    Xa = torch.from_numpy(f[:nt]).float().to(ACE)
    for _ in range(10):
        p = torch.randperm(len(Xa))
        for i in range(0, len(p), 4):
            ix = p[i:i+4]; opt.zero_grad()
            nn.functional.mse_loss(ae(Xa[ix]), Xa[ix]).backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        lats = ae.encode(torch.from_numpy(f).float().to(ACE)).numpy()
        nlats = ae.encode(torch.from_numpy(nf).float().to(ACE)).numpy()
    mse = nn.functional.mse_loss(ae(torch.from_numpy(f[:64]).float().to(ACE)), torch.from_numpy(f[:64]).float()).item()
    H,W = lats.shape[2:]
    X = pad32(np.concatenate([lats, make_af(acts, na, H, W)], axis=1))
    Y = pad32(nlats)
    data[name] = {'X': X, 'Y': Y}
    del ae, Xa, f, nf, lats, nlats
    print(f'  {len(acts)} samples, PSNR={10*np.log10(1/mse):.1f}dB, {H}x{W}')

# STEP 2: WFM pre-training
train = [g for g,_ in GAMES if g != HOLDOUT]
print(f'\n=== WFM: {train} -> {HOLDOUT} (batch=4, {100} epochs) ===')
model = WFM().to(WFM_DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()
for ep in range(100):
    model.train()
    for g in train:
        Xt, Yt = data[g]['X'], data[g]['Y']
        n = min(len(Xt), 3000)
        xt = torch.from_numpy(Xt[:n]).float().to(WFM_DEV); yt = torch.from_numpy(Yt[:n]).float().to(WFM_DEV)
        p = torch.randperm(n, device=WFM_DEV)
        for i in range(0, n, 4):
            ix = p[i:i+4]; opt.zero_grad()
            nn.functional.mse_loss(model(xt[ix]), yt[ix]).backward(); opt.step()
    if ep%20==0 or ep==99:
        print(f'  ep {ep+1:3d}/100 [{((time.time()-t0)/60):.0f}m]')

os.makedirs('experiments/results', exist_ok=True)
torch.save(model.state_dict(), 'experiments/results/wfm_5env.pt')

# STEP 3: Transfer to holdout
Xb, Yb = data[HOLDOUT]['X'], data[HOLDOUT]['Y']
nft = int(len(Xb)*0.7)
Xft = torch.from_numpy(Xb[:nft]).float().to(WFM_DEV); Yft = torch.from_numpy(Yb[:nft]).float().to(WFM_DEV)
Xfv = torch.from_numpy(Xb[nft:nft+256]).float().to(WFM_DEV); Yfv = torch.from_numpy(Yb[nft:nft+256]).float().to(WFM_DEV)

model.eval()
with torch.no_grad(): z = nn.functional.mse_loss(model(Xfv), Yfv).item()

mf = WFM().to(WFM_DEV); mf.load_state_dict(torch.load('experiments/results/wfm_5env.pt'))
ot = torch.optim.Adam(mf.parameters(), lr=1e-4)
for ep in range(30):
    mf.train(); p = torch.randperm(nft, device=WFM_DEV)
    for i in range(0, nft, 4):
        ix = p[i:i+4]; ot.zero_grad()
        nn.functional.mse_loss(mf(Xft[ix]), Yft[ix]).backward(); ot.step()
mf.eval()
with torch.no_grad(): ft = nn.functional.mse_loss(mf(Xfv), Yfv).item()

ms = WFM().to(WFM_DEV); os_ = torch.optim.Adam(ms.parameters(), lr=1e-3)
for ep in range(100):
    ms.train(); p = torch.randperm(nft, device=WFM_DEV)
    for i in range(0, nft, 4):
        ix = p[i:i+4]; os_.zero_grad()
        nn.functional.mse_loss(ms(Xft[ix]), Yft[ix]).backward(); os_.step()
ms.eval()
with torch.no_grad(): s_ = nn.functional.mse_loss(ms(Xfv), Yfv).item()

print(f'\n{"="*50}')
print(f'RESULTS ({len(train)} games pre-train, holdout={HOLDOUT}):')
print(f'  Zero-shot: {z:.2e}  FT 30ep: {ft:.2e}  Scratch 100ep: {s_:.2e}')
print(f'  FT vs scratch: {s_/ft:.1f}x {"WIN!" if ft<s_ else "LOSS"}')
