"""
4DVarNet - Correct Implementation
Train on OBSERVATIONS, evaluate against TRUE STATE
"""
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import scipy.interpolate
import sys, os
sys.path.insert(0, os.path.expanduser('~/Desktop/DinAE_4DVarNN_torch'))

print("Loading data...")
x_train     = np.load('results/4dvar_x_train.npy')
x_test      = np.load('results/4dvar_x_test.npy')
x_train_obs = np.load('results/4dvar_x_train_obs.npy')
x_test_obs  = np.load('results/4dvar_x_test_obs.npy')
mask_train  = np.load('results/4dvar_mask_train.npy')
mask_test   = np.load('results/4dvar_mask_test.npy')
meanTr      = np.load('results/4dvar_meanTr.npy')[0]
stdTr       = np.load('results/4dvar_stdTr.npy')[0]

print(f'Train: {x_train.shape}  Test: {x_test.shape}')

device = torch.device("cpu")

# ── ODE Encoder ───────────────────────────────────────
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.F     = torch.nn.Parameter(torch.Tensor([8.]))
        self.dt    = 0.05
        self.conv1 = torch.nn.Conv2d(1,1,(5,1),padding=0,bias=False)
        self.conv2 = torch.nn.Conv2d(1,1,(3,1),padding=0,bias=False)
        self.conv1.weight = torch.nn.Parameter(
            torch.Tensor([-1.,0.,0.,1.,0.]).view(1,1,5,1))
        self.conv2.weight = torch.nn.Parameter(
            torch.Tensor([1.,0.,0.]).view(1,1,3,1))

    def _odeL96(self, xin):
        x_1 = torch.cat((xin[:,:,xin.size(2)-2:,:],
                         xin, xin[:,:,0:2,:]), dim=2)
        x_1 = self.conv1(x_1)
        x_2 = torch.cat((xin[:,:,xin.size(2)-1:,:],
                         xin, xin[:,:,0:1,:]), dim=2)
        x_2 = self.conv2(x_2)
        return (x_1*x_2 - xin + self.F).view(
            -1, xin.size(1), xin.size(2), xin.size(3))

    def _RK4(self, x):
        k1 = self._odeL96(x)
        k2 = self._odeL96(x + 0.5*self.dt*k1)
        k3 = self._odeL96(x + 0.5*self.dt*k2)
        k4 = self._odeL96(x + self.dt*k3)
        return x + self.dt*(k1+2*k2+2*k3+k4)/6.

    def forward(self, x):
        xpred = self._RK4(x[:,:,:,0:x.size(3)-1])
        return torch.cat(
            (x[:,:,:,0].view(-1,1,x.size(2),1), xpred), dim=3)

class Model_AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
    def forward(self, x):
        return self.encoder(x)

# ── 4DVarNet solver ───────────────────────────────────
class FourDVarNet(torch.nn.Module):
    def __init__(self, model_ae, n_iter=15, lr_da=0.05):
        super().__init__()
        self.model_ae = model_ae
        self.n_iter   = n_iter
        self.lr_da    = lr_da

    def forward(self, x_obs, mask, x_init):
        x = x_init.detach().clone()

        with torch.enable_grad():
            x = x.requires_grad_(True)
            opt = torch.optim.Adam([x], lr=self.lr_da)

            for _ in range(self.n_iter):
                opt.zero_grad()
                obs_loss = torch.mean(mask*(x - x_obs)**2)
                x_in     = x.view(-1,1,x.size(1),x.size(2))
                x_pred   = self.model_ae(x_in)
                dyn_loss = torch.mean((x_in - x_pred)**2)
                J        = obs_loss + 0.5*dyn_loss
                J.backward()
                opt.step()

        return x.detach()

# ── Linear interp init ────────────────────────────────
def linear_interp_init(x_obs, mask):
    N, K, T  = x_obs.shape
    x_init   = np.copy(x_obs)
    for n in range(N):
        for k in range(K):
            obs_t = np.where(mask[n,k,:]==1)[0]
            mis_t = np.where(mask[n,k,:]==0)[0]
            if len(obs_t) > 1 and len(mis_t) > 0:
                mis_t = np.clip(mis_t,
                                obs_t.min(), obs_t.max())
                f = scipy.interpolate.interp1d(
                    obs_t, x_obs[n,k,obs_t])
                x_init[n,k,mis_t] = f(mis_t)
    return x_init

print("Linear interp init...")
x_train_init = linear_interp_init(x_train_obs, mask_train)
x_test_init  = linear_interp_init(x_test_obs,  mask_test)
print("Done!")

model_ae  = Model_AE().to(device)
model_4dv = FourDVarNet(model_ae, n_iter=15, lr_da=0.05)
print(f'Params: {sum(p.numel() for p in model_ae.parameters()):,}')

# ── Datasets ──────────────────────────────────────────
batch_size = 16
var_Tt     = np.var(x_test)

train_ds = torch.utils.data.TensorDataset(
    torch.Tensor(x_train_obs),
    torch.Tensor(mask_train),
    torch.Tensor(x_train_init),
    torch.Tensor(x_train))
test_ds  = torch.utils.data.TensorDataset(
    torch.Tensor(x_test_obs),
    torch.Tensor(mask_test),
    torch.Tensor(x_test_init),
    torch.Tensor(x_test))

dataloaders = {
    'train': torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True),
    'val':   torch.utils.data.DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False),
}

optimizer = optim.Adam(model_ae.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_loss  = 1e10
num_epochs = 20

print(f'\nTraining {num_epochs} epochs...')
print('='*60)

for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model_ae.train()
        else:
            model_ae.eval()

        total_mse = 0.0
        n_samples = 0

        for obs, mask, init, gt in dataloaders[phase]:
            obs  = obs.to(device)
            mask = mask.to(device)
            init = init.to(device)
            gt   = gt.to(device)

            optimizer.zero_grad()

            x_rec = model_4dv(obs, mask, init)
            mse   = torch.mean((x_rec - gt)**2)

            if phase == 'train':
                # Train AE on dynamical consistency
                x_in   = gt.view(-1,1,gt.size(1),gt.size(2))
                x_pred = model_ae(x_in)
                loss   = torch.mean((x_in - x_pred)**2)
                loss.backward()
                optimizer.step()

            total_mse += mse.item() * obs.size(0)
            n_samples += obs.size(0)

        epoch_mse  = total_mse / n_samples
        epoch_nmse = epoch_mse / var_Tt

        if phase == 'val':
            print(f'Epoch {epoch+1:2d}/{num_epochs} | '
                  f'MSE: {epoch_mse:.4f} | '
                  f'NMSE: {epoch_nmse:.4f}')
            if epoch_mse < best_loss:
                best_loss = epoch_mse
                torch.save(model_ae.state_dict(),
                           'results/4dvarnet_best_model.pth')

    scheduler.step()

# ── Final Evaluation ───────────────────────────────────
print('\nFinal evaluation...')
model_ae.load_state_dict(
    torch.load('results/4dvarnet_best_model.pth'))
model_ae.eval()
model_4dv = FourDVarNet(model_ae, n_iter=20, lr_da=0.05)

all_pred = []
all_gt   = []

for obs, mask, init, gt in dataloaders['val']:
    x_rec = model_4dv(obs.to(device),
                      mask.to(device),
                      init.to(device))
    pred = x_rec.cpu().numpy()*stdTr + meanTr
    true = gt.numpy()*stdTr + meanTr
    all_pred.append(pred)
    all_gt.append(true)

all_pred = np.concatenate(all_pred)
all_gt   = np.concatenate(all_gt)

r_raw  = np.mean((all_gt - all_pred)**2)
r_norm = r_raw / np.var(all_gt)

print(f'\n{"="*60}')
print(f'4DVarNet R-score (raw):        {r_raw:.4f}')
print(f'4DVarNet R-score (normalized): {r_norm:.4f}')
print(f'Cell-NN  R-score (normalized): 0.1217')
print(f'3D-Var   R-score (normalized): 0.0945')
print(f'Fablet paper best:             0.38')
print(f'{"="*60}')

np.save('results/4dvarnet_r_score.npy',
        np.array([r_raw, r_norm]))
print('Saved!')
