#!/usr/bin/env python3
"""
Tier 1 Gap Filling:
A) R=4,5,8 MLP at high power (50K test, patience=50, 500 epochs)
B) Diffusion full training (500 epochs, 50K test)
C) GAN full training (500 epochs, 50K test)
"""
import torch, torch.nn as nn, numpy as np, hashlib, struct, json, time, math, signal, sys
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

INTERRUPTED = False
def sighandler(s,f):
    global INTERRUPTED; INTERRUPTED = True
signal.signal(signal.SIGINT, sighandler)

sandbox = Path("/mnt/d/sha256-ml-redux")

# === Shared utilities ===

def count_lz_btc(h):
    rev = h[::-1]
    c = 0
    for b in rev:
        if b == 0: c += 8
        else: return c + (8 - b.bit_length())
    return c

def update_status(msg):
    with open(sandbox / "status.json", "w") as f:
        json.dump({"phase": "tier1_gaps", "message": msg, "timestamp": time.time()}, f)

# === Part A: R=4,5,8 high-power MLP ===

def part_a():
    print("="*60)
    print("PART A: High-power MLP at R=4, 5, 8")
    print("="*60)

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(992, 1024), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(1024, 32))
        def forward(self, x): return self.net(x)

    for R in [4, 5, 8]:
        if INTERRUPTED: break
        update_status(f"Part A: R={R}")
        print(f"\n--- R={R} ---")

        f = sandbox / "data" / f"dataset_reduced_r{R:02d}.npy"
        if not f.exists():
            print(f"  Dataset not found: {f}")
            continue

        data = np.load(f)
        n = len(data)
        X = torch.FloatTensor(data[:, :992])
        Y = torch.FloatTensor(data[:, 992:])

        # 80/10/10 split
        tr = int(0.8*n); va = int(0.9*n)
        train_dl = DataLoader(TensorDataset(X[:tr], Y[:tr]), batch_size=128, shuffle=True)
        val_dl = DataLoader(TensorDataset(X[tr:va], Y[tr:va]), batch_size=256)
        test_X, test_Y = X[va:].to(device), Y[va:].to(device)
        print(f"  Train={tr}, Val={va-tr}, Test={n-va}")

        model = MLP().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler("cuda")

        best_val = float('inf'); patience = 0; best_epoch = 0

        for epoch in range(500):
            if INTERRUPTED: break
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                with torch.amp.autocast(device_type="cuda"):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()

            model.eval()
            vl = 0; nb = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    with torch.amp.autocast(device_type="cuda"):
                        vl += criterion(model(xb), yb).item()
                    nb += 1
            vl /= nb

            if vl < best_val:
                best_val = vl; patience = 0; best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1

            if epoch % 50 == 0:
                print(f"  Epoch {epoch}: val_loss={vl:.6f}, patience={patience}")

            if epoch >= 100 and patience >= 50:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Evaluate
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = model(test_X)
            preds = (logits > 0.0).float()
            per_bit = (preds == test_Y).float().mean(dim=0).cpu().numpy()
            overall = per_bit.mean()

        # P-values
        n_test = len(test_X)
        sig_bits = 0
        for bit in range(32):
            z = (per_bit[bit] - 0.5) / math.sqrt(0.25 / n_test)
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
            if p < 0.01/32 and per_bit[bit] > 0.505:
                sig_bits += 1

        print(f"  R={R}: acc={overall:.4f}, sig_bits={sig_bits}/32, best_epoch={best_epoch}")
        results[R] = {
            "accuracy": float(overall),
            "significant_bits": sig_bits,
            "per_bit": [float(x) for x in per_bit],
            "n_test": n_test,
            "best_epoch": best_epoch
        }

    return results


# === Part B: Diffusion full training ===

def part_b():
    print("\n" + "="*60)
    print("PART B: Diffusion full training (500 epochs)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(sandbox / "data" / "dataset_reduced_r64.npy")
    n = len(data)
    headers = torch.FloatTensor(data[:, :992])
    nonces = torch.FloatTensor(data[:, 992:])

    tr = int(0.8*n)
    train_dl = DataLoader(TensorDataset(headers[:tr], nonces[:tr]), batch_size=256, shuffle=True)
    test_headers = headers[tr:].numpy()
    test_stubs = [np.packbits(data[i, :608].astype(np.uint8)).tobytes() for i in range(tr, n)]

    # Best config from Phase 3.2: depth=2, width=1024, lr=1e-4
    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(992 + 32 + 1, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Linear(1024, 32))
        def forward(self, noisy_nonce, header, t):
            return self.net(torch.cat([noisy_nonce, header, t.unsqueeze(-1)], dim=-1))

    model = Denoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    T = 100
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, 0)

    print("Training diffusion model...")
    for epoch in range(500):
        if INTERRUPTED: break
        total_loss = 0; nb = 0
        model.train()
        for hdr_b, non_b in train_dl:
            hdr_b, non_b = hdr_b.to(device), non_b.to(device)
            t = torch.randint(0, T, (len(hdr_b),), device=device)
            noise = torch.randn_like(non_b)
            ab = alpha_bars[t].unsqueeze(-1)
            noisy = torch.sqrt(ab) * non_b + torch.sqrt(1-ab) * noise

            opt.zero_grad()
            pred_noise = model(noisy, hdr_b, t.float()/T)
            loss = nn.MSELoss()(pred_noise, noise)
            loss.backward()
            opt.step()
            total_loss += loss.item(); nb += 1

        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: loss={total_loss/nb:.6f}")

    # Generate and validate
    print("Generating 50000 nonces...")
    model.eval()
    valid = 0; total = min(50000, len(test_stubs))

    with torch.no_grad():
        for i in range(total):
            if i % 10000 == 0: print(f"  {i}/{total}")
            hdr = torch.FloatTensor(test_headers[i]).unsqueeze(0).to(device)
            x = torch.randn(1, 32).to(device)
            for t_idx in reversed(range(T)):
                t_tensor = torch.FloatTensor([t_idx/T]).to(device)
                pred = model(x, hdr, t_tensor)
                ab = alpha_bars[t_idx]
                ab_prev = alpha_bars[t_idx-1] if t_idx > 0 else torch.tensor(1.0)
                x = (x - (1-alphas[t_idx])/torch.sqrt(1-ab) * pred) / torch.sqrt(alphas[t_idx])
                if t_idx > 0:
                    x += torch.sqrt(betas[t_idx]) * torch.randn_like(x)

            nonce_bits = (x.squeeze() > 0).cpu().numpy().astype(np.uint8)
            nonce_bytes = np.packbits(nonce_bits).tobytes()
            header_80 = test_stubs[i] + nonce_bytes
            h = hashlib.sha256(hashlib.sha256(header_80).digest()).digest()
            if count_lz_btc(h) >= 1:
                valid += 1

    rate = valid / total
    print(f"  Validity: {valid}/{total} = {rate:.4f} (baseline 0.50)")
    return {"validity_rate": rate, "valid": valid, "total": total, "epochs": 500}


# === Part C: GAN full training ===

def part_c():
    print("\n" + "="*60)
    print("PART C: GAN full training (500 epochs)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(sandbox / "data" / "dataset_reduced_r64.npy")
    n = len(data)
    headers = torch.FloatTensor(data[:, :992])
    nonces = torch.FloatTensor(data[:, 992:])

    tr = int(0.8*n)
    train_dl = DataLoader(TensorDataset(headers[:tr], nonces[:tr]), batch_size=256, shuffle=True)
    test_headers = headers[tr:].numpy()
    test_stubs = [np.packbits(data[i, :608].astype(np.uint8)).tobytes() for i in range(tr, n)]

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(992 + 32, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 32))
        def forward(self, header, noise):
            return self.net(torch.cat([header, noise], dim=-1))

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(992 + 32, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 1))
        def forward(self, header, nonce):
            return self.net(torch.cat([header, nonce], dim=-1))

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.0, 0.9))

    print("Training WGAN-GP...")
    for epoch in range(500):
        if INTERRUPTED: break
        for hdr_b, non_b in train_dl:
            hdr_b, non_b = hdr_b.to(device), non_b.to(device)
            bs = len(hdr_b)

            # Discriminator
            for _ in range(5):
                noise = torch.randn(bs, 32, device=device)
                fake = gen(hdr_b, noise).detach()
                d_real = disc(hdr_b, non_b).mean()
                d_fake = disc(hdr_b, fake).mean()

                # Gradient penalty
                eps = torch.rand(bs, 1, device=device)
                interp = eps * non_b + (1-eps) * fake
                interp.requires_grad_(True)
                d_interp = disc(hdr_b, interp)
                grad = torch.autograd.grad(d_interp.sum(), interp, create_graph=True)[0]
                gp = ((grad.norm(2, dim=1) - 1)**2).mean()

                loss_d = d_fake - d_real + 10 * gp
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # Generator
            noise = torch.randn(bs, 32, device=device)
            fake = gen(hdr_b, noise)
            loss_g = -disc(hdr_b, fake).mean()
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: g_loss={loss_g.item():.4f}, d_loss={loss_d.item():.4f}")

    # Generate and validate
    print("Generating 50000 nonces...")
    gen.eval()
    valid = 0; total = min(50000, len(test_stubs))

    with torch.no_grad():
        for i in range(total):
            if i % 10000 == 0: print(f"  {i}/{total}")
            hdr = torch.FloatTensor(test_headers[i]).unsqueeze(0).to(device)
            noise = torch.randn(1, 32, device=device)
            fake = gen(hdr, noise)
            nonce_bits = (fake.squeeze() > 0).cpu().numpy().astype(np.uint8)
            nonce_bytes = np.packbits(nonce_bits).tobytes()
            header_80 = test_stubs[i] + nonce_bytes
            h = hashlib.sha256(hashlib.sha256(header_80).digest()).digest()
            if count_lz_btc(h) >= 1:
                valid += 1

    rate = valid / total
    print(f"  Validity: {valid}/{total} = {rate:.4f} (baseline 0.50)")
    return {"validity_rate": rate, "valid": valid, "total": total, "epochs": 500}


# === Main ===

if __name__ == "__main__":
    print("Tier 1 Gap Filling")
    print("="*60)
    all_results = {}

    all_results["part_a_mlp_r4_5_8"] = part_a()

    if not INTERRUPTED:
        all_results["part_b_diffusion"] = part_b()

    if not INTERRUPTED:
        all_results["part_c_gan"] = part_c()

    with open(sandbox / "data" / "tier1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("TIER 1 COMPLETE")
    print("="*60)

    # Summary
    if "part_a_mlp_r4_5_8" in all_results:
        for R, res in all_results["part_a_mlp_r4_5_8"].items():
            print(f"  R={R}: acc={res['accuracy']:.4f}, sig_bits={res['significant_bits']}")
    if "part_b_diffusion" in all_results:
        print(f"  Diffusion: validity={all_results['part_b_diffusion']['validity_rate']:.4f}")
    if "part_c_gan" in all_results:
        print(f"  GAN: validity={all_results['part_c_gan']['validity_rate']:.4f}")
