# Notebook'ların ortak kullandığı kod: veri hazırlama, Dataset, model, kayıp, metrik ve eğitim döngüsü.
# İlk hali 02_baseline içinde yazıldı, deneyler büyüyünce buraya taşındı.
import os, glob, random, zipfile, time, json, urllib.request
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# ---------- Yollar ve sabitler ----------
ROOT     = '/content/drive/MyDrive/multispectral-weed-segmentation'
DATA_DIR = f'{ROOT}/data'
CKPT_DIR = f'{ROOT}/checkpoints'
RES_DIR  = f'{ROOT}/results'
URL      = 'https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip'
ZIP_PATH = f'{DATA_DIR}/weedsgalore-dataset.zip'
LOCAL    = '/content/data'

DATES    = ['2023-05-25', '2023-05-30', '2023-06-06', '2023-06-15']
BANDS    = ['B', 'G', 'R', 'RE', 'NIR']
SINIFLAR = ['arka plan', 'mısır', 'amaranth', 'darıcan', 'quickweed', 'diğer ot']
SIN3     = ['arka plan', 'mısır', 'ot']
IGNORE   = 255
BANT_IDX = {b: i for i, b in enumerate(BANDS)}
KANAL_SECIMI = {'rgb':      ['R', 'G', 'B'],
                'rgb_exg':  ['R', 'G', 'B', 'ExG'],
                'msi':      ['R', 'G', 'B', 'RE', 'NIR'],
                'msi_ndvi': ['R', 'G', 'B', 'RE', 'NIR', 'NDVI']}
SEED   = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# hazirla() doldurur
DATASET, NORM_ORT, NORM_STD = None, None, None
splits, HAM, MASKE, IND_ORT, IND_STD = {}, {}, {}, {}, {}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ham_oku(ad):
    return np.stack([np.array(Image.open(f'{DATASET}/{ad[:10]}/images/{ad}_{b}.png')) for b in BANDS])

def maske_oku(ad):
    return np.array(Image.open(f'{DATASET}/{ad[:10]}/semantics/{ad}.png'))

def indeksler(x):
    B, G, R, RE, NIR = x
    return {'NDVI': (NIR - R) / (NIR + R + 1e-6),
            'ExG':  (2 * G - R - B) / (B + G + R + 1e-6)}

def hazirla():
    """Veriyi indirir/açar, görüntüleri belleğe alır, normalizasyon istatistiklerini yükler ya da hesaplar."""
    global DATASET, NORM_ORT, NORM_STD
    for d in [DATA_DIR, CKPT_DIR, RES_DIR]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(LOCAL):
        if not os.path.exists(ZIP_PATH):
            print('Veri indiriliyor (~337 MB)...')
            urllib.request.urlretrieve(URL, ZIP_PATH + '.part')
            os.rename(ZIP_PATH + '.part', ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(LOCAL)
    DATASET = os.path.dirname(glob.glob(f'{LOCAL}/**/2023-05-25', recursive=True)[0])

    for s in ['train', 'val', 'test']:
        with open(f'{DATASET}/splits/{s}.txt') as f:
            splits[s] = [satir.strip() for satir in f if satir.strip()]
    for ad in splits['train'] + splits['val'] + splits['test']:
        HAM[ad], MASKE[ad] = ham_oku(ad), maske_oku(ad)

    # Bant istatistikleri yalnız train'den (01_kesif ile aynı hesap)
    if os.path.exists(f'{RES_DIR}/norm_ort.npy'):
        NORM_ORT = np.load(f'{RES_DIR}/norm_ort.npy')
        NORM_STD = np.load(f'{RES_DIR}/norm_std.npy')
    else:
        print('Normalizasyon dosyası yok, train setinden hesaplanıyor...')
        toplam, kare, piksel = np.zeros(5), np.zeros(5), 0
        for ad in splits['train']:
            x = HAM[ad].astype(np.float64) / 65535
            toplam += x.sum(axis=(1, 2)); kare += (x ** 2).sum(axis=(1, 2)); piksel += x[0].size
        NORM_ORT = (toplam / piksel).astype(np.float32)
        NORM_STD = np.sqrt(kare / piksel - (toplam / piksel) ** 2).astype(np.float32)
        np.save(f'{RES_DIR}/norm_ort.npy', NORM_ORT)
        np.save(f'{RES_DIR}/norm_std.npy', NORM_STD)

    # NDVI ve ExG için de train ortalaması/std'si
    ort, kare = {'NDVI': [], 'ExG': []}, {'NDVI': [], 'ExG': []}
    for ad in splits['train']:
        ind = indeksler(HAM[ad].astype(np.float32) / 65535)
        for k in ind:
            ort[k].append(ind[k].mean()); kare[k].append((ind[k] ** 2).mean())
    for k in ort:
        IND_ORT[k] = float(np.mean(ort[k]))
        IND_STD[k] = float(np.sqrt(np.mean(kare[k]) - IND_ORT[k] ** 2))
    print(f'Hazır: {len(HAM)} görüntü | split {[len(v) for v in splits.values()]} | cihaz {DEVICE}')

# ---------- Dataset ----------
class OtDataset(Dataset):
    def __init__(self, adlar, kanallar='rgb', n_sinif=3, egitim=False, kirp=512):
        self.adlar, self.kanallar = adlar, KANAL_SECIMI[kanallar]
        self.n_sinif, self.egitim, self.kirp = n_sinif, egitim, kirp

    def __len__(self):
        return len(self.adlar)

    def __getitem__(self, i):
        ad = self.adlar[i]
        x = HAM[ad].astype(np.float32) / 65535
        y = MASKE[ad].astype(np.int64)
        if self.n_sinif == 3:
            y = np.where(y > 1, 2, y)          # bütün ot türleri tek sınıf
        if self.egitim:
            k = self.kirp
            r, c = np.random.randint(0, 600 - k + 1, size=2)
            x, y = x[:, r:r + k, c:c + k], y[r:r + k, c:c + k]
            if np.random.rand() < 0.5: x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5: x, y = x[:, ::-1, :], y[::-1, :]
            n = np.random.randint(4)
            x, y = np.rot90(x, n, axes=(1, 2)), np.rot90(y, n)
            x = x * np.random.uniform(0.9, 1.1)   # tüm bantlara aynı çarpan, bant oranları bozulmasın
            x = np.clip(x + np.random.normal(0, 0.01, x.shape).astype(np.float32), 0, 1)
        ind = indeksler(x) if any(k in IND_ORT for k in self.kanallar) else {}
        kanal = []
        for k in self.kanallar:
            if k in BANT_IDX:
                j = BANT_IDX[k]
                kanal.append((x[j] - NORM_ORT[j]) / NORM_STD[j])
            else:
                kanal.append((ind[k] - IND_ORT[k]) / IND_STD[k])
        x = np.stack(kanal).astype(np.float32)
        if not self.egitim:
            # 600 -> 608: giriş 32'ye bölünmeli, dolgu pikselleri metriğe girmiyor
            x = np.pad(x, ((0, 0), (0, 8), (0, 8)), mode='reflect')
            y = np.pad(y, ((0, 8), (0, 8)), constant_values=IGNORE)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))

def worker_init(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)

# ---------- Model, kayıp, metrik ----------
def model_olustur(n_kanal, n_sinif):
    return smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet',
                             in_channels=n_kanal, classes=n_sinif).to(DEVICE)

def kayip_olustur(tur='ce', agirlik=None):
    """'ce': çapraz entropi (agirlik verilirse ağırlıklı), 'focal_dice': focal (gamma=2) + Dice."""
    if tur == 'focal_dice':
        focal = smp.losses.FocalLoss('multiclass', gamma=2.0, ignore_index=IGNORE)
        dice = smp.losses.DiceLoss('multiclass', from_logits=True, ignore_index=IGNORE)
        return lambda cikti, hedef: focal(cikti.float(), hedef) + dice(cikti.float(), hedef)
    if tur != 'ce':
        raise ValueError(f'Bilinmeyen kayıp: {tur}')
    w = None if agirlik is None else torch.as_tensor(agirlik, dtype=torch.float32, device=DEVICE)
    return nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)

def sinif_agirliklari(n_sinif=6):
    """Train maskelerinden karekök medyan-frekans ağırlıkları: sqrt(median(f) / f)."""
    say = np.zeros(n_sinif)
    for ad in splits['train']:
        y = MASKE[ad] if n_sinif == 6 else np.minimum(MASKE[ad], 2)
        say += np.bincount(y.ravel(), minlength=n_sinif)[:n_sinif]
    f = say / say.sum()
    return np.sqrt(np.median(f) / f)

def karisiklik(pred, y, n):
    gecerli = y != IGNORE
    return torch.bincount(y[gecerli] * n + pred[gecerli], minlength=n * n).reshape(n, n)

def metrikler(cm):
    cm = cm.double()
    tp, gercek, tahmin = cm.diag(), cm.sum(1), cm.sum(0)
    iou       = tp / (gercek + tahmin - tp).clamp(min=1)
    recall    = tp / gercek.clamp(min=1)
    precision = tp / tahmin.clamp(min=1)
    f1        = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    return {'miou': iou.mean().item(), 'iou': iou.cpu().numpy(),
            'recall': recall.cpu().numpy(), 'precision': precision.cpu().numpy(), 'f1': f1.cpu().numpy(),
            'ot_recall': (cm[2:, 2:].sum() / cm[2:, :].sum()).item(),   # tüm ot türleri birlikte
            'misir_fp':  (cm[1, 2:].sum() / cm[1, :].sum()).item()}    # ot sanılan mısır oranı

@torch.no_grad()
def degerlendir(model, ds, n_sinif, kayip_fn=None, bs=4):
    model.eval()
    yuk = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
    cm_top   = torch.zeros(n_sinif, n_sinif, dtype=torch.long, device=DEVICE)
    cm_tarih = {d: torch.zeros_like(cm_top) for d in DATES}
    kayip_top, i = 0.0, 0
    for x, y in yuk:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast('cuda', dtype=torch.float16):
            cikti = model(x)
            if kayip_fn is not None:
                kayip_top += kayip_fn(cikti, y).item() * len(x)
        pred = cikti.argmax(1)
        for b in range(len(x)):
            cm = karisiklik(pred[b], y[b], n_sinif)
            cm_top += cm
            cm_tarih[ds.adlar[i][:10]] += cm
            i += 1
    return cm_top, cm_tarih, kayip_top / len(ds)

# ---------- Uçuş yüksekliği benzetimi (E3) ----------
def gsd_boz(ham, f):
    """f x f blokların ortalaması (büyük sensör pikseli), sonra tekrar 600x600'e bilineer büyütme."""
    x = torch.from_numpy(ham.astype(np.float32))[None]
    x = F.avg_pool2d(x, kernel_size=f)
    x = F.interpolate(x, size=(600, 600), mode='bilinear', align_corners=False)
    return x[0].round().clamp(0, 65535).numpy().astype(np.uint16)

# ---------- Eğitim ----------
def egit(deney, kanallar='rgb', n_sinif=3, epoch=100, bs=8, lr=1e-3, kayip='ce', agirlik=None, seed=SEED):
    set_seed(seed)
    ds_tr = OtDataset(splits['train'], kanallar, n_sinif, egitim=True)
    ds_va = OtDataset(splits['val'],   kanallar, n_sinif, egitim=False)
    yuk = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=2, drop_last=True,
                     worker_init_fn=worker_init, generator=torch.Generator().manual_seed(seed))
    model    = model_olustur(len(KANAL_SECIMI[kanallar]), n_sinif)
    kayip_fn = kayip_olustur(kayip, agirlik)
    opt      = torch.optim.Adam(model.parameters(), lr=lr)
    toplam   = epoch * len(yuk)
    sched    = torch.optim.lr_scheduler.LambdaLR(opt, lambda a: (1 - a / toplam) ** 0.9)   # poly
    scaler   = torch.amp.GradScaler()
    gecmis, en_iyi, en_iyi_ep, t0 = [], -1.0, 0, time.time()
    for ep in range(1, epoch + 1):
        model.train()
        tr_kayip = 0.0
        for x, y in yuk:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                k = kayip_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(k).backward(); scaler.step(opt); scaler.update()
            sched.step()
            tr_kayip += k.item()
        cm, _, va_kayip = degerlendir(model, ds_va, n_sinif, kayip_fn)
        m = metrikler(cm)
        gecmis.append({'epoch': ep, 'train_kayip': tr_kayip / len(yuk), 'val_kayip': va_kayip,
                       'val_miou': m['miou'], 'ot_recall': m['ot_recall'], 'misir_fp': m['misir_fp'],
                       'lr': sched.get_last_lr()[0], **{f'iou_{c}': v for c, v in enumerate(m['iou'])}})
        if m['miou'] > en_iyi:
            en_iyi, en_iyi_ep = m['miou'], ep
            torch.save(model.state_dict(), f'{CKPT_DIR}/{deney}.pt')
        if ep == 1 or ep % 10 == 0:
            print(f'[{deney}] ep {ep:3d} | train {gecmis[-1]["train_kayip"]:.3f} | val {va_kayip:.3f} '
                  f'| val mIoU {m["miou"]:.3f} | en iyi {en_iyi:.3f} (ep {en_iyi_ep}) | {(time.time() - t0) / 60:.1f} dk')
    df = pd.DataFrame(gecmis)
    df.to_csv(f'{RES_DIR}/{deney}_gecmis.csv', index=False)
    ozet = {'deney': deney, 'kanallar': kanallar, 'n_sinif': n_sinif, 'kayip': kayip, 'agirlik': agirlik,
            'seed': seed, 'epoch': epoch, 'bs': bs, 'lr': lr, 'en_iyi_epoch': en_iyi_ep,
            'val_miou': en_iyi, 'sure_dk': round((time.time() - t0) / 60, 1)}
    with open(f'{RES_DIR}/{deney}_ozet.json', 'w') as f:
        json.dump(ozet, f, indent=2)
    return df, ozet
