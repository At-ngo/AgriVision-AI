"""
hybrid_ensemble.py
==================
Hybrid Multimodal Ensemble (v6) combining:
  1. Deep Learning: SS-CFNet v5 (ResNet-18 RGB + Pixel-SNV HS + VI MS) with 3x TTA
  2. Chemometric Spectral SVM-RBF (916 physics-informed HS/MS/RGB features)
  3. Chemometric Spectral Regularized Logistic Regression
  4. Automatic Zero-HS Fallback: routes corrupt/all-zero HS samples 100% to Deep Learning (RGB-driven)

Logs full evaluation metrics, Confusion Matrix Heatmap, Model Artifact (.joblib),
and Test Submission CSV directly to Weights & Biases (WandB).
"""

import argparse
import os
import shutil
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from src.models.model import build_multimodal_model
from src.modules.dataset import (
    MultimodalDataset,
    MultimodalTestDataset,
    _read_tiff_chw,
    get_transforms,
    load_split_manifest,
)
from src.modules.reporting import (
    save_classification_report_artifacts,
    save_confusion_matrix_artifacts,
)
from src.modules.utils import load_config, set_seed


def extract_chemometric_features(hs_dir: str, ms_dir: str, rgb_dir: str, file_list: list[str]):
    """
    Extract 916-dim physics-informed chemometric & spatial features from HS, MS, and RGB.
    Returns:
        X: np.ndarray of shape (N, 916)
        valid_hs_flags: np.ndarray of shape (N,) with 1.0 if HS is non-zero else 0.0
    """
    feats = []
    valid_hs_flags = []

    for fname in file_list:
        base = os.path.splitext(fname)[0]
        hs_path = os.path.join(hs_dir, base + ".tif")
        ms_path = os.path.join(ms_dir, base + ".tif")
        rgb_path = os.path.join(rgb_dir, fname)
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(rgb_dir, base + ".png")

        hs = (
            _read_tiff_chw(hs_path, 125).astype(np.float32)
            if os.path.exists(hs_path)
            else np.zeros((125, 32, 32), dtype=np.float32)
        )
        ms = (
            _read_tiff_chw(ms_path, 5).astype(np.float32)
            if os.path.exists(ms_path)
            else np.zeros((5, 64, 64), dtype=np.float32)
        )
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0

        is_valid_hs = float(hs.max() > 0)
        valid_hs_flags.append(is_valid_hs)

        # 1. HS Foreground-masked L2-normalized / SNV spectrum (illumination invariant)
        norm = np.linalg.norm(hs, axis=0, keepdims=True) + 1e-6
        h_norm = hs / norm
        mask = norm[0] > np.percentile(norm[0], 25)
        if mask.sum() < 5:
            mask = np.ones((32, 32), dtype=bool)
        h_fg = h_norm[:, mask]  # (125, N_fg)
        mean_spec = h_fg.mean(axis=1)
        std_spec = h_fg.std(axis=1)

        # Smoothed 1st & 2nd spectral derivatives (Red-Edge slope & inflection point)
        kernel = np.ones(5, dtype=np.float32) / 5.0
        smooth_spec = np.convolve(mean_spec, kernel, mode="valid")
        d1 = np.diff(smooth_spec)
        d2 = np.diff(d1)

        # 25 sub-bands & all pairwise Normalized Difference Spectral Indices (NDSI)
        sub25 = mean_spec.reshape(25, 5).mean(axis=1)
        sub25_std = h_fg.reshape(25, 5, -1).mean(axis=1).std(axis=1)
        ratios = []
        for i in range(25):
            for j in range(i + 1, 25):
                ratios.append((sub25[j] - sub25[i]) / (sub25[j] + sub25[i] + 1e-6))

        # 2. MS normalized quantiles, spatial spot gradients, and Vegetation Indices
        m_norm = ms / (np.linalg.norm(ms, axis=0, keepdims=True) + 1e-6)
        q_ms = np.percentile(m_norm, [10, 25, 50, 75, 90], axis=(1, 2)).flatten()
        dx_ms = np.abs(ms[:, :, 1:] - ms[:, :, :-1]).mean(axis=(1, 2)) / (ms.mean(axis=(1, 2)) + 1e-6)
        dy_ms = np.abs(ms[:, 1:, :] - ms[:, :-1, :]).mean(axis=(1, 2)) / (ms.mean(axis=(1, 2)) + 1e-6)

        eps = 1e-5
        ndvi = (ms[4] - ms[2]) / (ms[4] + ms[2] + eps)
        ndre = (ms[4] - ms[3]) / (ms[4] + ms[3] + eps)
        gndvi = (ms[4] - ms[1]) / (ms[4] + ms[1] + eps)
        q_vi = np.concatenate([
            np.percentile(ndvi, [10, 25, 50, 75, 90]),
            np.percentile(ndre, [10, 25, 50, 75, 90]),
            np.percentile(gndvi, [10, 25, 50, 75, 90]),
        ])

        # 3. RGB spatial quantiles & color ratios (captures localized orange/brown rust spots)
        c_flat = rgb.reshape(-1, 3)
        q_rgb = np.percentile(c_flat, [10, 25, 50, 75, 90], axis=0).flatten()
        r_g = (rgb[:, :, 0] - rgb[:, :, 1]) / (rgb[:, :, 0] + rgb[:, :, 1] + 1e-6)
        g_b = (rgb[:, :, 1] - rgb[:, :, 2]) / (rgb[:, :, 1] + rgb[:, :, 2] + 1e-6)
        q_rg = np.percentile(r_g, [10, 25, 50, 75, 90, 95])
        q_gb = np.percentile(g_b, [10, 25, 50, 75, 90, 95])

        vec = np.concatenate([
            mean_spec, std_spec, d1, d2,
            sub25, sub25_std, np.array(ratios, dtype=np.float32),
            q_ms, dx_ms, dy_ms, q_vi,
            q_rgb, q_rg, q_gb,
        ])
        feats.append(vec)

    X = np.nan_to_num(np.array(feats, dtype=np.float32))
    return X, np.array(valid_hs_flags, dtype=np.float32)


@torch.no_grad()
def predict_dl_tta(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Run 3x Test-Time Augmentation (Original + HFlip + VFlip) with SS-CFNet v5."""
    model.eval()
    probs_list = []
    for batch in loader:
        hs, ms, rgb = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        p1 = F.softmax(model(hs, ms, rgb), dim=1)
        p2 = F.softmax(model(torch.flip(hs, [3]), torch.flip(ms, [3]), torch.flip(rgb, [3])), dim=1)
        p3 = F.softmax(model(torch.flip(hs, [2]), torch.flip(ms, [2]), torch.flip(rgb, [2])), dim=1)
        p = (p1 + p2 + p3) / 3.0
        probs_list.append(p.cpu().numpy())
    return np.vstack(probs_list)


def blend_probabilities(
    p_dl: np.ndarray,
    p_svm: np.ndarray,
    p_lr: np.ndarray,
    valid_hs: np.ndarray,
    w_dl: float = 0.30,
    w_svm: float = 0.50,
    w_lr: float = 0.20,
) -> np.ndarray:
    """Blend Deep Learning & Spectral ML probabilities with automatic Zero-HS Fallback."""
    p_ens = np.zeros_like(p_dl)
    for i in range(len(p_dl)):
        if valid_hs[i] > 0.5:
            p_ens[i] = w_dl * p_dl[i] + w_svm * p_svm[i] + w_lr * p_lr[i]
        else:
            # Corrupt/zero HS tensor -> rely 100% on RGB-driven Deep Learning prediction
            p_ens[i] = p_dl[i]
    return p_ens


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Ensemble v6 (SS-CFNet v5 + Spectral SVM-RBF + LogReg)")
    parser.add_argument(
        "--dl_checkpoint",
        type=str,
        default="checkpoints/multimodal_resnet18_imgsize224_batch16_epoch100_lr0.0003.pth",
        help="Path to best SS-CFNet v5 PyTorch checkpoint",
    )
    parser.add_argument("--w_dl", type=float, default=0.30, help="Ensemble weight for SS-CFNet v5 + TTA")
    parser.add_argument("--w_svm", type=float, default=0.50, help="Ensemble weight for Spectral SVM-RBF")
    parser.add_argument("--w_lr", type=float, default=0.20, help="Ensemble weight for Spectral LogReg")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="AI-for-Agriculture-2026")
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", "thanhphat20502-studocu"))
    parser.add_argument("--wandb_run_name", type=str, default="Hybrid_Ensemble_v6_SSCFNet_SpectralSVM")
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=os.environ.get(
            "WANDB_API_KEY",
            "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",
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    cfg = load_config([
        "src/configs/paths_local.yaml",
        "src/configs/model.yaml",
        "src/configs/train.yaml",
        "src/configs/dataset.yaml",
        "src/configs/experiments/sg_cfnet_multimodal.yaml",
    ])
    set_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_files, val_files, class_to_idx = load_split_manifest(cfg.SPLIT_MANIFEST_PATH)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    y_tr = np.array([class_to_idx[f.split("_")[0]] for f in train_files])
    y_va = np.array([class_to_idx[f.split("_")[0]] for f in val_files])

    print("=" * 72)
    print("  HYBRID ENSEMBLE v6: SS-CFNet v5 (TTA) + Spectral SVM-RBF + LogReg")
    print("=" * 72)

    # 1. Extract Chemometric Features
    print("\n[1/5] Extracting 916-dim chemometric spectral features (Train & Val)...")
    X_tr, valid_tr = extract_chemometric_features(cfg.TRAIN_HS_DIR, cfg.TRAIN_MS_DIR, cfg.TRAIN_RGB_DIR, train_files)
    X_va, valid_va = extract_chemometric_features(cfg.TRAIN_HS_DIR, cfg.TRAIN_MS_DIR, cfg.TRAIN_RGB_DIR, val_files)
    nz_tr = valid_tr > 0.5
    print(f"      Feature dim: {X_tr.shape[1]} | Valid HS train: {int(nz_tr.sum())}/{len(y_tr)}, val: {int(valid_va.sum())}/{len(y_va)}")

    # 2. Fit Chemometric Spectral Models + 5-Fold Stratified CV Audit (Zero-Overfit Verification)
    print("\n[2/5] Training Spectral SVM-RBF & LogReg (with 5-Fold Stratified CV audit)...")
    svm_clf = make_pipeline(StandardScaler(), SVC(C=5.0, kernel="rbf", probability=True, random_state=cfg.SEED))
    lr_clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=2000, random_state=cfg.SEED))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
    oof_svm = cross_val_predict(svm_clf, X_tr[nz_tr], y_tr[nz_tr], cv=skf, method="predict_proba")
    oof_lr = cross_val_predict(lr_clf, X_tr[nz_tr], y_tr[nz_tr], cv=skf, method="predict_proba")
    oof_svm_acc = accuracy_score(y_tr[nz_tr], oof_svm.argmax(1))
    oof_lr_acc = accuracy_score(y_tr[nz_tr], oof_lr.argmax(1))
    print(f"      5-Fold Train OOF Acc -> Spectral SVM-RBF: {oof_svm_acc:.4f} | Spectral LogReg: {oof_lr_acc:.4f}")

    svm_clf.fit(X_tr[nz_tr], y_tr[nz_tr])
    lr_clf.fit(X_tr[nz_tr], y_tr[nz_tr])

    p_va_svm = svm_clf.predict_proba(X_va)
    p_va_lr = lr_clf.predict_proba(X_va)

    # 3. Load SS-CFNet v5 Checkpoint & Run 3x TTA
    print(f"\n[3/5] Loading SS-CFNet v5 checkpoint & running 3x TTA: {args.dl_checkpoint}")
    model = build_multimodal_model(cfg, device)
    ckpt = torch.load(args.dl_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, tfm_val = get_transforms(cfg)
    val_ds = MultimodalDataset(
        cfg.TRAIN_HS_DIR,
        cfg.TRAIN_MS_DIR,
        cfg.TRAIN_RGB_DIR,
        transform=tfm_val,
        file_list=val_files,
        class_to_idx=class_to_idx,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    p_va_dl = predict_dl_tta(model, val_loader, device)

    # 4. Hybrid Ensemble Blending & Evaluation
    print(f"\n[4/5] Evaluating Hybrid Ensemble (DL={args.w_dl:.2f}, SVM={args.w_svm:.2f}, LR={args.w_lr:.2f})...")
    p_va_ens = blend_probabilities(p_va_dl, p_va_svm, p_va_lr, valid_va, args.w_dl, args.w_svm, args.w_lr)
    y_pred = p_va_ens.argmax(1)

    val_acc = accuracy_score(y_va, y_pred)
    val_f1 = f1_score(y_va, y_pred, average="macro", zero_division=0)
    val_prec = precision_score(y_va, y_pred, average="macro", zero_division=0)
    val_rec = recall_score(y_va, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_va, y_pred)
    rep_dict = classification_report(y_va, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    print("\n" + "=" * 72)
    print(f"  HYBRID ENSEMBLE v6 VALIDATION RESULTS (N={len(y_va)})")
    print("=" * 72)
    print(f"  Accuracy        : {val_acc:.4f} ({int((y_pred == y_va).sum())}/{len(y_va)})")
    print(f"  Macro F1        : {val_f1:.4f}")
    print(f"  Macro Precision : {val_prec:.4f}")
    print(f"  Macro Recall    : {val_rec:.4f}")
    print("\nConfusion Matrix (rows=True, cols=Pred):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))
    print("\nClassification Report:")
    print(classification_report(y_va, y_pred, target_names=class_names, digits=4))

    # Save Evaluation Artifacts
    output_dir = getattr(cfg, "OUTPUT_DIR", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    cm_paths = save_confusion_matrix_artifacts(y_va, y_pred, class_names, output_dir, prefix="hybrid_v6_val")
    rep_paths = save_classification_report_artifacts(rep_dict, output_dir, prefix="hybrid_v6_val")

    # Save Hybrid Ensemble Artifact (.joblib)
    ensemble_ckpt_path = os.path.join("checkpoints", "hybrid_ensemble_v6.joblib")
    joblib.dump(
        {
            "svm_clf": svm_clf,
            "lr_clf": lr_clf,
            "weights": {"dl": args.w_dl, "svm": args.w_svm, "lr": args.w_lr},
            "dl_checkpoint": args.dl_checkpoint,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "oof_svm_acc": oof_svm_acc,
        },
        ensemble_ckpt_path,
    )
    print(f"\n[Saved] Hybrid Ensemble checkpoint: {ensemble_ckpt_path}")

    # 5. Test Set Inference (300 unlabeled images in Data/val/)
    submission_path = None
    if getattr(cfg, "TEST_RGB_DIR", None) and os.path.exists(cfg.TEST_RGB_DIR):
        print("\n[5/5] Running Hybrid Ensemble v6 inference on 300 submission test images...")
        test_ds = MultimodalTestDataset(cfg.TEST_HS_DIR, cfg.TEST_MS_DIR, cfg.TEST_RGB_DIR, transform=tfm_val)
        test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
        test_files = test_ds.files

        X_te, valid_te = extract_chemometric_features(cfg.TEST_HS_DIR, cfg.TEST_MS_DIR, cfg.TEST_RGB_DIR, test_files)
        p_te_svm = svm_clf.predict_proba(X_te)
        p_te_lr = lr_clf.predict_proba(X_te)
        p_te_dl = predict_dl_tta(model, test_loader, device)

        p_te_ens = blend_probabilities(p_te_dl, p_te_svm, p_te_lr, valid_te, args.w_dl, args.w_svm, args.w_lr)
        te_preds = p_te_ens.argmax(1)

        sub_df = pd.DataFrame({
            "Id": test_files,
            "Category": [idx_to_class[int(c)] for c in te_preds],
        })
        submission_path = "submission_hybrid_v6.csv"
        sub_df.to_csv(submission_path, index=False)
        shutil.copy2(submission_path, os.path.join(output_dir, "submission_hybrid_v6.csv"))
        shutil.copy2(submission_path, "submission.csv")
        print(f"      Saved submission to: {submission_path} (and updated submission.csv)")
        print("      Test prediction distribution:")
        print(sub_df["Category"].value_counts().to_string())

    # 6. Log to WandB
    if args.wandb:
        import wandb

        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "architecture": "Hybrid_Ensemble_v6 (SS-CFNet_v5_TTA + Spectral_SVM_RBF + Spectral_LogReg)",
                "w_dl": args.w_dl,
                "w_svm": args.w_svm,
                "w_lr": args.w_lr,
                "dl_checkpoint": args.dl_checkpoint,
                "chemometric_feature_dim": int(X_tr.shape[1]),
                "zero_hs_fallback": True,
                "seed": cfg.SEED,
            },
        )

        log_dict = {
            "accuracy/val": val_acc,
            "val/acc": val_acc,
            "metrics/val_f1_macro": val_f1,
            "metrics/val_precision_macro": val_prec,
            "metrics/val_recall_macro": val_rec,
            "train/oof_svm_acc": oof_svm_acc,
            "train/oof_lr_acc": oof_lr_acc,
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=y_va.tolist(), preds=y_pred.tolist(), class_names=class_names
            ),
        }
        for c_name in class_names:
            log_dict[f"class_f1/{c_name}"] = rep_dict[c_name]["f1-score"]
            log_dict[f"class_recall/{c_name}"] = rep_dict[c_name]["recall"]
            log_dict[f"class_precision/{c_name}"] = rep_dict[c_name]["precision"]

        cm_png = cm_paths.get("png")
        if cm_png and os.path.exists(cm_png):
            log_dict["confusion_matrix_heatmap"] = wandb.Image(cm_png, caption="Hybrid Ensemble v6 Confusion Matrix")
            log_dict["val/confusion_matrix_image"] = wandb.Image(cm_png, caption="Hybrid Ensemble v6 Confusion Matrix")

        wandb.log(log_dict)

        for k, v in log_dict.items():
            if isinstance(v, (int, float)):
                wandb.summary[k] = v

        # Upload Hybrid Ensemble model artifact
        model_art = wandb.Artifact("hybrid-ensemble-v6", type="model")
        model_art.add_file(ensemble_ckpt_path)
        wandb.log_artifact(model_art)

        if submission_path and os.path.exists(submission_path):
            sub_art = wandb.Artifact("submission-hybrid-v6", type="dataset")
            sub_art.add_file(submission_path)
            wandb.log_artifact(sub_art)

        wandb.finish()
        print(f"\n[WandB] Successfully logged run '{args.wandb_run_name}'!")

    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
