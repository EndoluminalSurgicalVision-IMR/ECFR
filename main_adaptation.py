import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.build_dataset import StreamingMedicalDataset
from models.memory_bank import DynamicMemoryBank
from methods.ecfr import ECFRAdaptation, configure_ln_parameters
from methods.grata import GradientAlignmentAdaptation
from utils.metrics import compute_accuracy, compute_brier_score, compute_ece
from utils.visualization import plot_ecfr_2d_dynamics

def parse_args():
    parser = argparse.ArgumentParser(description="Test-Time Adaptation for Medical FMs")
    parser.add_argument('--dataset_csv', type=str, required=True, help="Path to test dataset CSV")
    parser.add_argument('--data_root', type=str, default="", help="Optional root directory for relative file paths in the CSV")
    parser.add_argument('--method', type=str, choices=['ecfr', 'grata'], default='ecfr')
    parser.add_argument('--lr', type=float, required=True, help="Learning rate selected from the predefined sweep")
    parser.add_argument('--input_size', type=int, default=224, help="Input resolution for the image transforms")
    parser.add_argument('--demo_backbone', type=str, choices=['vit_b_16', 'resnet50'], default='vit_b_16')
    parser.add_argument(
        '--adapt_params',
        type=str,
        choices=['layernorm', 'norm'],
        default='layernorm',
        help="Parameter subset for TTA. Paper experiments use 'layernorm'. Use 'norm' only for demo backbones without LayerNorm.",
    )
    parser.add_argument('--capacity', type=int, default=128, help="Memory bank capacity")
    parser.add_argument(
        '--warmup_size',
        type=int,
        default=5,
        help="Minimum memory length required before signed max updates are activated",
    )
    parser.add_argument('--ema_alpha', type=float, default=0.999, help="EMA momentum for the teacher model")
    parser.add_argument('--restore_prob', type=float, default=0.01, help="Stochastic restoration probability")
    parser.add_argument(
        '--quantile',
        type=float,
        default=0.5,
        help="Dynamic-threshold quantile: 0.5 for ECFR-0.5, or prior target accuracy for ECFR-Acc",
    )
    parser.add_argument('--output_csv', type=str, default=None, help="Optional path to append run metrics")
    parser.add_argument('--plot_path', type=str, default='outputs/ECFR_Quadrant_Flow.png', help="Path for the ECFR flow plot")
    parser.add_argument('--no_plot', action='store_true', help="Disable ECFR flow plotting")
    parser.add_argument('--seed', type=int, default=2026, help="Random seed")
    args = parser.parse_args()
    if not 0.0 <= args.quantile <= 1.0:
        raise ValueError("--quantile must be in [0, 1]; use 0.5 for ECFR-0.5 or prior accuracy for ECFR-Acc.")
    if not 0.0 <= args.restore_prob <= 1.0:
        raise ValueError("--restore_prob must be in [0, 1].")
    if not 0.0 <= args.ema_alpha < 1.0:
        raise ValueError("--ema_alpha must be in [0, 1).")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_demo_model(backbone):
    """Build a compact demo backbone.

    Replace this function with the official Ark+ or DermLIP loader when
    reproducing the paper results.
    """
    import torchvision.models as models

    if backbone == "vit_b_16":
        return models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    if backbone == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    raise ValueError(f"Unsupported demo backbone: {backbone}")


def append_metrics(output_csv, row):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method_name = "GraTa" if args.method == "grata" else "ECFR"
    print(f"Initializing {method_name} adaptation on {device}...")

    # 1. Initialize Dataset & Dataloader (Batch Size = 1 for continuous streaming)
    dataset = StreamingMedicalDataset(csv_path=args.dataset_csv, input_size=args.input_size, root_dir=args.data_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2. Initialize Model
    # In practice, load the Ark+ or DermLIP checkpoint used in your experiment.
    model = build_demo_model(args.demo_backbone).to(device)
    
    # 3. Configure parameter-efficient fine-tuning on normalization layers
    opt_params = configure_ln_parameters(model, mode=args.adapt_params)
    optimizer = optim.SGD(opt_params, lr=args.lr, momentum=0.9)

    # 4. Initialize Adaptation Method & Memory Bank
    if args.method == 'ecfr':
        method = ECFRAdaptation(
            model,
            optimizer,
            ema_alpha=args.ema_alpha,
            p_restore=args.restore_prob,
        )
        memory_bank = DynamicMemoryBank(
            capacity=args.capacity,
            quantile=args.quantile,
            warmup_size=args.warmup_size,
        )
    elif args.method == 'grata':
        method = GradientAlignmentAdaptation(model, optimizer)
    
    # 5. Streaming Metrics Tracking
    all_targets, all_probs = [], []
    all_entropies, all_consistencies = [], []
    
    # 6. Online Adaptation Loop
    model.train() # TTA requires train mode for LN updates
    
    for x_clean, x_weak, x_strong, target, _ in tqdm(dataloader, desc="Adapting"):
        x_clean, x_weak, x_strong = x_clean.to(device), x_weak.to(device), x_strong.to(device)
        
        if args.method == 'ecfr':
            logits, ent, cons = method.forward_and_adapt(x_clean, x_weak, x_strong, memory_bank)
            all_entropies.extend(ent.cpu().numpy())
            all_consistencies.extend(cons.cpu().numpy())
        elif args.method == 'grata':
            logits = method.forward_and_adapt(x_weak, x_strong)
            
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.detach().cpu())
        all_targets.extend(target.numpy())

    # 7. Evaluate Metrics
    final_probs = torch.cat(all_probs, dim=0)
    final_targets = torch.tensor(all_targets)
    
    acc = compute_accuracy(final_probs, final_targets)
    brier = compute_brier_score(final_probs, final_targets, num_classes=final_probs.shape[1])
    ece = compute_ece(final_probs, final_targets)
    
    print("\n" + "="*40)
    print(f"Results for {method_name}:")
    print(f"Accuracy: {acc:.4f} | Brier: {brier:.4f} | ECE: {ece:.4f}")
    print("="*40)

    if args.output_csv:
        append_metrics(
            args.output_csv,
            {
                "method": args.method,
                "lr": args.lr,
                "capacity": args.capacity,
                "quantile": args.quantile,
                "warmup_size": args.warmup_size,
                "ema_alpha": args.ema_alpha,
                "restore_prob": args.restore_prob,
                "input_size": args.input_size,
                "adapt_params": args.adapt_params,
                "accuracy": acc,
                "brier": brier,
                "ece": ece,
            },
        )
        print(f"Appended metrics to {args.output_csv}")

    # 8. Visualization (ECFR only)
    if args.method == 'ecfr' and not args.no_plot and len(all_entropies) > 0:
        ent_thresh, cons_thresh = memory_bank.get_thresholds()
        
        # One-Hot Brier Score
        targets_one_hot = torch.nn.functional.one_hot(final_targets, num_classes=final_probs.shape[1]).numpy()
        brier_solos = np.sum((final_probs.numpy() - targets_one_hot)**2, axis=1)
        
        plot_ecfr_2d_dynamics(
            entropies=np.array(all_entropies), 
            cons_losses=np.array(all_consistencies), 
            brier_solo_values=brier_solos, 
            entropy_thresh=ent_thresh, 
            cons_thresh=cons_thresh,
            save_path=args.plot_path,
        )
        print(f"Generated entropy-consistency scatter plot: {args.plot_path}")

if __name__ == "__main__":
    main()
