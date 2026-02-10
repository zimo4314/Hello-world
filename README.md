# Hello-world

LTGQ++ Fusion Model with Ablation Study Support

## Files

- **train** - Original LTGQ++ Fusion training script (1324 lines)
- **train_ablation.py** - Complete ablation experiment version (1349 lines)
  - Adds three ablation switches: `--disable_qfm`, `--disable_dtm`, `--disable_history`
  - All other code identical to original
- **ABLATION_GUIDE.md** - Comprehensive ablation study guide
- **compare_train_files.py** - Helper script to compare train files

## Quick Start

### Full Model Training
```bash
python train_ablation.py --save_dir ./results/full
```

### Ablation Experiments
```bash
# Without QFM
python train_ablation.py --disable_qfm --save_dir ./results/wo_qfm

# Without DTM
python train_ablation.py --disable_dtm --save_dir ./results/wo_dtm

# Without History
python train_ablation.py --disable_history --save_dir ./results/wo_history
```

See **ABLATION_GUIDE.md** for detailed documentation.

