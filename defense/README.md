# GAKer defense baseline

This folder implements the **recommended baseline** you confirmed:

- **training:** `(x_adv, x_s)`
- **inference:** only `x_adv`
- **target image:** optional training-only supervision, disabled by default

## Folder contents

- `dataset_builder.py` — creates `defense/dataset/train` and `defense/dataset/test` from an exported adversarial dataset
- `datasets.py` — paired dataset loader using `adv` + `source`
- `model.py` — encoder → latent purifier → decoder baseline
- `losses.py` — pixel + source-feature + classification losses
- `train_defense.py` — train baseline defense on `train/` only
- `evaluate_defense.py` — evaluate checkpoint later on held-out `test/`
- `infer_defense.py` — runtime purification using only adversarial image input

## 1) Build defense dataset split

Example:

```bash
python -m defense.dataset_builder \
  --export_root ./exported_adv/resnet50_gaker_imagenet_known_2000 \
  --output_root ./defense/dataset \
  --train_ratio 0.7 \
  --seed 42 \
  --use_symlink true
```

This creates:

```text
defense/
  dataset/
    train/
      adv/
      source/
      target/
      metadata.jsonl
    test/
      adv/
      source/
      target/
      metadata.jsonl
```

## 2) Train defense

This script uses only `defense/dataset/train`. It does **not** evaluate on `test/` during training.
`best_defense.pt` is selected by the **lowest training loss** only.

Example for ImageNet / ResNet-50 frozen source model:

```bash
python -m defense.train_defense \
  --dataset_root ./defense/dataset \
  --save_dir ./defense/checkpoints/imagenet_resnet50_baseline \
  --image_size 224 \
  --batch_size 16 \
  --epochs 20 \
  --source_model_name resnet50 \
  --source_model_num_classes 1000 \
  --use_imagenet_pretrained true \
  --lambda_pix 1.0 \
  --lambda_src_feat 1.0 \
  --lambda_cls 0.25
```

## 3) Evaluate defense

```bash
python -m defense.evaluate_defense \
  --split_root ./defense/dataset/test \
  --checkpoint_path ./defense/checkpoints/imagenet_resnet50_baseline/best_defense.pt \
  --image_size 224 \
  --batch_size 16 \
  --source_model_name resnet50 \
  --source_model_num_classes 1000 \
  --use_imagenet_pretrained true
```

## 4) Inference with only adversarial image input

Single image:

```bash
python -m defense.infer_defense \
  --checkpoint_path ./defense/checkpoints/imagenet_resnet50_baseline/best_defense.pt \
  --input_image ./defense/dataset/test/adv/sample_000001.png \
  --output_dir ./defense/inference_outputs
```

Entire folder:

```bash
python -m defense.infer_defense \
  --checkpoint_path ./defense/checkpoints/imagenet_resnet50_baseline/best_defense.pt \
  --input_dir ./defense/dataset/test/adv \
  --output_dir ./defense/inference_outputs
```

## Notes

- The baseline is intentionally simple and deployable first.
- Anti-target supervision exists in the code through `--lambda_anti_tgt`, but it is **off by default**.
- Runtime input remains only the perturbed image.
- `train_defense.py` and `evaluate_defense.py` are now fully separated: train on `train/`, test later on `test/`.
