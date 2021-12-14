# DDP train
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py \
  --opt options/swinir/train_swinir_sr_classical_x2.json  --dist True

# DP train
python main_train_psnr.py --opt options/swinir/train_swinir_sr_classical_x2.json

# test debug
python main_test_swinir.py --opt options/swinir/train_swinir_sr_classical_x2.json

# test model_zoo
python main_test_swinir_orig.py --task classical_sr --scale 2 --training_patch_size 48 \
  --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth \
  --folder_lq /mnt/cfs/algorithm/public_data/SR/test/Set5/LRbicx2 \
  --folder_gt /mnt/cfs/algorithm/public_data/SR/test/Set5/GTmod12

# test local model
python main_test_swinir_orig.py --task classical_sr --scale 2 --training_patch_size 48 \
  --model_path superresolution/swinir_sr_classical_patch48_x2/models/500000_G.pth \
  --folder_lq /mnt/cfs/algorithm/public_data/SR/test/Set5/LRbicx2 \
  --folder_gt /mnt/cfs/algorithm/public_data/SR/test/Set5/GTmod12
