###### 1. Train ######
python train.py \
--model_name resnet50 \
--num_classes 45 \
--gpu_ids [GPU_IDS] \
--batch_size 64 \
--num_workers 4 \
--dataset_path /home/igarss/NWPU-RESISC45 \
--dataset_name NWPU \
--exp_name NWPU \
--epoch 120 \
--lr 1e-3 \
--weight_decay 5e-4
###### 2. Search ######
python search.py \
--method Moea \
--net_path /home/igarss/search_results/NWPU/NWPU-RESISC45_pretrain_best0.918.pth \
--save_path search_results \
##### 3. Fine-tuning #######
python finetune.py \
--model_name resnet50 \
--num_classes 45 \
--net_path /home/igarss/search_results/NWPU/NWPU-RESISC45_pretrain_best0.918.pth \
--gpu_ids [GPU_IDS] \
--num_workers 4 \
--batch_size 128 \
--dataset_path /home/igarss/NWPU-RESISC45 \
--dataset_name NWPU \
--exp_name NWPU \
--epoch 50 \
--lr 1e-4 \
--weight_decay 5e-4\
--inputs /home/igarss/search_results/NWPU/7_3/nsga2_NWPU_pop_proph/Phen.csv\
--outputs /home/igarss/search_results/NWPU/7_3/nsga2_NWPU_pop_proph/output.csv
