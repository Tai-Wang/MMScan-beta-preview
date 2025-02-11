export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --checkpoint_dir ./ckpts/opt-1.3b/ll3da-mmscan-tuned-human_check \
    --test_ckpt /mnt/petrelfs/linjingli/mmscan_modelzoo-main/llmzoo/LL3DA/ckpts/opt-1.3b/ll3da-mmscan-tuned/checkpoint_140k.pth \
    --dataset comparison \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --dist_url tcp://localhost:1233 \
    --criterion 'CiDEr' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 8 --ngpus 2 \
    --max_des_len 224 \
    --max_prompt 1 \
    --use_beam_search \
    --test_only

