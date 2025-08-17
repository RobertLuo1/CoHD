export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export TORCH_DISTRIBUTED_DEBUG=INFO
python train_net.py \
    --config-file configs/referring_swin_tiny.yaml \
    --num-gpus 8 --dist-url auto \
    OUTPUT_DIR "../../results/CoHD_Grefcoco_Swin_tiny"