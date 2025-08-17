python train_net.py \
    --config-file configs/referring_swin_tiny_eval.yaml \
    --num-gpus 8 --dist-url auto \
    --eval-only \
    MODEL.WEIGHTS "../checkpoints/upload_checkpoint/grefcoco_swin_tiny/CoHD_grefcoco_swin_tiny.pth" \
    OUTPUT_DIR "../results/CoHD_Grefcoco_Swin_tiny_eval"