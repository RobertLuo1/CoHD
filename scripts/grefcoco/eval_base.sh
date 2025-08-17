python train_net.py \
    --config-file configs/referring_swin_base_eval.yaml \
    --num-gpus 8 --dist-url auto \
    --eval-only \
    MODEL.WEIGHTS "../checkpoints/upload_checkpoint/grefcoco_swin_base/CoHD_grefcoco_swin_base.pth" \
    OUTPUT_DIR "../results/CoHD_Grefcoco_Swin_base_eval"