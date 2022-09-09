TOTAL_UPDATES=500000    # Total number of training steps
WARMUP_UPDATES=25000    # Warmup the learning rate over this many updates

PEAK_LR=0.0005         # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=48        # Number of sequences per batch (batch size)
UPDATE_FREQ=4          # Increase the batch size
DECAY=0.01

DATA_DIR=./hf-wiki-books

python -m torch.distributed.launch --nproc_per_node=4 --master_port 29529 \
    $(which fairseq-train) --fp16 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch slstm1280_ln_posinput_layer24 --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.0 --weight-decay $DECAY --disable-validation --tensorboard-logdir './slstm/1280-layer24-lr5-4-tu500k-wu25k' \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ  --ddp-backend=legacy_ddp --skip-invalid-size-inputs-valid-test \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 2000  --save-dir './slstm/1280-layer24-lr5-4-tu500k-wu25k' --save-interval 1
