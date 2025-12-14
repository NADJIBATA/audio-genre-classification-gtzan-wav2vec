from fairseq import checkpoint_utils

try:
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(['models/wav2vec/checkpoints/checkpoint12.pt'])
    print('Loaded models:', len(models))
    print('CFG type:', type(cfg))
    print('Task type:', type(task))
except Exception as e:
    import traceback
    traceback.print_exc()
    print('ERROR:', e)
