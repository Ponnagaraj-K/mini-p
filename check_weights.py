import torch, os

files = [
    'weights/best_generator.pt',
    'weights/best_generator (1).pt',
    'weights/checkpoint_epoch_50.pt',
    'weights/checkpoint_epoch_65.pt',
]

for path in files:
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        continue
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        g_loss = ckpt.get('g_loss', 'N/A')
        epoch  = ckpt.get('epoch', 'N/A')
        print(f"{path} -> epoch={epoch}, g_loss={g_loss}")
    else:
        print(f"{path} -> pure state_dict (no metadata)")
