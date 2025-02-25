import torch
import argparse
import matplotlib.pyplot as plt
import json

def load_and_check_model(model_path):
    # 加载模型文件
    print(f"\nLoading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 打印所有的键和相关信息
    print("\n=== Model checkpoint contents ===")
    for key in checkpoint.keys():
        print(f"\n{'='*20} {key} {'='*20}")
        if key == 'epoch':
            print(f"Current epoch: {checkpoint[key]}")
        elif key == 'best_loss':
            print(f"Best loss: {checkpoint[key]}")
        elif key == 'val_loss':
            print(f"Validation loss: {checkpoint[key]}")
        elif key == 'state_dict':
            print(f"Total parameters: {len(checkpoint[key])}")
            print("\nFirst 5 layer parameters:")
            for i, (param_name, param) in enumerate(checkpoint[key].items()):
                if i >= 5:
                    break
                print(f"{param_name}:")
                print(f"  Shape: {tuple(param.shape)}")
                print(f"  Dtype: {param.dtype}")
                print(f"  Value range: [{param.min().item():.6f}, {param.max().item():.6f}]")
        elif key == 'optimizer':
            print("Optimizer info:")
            print(f"Type: {type(checkpoint[key])}")
            print("\nOptimizer content:")
            try:
                if isinstance(checkpoint[key], dict):
                    print(json.dumps(checkpoint[key], indent=2))
                else:
                    if hasattr(checkpoint[key], 'state'):
                        print("State keys:", list(checkpoint[key].state.keys()))
                    if hasattr(checkpoint[key], 'param_groups'):
                        for i, group in enumerate(checkpoint[key].param_groups):
                            print(f"\nParameter group {i}:")
                            for k, v in group.items():
                                if k != 'params':
                                    print(f"  {k}: {v}")
            except Exception as e:
                print(f"Error accessing optimizer content: {e}")
        elif key == 'scheduler':
            print("Scheduler info:")
            print(f"Type: {type(checkpoint[key])}")
            print("\nScheduler content:")
            try:
                if isinstance(checkpoint[key], dict):
                    scheduler_dict = {}
                    for k, v in checkpoint[key].items():
                        if isinstance(v, torch.Tensor):
                            scheduler_dict[k] = v.tolist()
                        else:
                            scheduler_dict[k] = v
                    print(json.dumps(scheduler_dict, indent=2))
                else:
                    for attr in ['base_lrs', 'last_epoch', '_step_count', 'verbose', 'optimizer']:
                        if hasattr(checkpoint[key], attr):
                            value = getattr(checkpoint[key], attr)
                            if isinstance(value, torch.Tensor):
                                value = value.tolist()
                            print(f"  {attr}: {value}")
            except Exception as e:
                print(f"Error accessing scheduler content: {e}")
        elif key == 'loss_record':
            losses = checkpoint[key]
            plt.figure(figsize=(10, 6))
            plt.plot(losses)
            plt.title('Training Loss Curve')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('loss_curve.png')
            plt.close()
            print("\nLoss curve has been saved as 'loss_curve.png'")
            
            # 打印一些统计信息
            print(f"\nLoss Statistics:")
            print(f"Initial loss: {losses[0]:.4f}")
            print(f"Final loss: {losses[-1]:.4f}")
            print(f"Minimum loss: {min(losses):.4f}")
            print(f"Maximum loss: {max(losses):.4f}")
        else:
            if isinstance(checkpoint[key], (int, float, str, bool)):
                print(checkpoint[key])
            else:
                print(f"Type: {type(checkpoint[key])}")
                try:
                    print(json.dumps(checkpoint[key], indent=2))
                except:
                    print(str(checkpoint[key]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check model checkpoint and visualize loss')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    
    args = parser.parse_args()
    load_and_check_model(args.model_path)
