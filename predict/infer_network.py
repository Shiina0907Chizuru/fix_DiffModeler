import os
import torch
from model.config_networks_new import define_network
from ops.os_operation import mkdir
from data_processing.Test_Dataset import Single_Dataset

def infer_network(input_map_path, save_dir, params):
    """Run inference with the new network.
    
    Args:
        input_map_path (str): Path to input map file
        save_dir (str): Directory to save results
        params (dict): Parameters for inference
    
    Returns:
        str: Path to output file
    """
    mkdir(save_dir)
    final_output_path = os.path.join(save_dir, "output.mrc")
    
    if os.path.exists(final_output_path):
        return final_output_path
        
    if not os.path.exists(params['model']['path']):
        raise ValueError("Please configure the model path in config json file.")
        
    params['model']['path'] = os.path.abspath(params['model']['path'])
    
    # Initialize model
    network = define_network(params)
    
    # Setup input data
    save_input_dir = os.path.join(save_dir, "Input")
    test_dataset = Single_Dataset(save_input_dir, "input_")
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        pin_memory=True,
        batch_size=params['model']['batch_size'],
        num_workers=params['model']['num_workers'],
        drop_last=False
    )
    
    print(f"Dataset loading finished with {len(test_dataset)} samples!")
    
    # Load model weights
    resume_model_path = os.path.abspath(params['model']['path'])
    network.load_network(resume_model_path)
    
    # Run inference
    network.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            network.feed_data(data)
            output = network.inference()
            
            # Save output
            # TODO: Implement your output saving logic here
            # This depends on your specific needs
            
    print(f"Inference finished! Output saved to {final_output_path}")
    
    return final_output_path
