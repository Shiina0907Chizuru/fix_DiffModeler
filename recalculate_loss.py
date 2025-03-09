import re
import os
import numpy as np
import argparse
from enum import Enum

class Mode(Enum):
    TRAIN = "train"
    VAL = "val"
    
    def __str__(self):
        return self.value

def recalculate_metrics(log_file_path, output_file, mode=Mode.VAL, loss_threshold=0.99, iou_threshold=0.01):
    """
    Calculate average loss (and IoU for training) per epoch from log files, excluding abnormal values.
    
    Args:
        log_file_path: Path to the log file
        output_file: Path to the output file
        mode: Mode.TRAIN or Mode.VAL to specify processing mode
        loss_threshold: Loss values greater than this threshold are considered abnormal
        iou_threshold: (Train mode only) IoU values less than this threshold are considered abnormal
    """
    # Check if file exists
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} not found.")
        return
    
    print(f"Processing {mode.value} log file: {log_file_path}")
    print(f"Mode: {mode.value}, Loss threshold: {loss_threshold}, IoU threshold: {iou_threshold}")
    
    # Dictionary to store metrics for each epoch
    epoch_data = {}
    
    # Patterns for different types of logs
    loss_patterns = [
        r"Epoch[: ]*(\d+).*loss[: ]*([\d\.]+)",
        r"epoch[: ]*(\d+).*loss[: ]*([\d\.]+)",
        r"Epoch[: ]*(\d+).*Loss[: ]*([\d\.]+)",
        r".*epoch.*?(\d+).*?loss.*?([\d\.]+)",
        r".*Epoch.*?(\d+).*?loss.*?([\d\.]+)"
    ]
    
    # IoU patterns (for training mode)
    iou_patterns = [
        r"Epoch[: ]*(\d+).*iou[: ]*([\d\.]+)",
        r"epoch[: ]*(\d+).*iou[: ]*([\d\.]+)",
        r"Epoch[: ]*(\d+).*IoU[: ]*([\d\.]+)",
        r".*epoch.*?(\d+).*?iou.*?([\d\.]+)",
        r".*Epoch.*?(\d+).*?IoU.*?([\d\.]+)"
    ]
    
    # Count total lines and matches
    total_lines = 0
    loss_matches = 0
    iou_matches = 0
    
    # Read the log file
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            # Try to match loss patterns
            loss_matched = False
            for pattern in loss_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    loss_matched = True
                    loss_matches += 1
                    try:
                        epoch = int(match.group(1))
                        loss = float(match.group(2))
                        
                        # Initialize epoch data if not exists
                        if epoch not in epoch_data:
                            epoch_data[epoch] = {
                                'losses': [],
                                'ious': [] if mode == Mode.TRAIN else None
                            }
                        
                        # Skip abnormal loss values
                        if loss > loss_threshold:
                            continue
                        
                        # Add loss to epoch data
                        epoch_data[epoch]['losses'].append(loss)
                        
                        # Print progress periodically
                        if (loss_matches + iou_matches) % 5000 == 0:
                            print(f"Processed {loss_matches} loss and {iou_matches} IoU matches from {total_lines} lines")
                            
                    except (ValueError, IndexError) as e:
                        print(f"Error processing loss in line {line_num}: {e}")
                    
                    break  # Stop trying patterns once we find a match
            
            # For training mode, try to match IoU patterns
            if mode == Mode.TRAIN:
                iou_matched = False
                for pattern in iou_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        iou_matched = True
                        iou_matches += 1
                        try:
                            epoch = int(match.group(1))
                            iou = float(match.group(2))
                            
                            # Initialize epoch data if not exists
                            if epoch not in epoch_data:
                                epoch_data[epoch] = {
                                    'losses': [],
                                    'ious': []
                                }
                            
                            # Skip abnormal IoU values
                            if iou < iou_threshold:
                                continue
                            
                            # Add IoU to epoch data
                            epoch_data[epoch]['ious'].append(iou)
                            
                        except (ValueError, IndexError) as e:
                            print(f"Error processing IoU in line {line_num}: {e}")
                        
                        break  # Stop trying patterns once we find a match
            
            # Debug: print a sample of unmatched lines
            if mode == Mode.TRAIN and not (loss_matched or iou_matched) and line_num <= 10:
                print(f"Unmatched line {line_num}: {line.strip()}")
            elif mode == Mode.VAL and not loss_matched and line_num <= 10:
                print(f"Unmatched line {line_num}: {line.strip()}")
    
    print(f"Finished processing {total_lines} lines with {loss_matches} loss and {iou_matches} IoU matches")
    print(f"Found data for {len(epoch_data)} epochs")
    
    # Calculate statistics for each epoch
    results = []
    for epoch in sorted(epoch_data.keys()):
        epoch_item = epoch_data[epoch]
        
        # Process loss data
        losses = epoch_item['losses']
        if losses:
            avg_loss = np.mean(losses)
            min_loss = min(losses)
            max_loss = max(losses)
            std_loss = np.std(losses) if len(losses) > 1 else 0
            
            result_line = (f"Epoch {epoch}: Average Loss = {avg_loss:.6f} "
                           f"(Min: {min_loss:.6f}, Max: {max_loss:.6f}, Std: {std_loss:.6f}, "
                           f"from {len(losses)} samples, excluding values > {loss_threshold})")
            
            # Process IoU data for training mode
            if mode == Mode.TRAIN and epoch_item['ious']:
                ious = epoch_item['ious']
                avg_iou = np.mean(ious)
                min_iou = min(ious)
                max_iou = max(ious)
                std_iou = np.std(ious) if len(ious) > 1 else 0
                
                result_line += (f"\n    Average IoU = {avg_iou:.6f} "
                               f"(Min: {min_iou:.6f}, Max: {max_iou:.6f}, Std: {std_iou:.6f}, "
                               f"from {len(ious)} samples, excluding values < {iou_threshold})")
            
            results.append(result_line)
    
    # Write results to output file
    with open(output_file, 'w') as f:
        f.write("\n".join(results))
    
    print(f"Results saved to {output_file}")
    return epoch_data

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate metrics from log files.')
    parser.add_argument('--mode', type=str, choices=['train', 'val'], default='val',
                        help='Processing mode: "train" or "val" (default: val)')
    parser.add_argument('--input', type=str, 
                        help='Input log file path (default: depends on mode)')
    parser.add_argument('--output', type=str, 
                        help='Output file path (default: depends on mode)')
    parser.add_argument('--loss-threshold', type=float, default=0.99,
                        help='Loss threshold, values above this are excluded (default: 0.99)')
    parser.add_argument('--iou-threshold', type=float, default=0.09,
                        help='IoU threshold for train mode, values below this are excluded (default: 0.09)')
    
    args = parser.parse_args()
    
    # Set default paths based on mode if not provided
    if args.input is None:
        args.input = "C:\\Users\\Z\\Desktop\\train.log" if args.mode == 'train' else "C:\\Users\\Z\\Desktop\\val.log"
    
    if args.output is None:
        args.output = "re_train_metrics.txt" if args.mode == 'train' else "re_val_loss.txt"
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    mode = Mode.TRAIN if args.mode == 'train' else Mode.VAL
    
    recalculate_metrics(
        log_file_path=args.input,
        output_file=args.output,
        mode=mode,
        loss_threshold=args.loss_threshold,
        iou_threshold=args.iou_threshold
    )
