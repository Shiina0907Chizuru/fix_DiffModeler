import os
import argparse
import shutil
from ops.os_operation import mkdir
import json
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description='DiffModeler Backbone Tracing')
    parser.add_argument('-F', type=str, required=True, help='input map path')
    parser.add_argument('--resolution', type=float, default=5.0, help='specify the resolution of the map')
    parser.add_argument('--config', type=str, default='config/diffmodeler.json', help='specifying the config path')
    parser.add_argument('--gpu', type=str, default=None, help='specify the gpu we will use')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--contour', type=float, default=0.0, help='Contour level for input map, suggested 0.5*[author_contour]. (Float), Default value: 0.0')
    parser.add_argument('--save_intermediate', action='store_true', help='Whether to save intermediate results')
    return parser.parse_args()

# Add a function to parse JSON with comments
def load_json_with_comments(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Remove inline comments (//...)
        content = re.sub(r'//.*?\n', '\n', content)
        # Remove trailing comments at end of lines
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        # Remove block comments (/* ... */)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Parse JSON
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing config file: {e}")
        print("Trying alternative method...")
        
        # If the above fails, try a more manual approach
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Remove comments and trailing commas
            cleaned_lines = []
            for line in lines:
                # Remove comments
                line = re.sub(r'//.*', '', line)
                cleaned_lines.append(line)
            
            cleaned_content = ''.join(cleaned_lines)
            return json.loads(cleaned_content)
        except Exception as e2:
            print(f"Failed to parse config file with second method: {e2}")
            exit(1)

def init_save_path(origin_map_path):
    save_path = os.path.join(os.getcwd(), 'Backbone_Trace_Result')
    mkdir(save_path)
    map_name = os.path.split(origin_map_path)[1].replace(".mrc", "")
    map_name = map_name.replace(".map", "")
    map_name = map_name.replace("(","").replace(")","")
    save_path = os.path.join(save_path, map_name)
    mkdir(save_path)
    return save_path, map_name

def set_up_environment(params):
    if params['resolution'] > 20:
        print("Maps with %.2f resolution is not supported! We only support maps with resolution 0-20A!" % params['resolution'])
        exit()
    
    # Set GPU
    gpu_id = params['gpu']
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # Process input map
    cur_map_path = os.path.abspath(params['F'])
    if cur_map_path.endswith(".gz"):
        from ops.os_operation import unzip_gz
        cur_map_path = unzip_gz(cur_map_path)

    # Setup output directory
    if params['output'] is None:
        save_path, map_name = init_save_path(cur_map_path)
    else:
        save_path = params['output']
        map_name = "input_backbone_trace"
        mkdir(save_path)
    
    # Pre-compile VESPER
    try:
        print("Pre-compiling VESPER to accelerate!")
        running_dir = os.path.dirname(os.path.abspath(__file__))
        os.system(f"cd {running_dir}; python -O -m compileall VESPER_CUDA")
    except:
        print("Pre-compile VESPER failed! No impact to main scripts!")
    
    save_path = os.path.abspath(save_path)
    
    # Process map
    from data_processing.Unify_Map import Unify_Map
    cur_map_path = Unify_Map(cur_map_path, os.path.join(save_path, map_name + "_unified.mrc"))
    
    from data_processing.Resize_Map import Resize_Map
    cur_map_path = Resize_Map(cur_map_path, os.path.join(save_path, map_name + ".mrc"))
    
    # Handle negative contour level
    if params['contour'] < 0:
        from ops.map_utils import increase_map_density
        cur_map_path = increase_map_density(cur_map_path, os.path.join(save_path, map_name+"_increase.mrc"), params['contour'])
        params['contour'] = 0
    
    # Segment map - 使用传入的contour参数而不是固定为0
    from modeling.map_utils import segment_map
    new_map_path = os.path.join(save_path, map_name + "_segment.mrc")
    segment_map(cur_map_path, new_map_path, contour=params['contour'])
    
    return save_path, new_map_path

def diffusion_trace_map(save_path, cur_map_path, params):
    if params['resolution'] >= 2:
        from predict.infer_diffusion import infer_diffem
        diffusion_dir = os.path.join(save_path, "infer_diffusion")
        diff_trace_map = infer_diffem(cur_map_path, diffusion_dir, params)
    else:
        print("Skip diffusion with very high resolution map %f" % params['resolution'])
        diff_trace_map = cur_map_path
    
    print(f"Diffusion process finished! Traced map saved here: {diff_trace_map}")
    
    # Segment this difftrace map
    from modeling.map_utils import segment_map
    diff_new_trace_map = os.path.join(save_path, "diffusion.mrc")
    segment_map(diff_trace_map, diff_new_trace_map, contour=0)
    
    return diff_new_trace_map

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load config with custom function that handles comments
    config = load_json_with_comments(args.config)
    
    # Create params dictionary with command line arguments AND preserve all config keys
    params = config.copy()  # First copy all config keys
    
    # Then override/add specific params from command line
    params.update({
        'F': args.F,
        'resolution': args.resolution,
        'gpu': args.gpu,
        'output': args.output,
        'contour': args.contour,
        'save_intermediate': args.save_intermediate,
    })
    
    # Setup environment and get processed map path
    running_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(running_dir)
    
    # Ensure model path is absolute
    if not os.path.isabs(params['model']['path']):
        params['model']['path'] = os.path.join(running_dir, params['model']['path'])
    
    # Set up environment
    save_path, processed_map_path = set_up_environment(params)
    
    # Run diffusion to trace the backbone
    backbone_map_path = diffusion_trace_map(save_path, processed_map_path, params)
    
    # Copy the result to a more accessible location
    final_output = os.path.join(save_path, "traced_backbone.mrc")
    shutil.copy(backbone_map_path, final_output)
    
    # Print final message
    print(f"✅ Backbone tracing completed successfully!")
    print(f"📊 Original map: {params['F']}")
    print(f"🔍 Resolution: {params['resolution']}Å")
    print(f"🧬 Traced backbone saved to: {final_output}")
    print(f"💡 You can visualize the backbone using Chimera or other molecular visualization tools")

if __name__ == "__main__":
    main()
