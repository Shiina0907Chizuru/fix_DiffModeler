#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import requests
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.mmcifio import MMCIFIO
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="从RCSB PDB下载CIF文件并转换为CryFold可用的格式")
    parser.add_argument("--output_dir", required=True, help="输出目录，用于保存下载的CIF文件")
    parser.add_argument("--protein_list", required=True, help="蛋白质ID列表文件，每行一个")
    parser.add_argument("--only_ca", action="store_true", help="是否只提取CA原子")
    parser.add_argument("--max_workers", type=int, default=5, help="最大并行下载线程数")
    parser.add_argument("--convert_to_point_cif", action="store_true", help="是否将下载的CIF转换为point.cif格式")
    parser.add_argument("--no_proxy", action="store_true", help="禁用代理进行下载")
    parser.add_argument("--download_method", choices=["requests", "curl", "wget", "urllib"], default="requests", 
                        help="下载方法: requests(默认), curl, wget或urllib")
    return parser.parse_args()

def download_with_requests(url, output_file, use_proxy=True):
    """使用requests库下载文件"""
    proxies = None
    if not use_proxy:
        proxies = {
            'http': None,
            'https': None
        }
    
    response = requests.get(url, timeout=30, proxies=proxies)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        return True
    else:
        print(f"下载失败，HTTP状态码: {response.status_code}")
        return False

def download_with_curl(url, output_file):
    """使用curl命令下载文件"""
    try:
        result = subprocess.run(
            ["curl", "-sSL", url, "-o", output_file, "--retry", "3"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"curl下载失败: {e}")
        return False

def download_with_wget(url, output_file):
    """使用wget命令下载文件"""
    try:
        result = subprocess.run(
            ["wget", url, "-O", output_file, "--quiet", "--tries=3"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"wget下载失败: {e}")
        return False

def download_with_urllib(url, output_file):
    """使用urllib下载文件"""
    try:
        import urllib.request
        # 创建无代理的opener
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, output_file)
        return True
    except Exception as e:
        print(f"urllib下载失败: {e}")
        return False

def download_cif(pdb_id, output_dir, max_retries=3, no_proxy=False, download_method="requests"):
    """
    下载指定PDB ID的CIF文件
    
    参数:
    - pdb_id: PDB ID，例如 "7rdr"
    - output_dir: 输出目录
    - max_retries: 最大重试次数
    - no_proxy: 是否禁用代理
    - download_method: 下载方法 (requests, curl, wget, urllib)
    
    返回:
    - 下载的CIF文件路径或None（如果下载失败）
    """
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    
    # 创建蛋白质目录
    protein_dir = os.path.join(output_dir, f"PDB-{pdb_id.upper()}")
    os.makedirs(protein_dir, exist_ok=True)
    
    # 目标文件路径
    output_file = os.path.join(protein_dir, f"{pdb_id}.cif")
    
    # 如果文件已存在且不为空，跳过下载
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"文件已存在: {output_file}，跳过下载")
        return output_file
    
    # 选择下载方法
    download_functions = {
        "requests": lambda: download_with_requests(url, output_file, not no_proxy),
        "curl": lambda: download_with_curl(url, output_file),
        "wget": lambda: download_with_wget(url, output_file),
        "urllib": lambda: download_with_urllib(url, output_file)
    }
    
    # 确保指定的下载方法存在
    if download_method not in download_functions:
        print(f"错误: 不支持的下载方法 '{download_method}'，使用默认方法 'requests'")
        download_method = "requests"
    
    download_fn = download_functions[download_method]
    
    # 尝试下载文件
    for attempt in range(max_retries):
        try:
            print(f"使用 {download_method} 下载 {url} 到 {output_file} (尝试 {attempt+1}/{max_retries})")
            
            # 调用选定的下载函数
            if download_fn():
                print(f"下载成功: {output_file}")
                return output_file
            
        except Exception as e:
            print(f"下载过程中出错: {e}")
        
        # 如果不是最后一次尝试，等待一段时间后重试
        if attempt < max_retries - 1:
            sleep_time = 2 ** attempt  # 指数退避
            print(f"等待 {sleep_time} 秒后重试...")
            time.sleep(sleep_time)
    
    # 如果所有方法都失败，尝试其他下载方法
    if download_method != "requests":
        try:
            print(f"尝试使用备用方法 'urllib' 下载...")
            if download_with_urllib(url, output_file):
                print(f"下载成功: {output_file}")
                return output_file
        except Exception as e:
            print(f"备用下载方法失败: {e}")
    
    print(f"下载 {pdb_id} 失败，已达到最大重试次数")
    return None

def extract_ca_from_cif(cif_file, output_cif, only_ca=True):
    """
    从CIF文件中提取CA原子并保存为新的CIF文件，将所有链的原子合并到一个链中
    残基ID连续从0开始，与CryFold格式一致
    
    参数:
    - cif_file: 输入CIF文件路径
    - output_cif: 输出CIF文件路径
    - only_ca: 是否只提取CA原子
    
    返回:
    - 转换后的CIF文件路径或None（如果转换失败）
    """
    try:
        # 忽略PDB构建警告
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            
            # 解析CIF文件
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('structure', cif_file)
            
            # 提取原子坐标
            atom_coords = []
            atom_names = []
            chain_ids = []  # 记录原始链ID，用于日志
            
            total_atoms = 0
            chain_count = 0
            
            # 处理残基去重
            for model in structure:
                for chain in model:
                    chain_id = chain.get_id()
                    chain_count += 1
                    chain_atom_count = 0
                    
                    # 建立残基字典，处理重复的残基
                    processed_residues = {}
                    for residue in chain:
                        # 跳过非标准残基
                        if residue.id[0] != ' ':
                            continue
                        
                        # 使用残基ID作为键
                        res_key = residue.id[1]
                        if res_key in processed_residues:
                            # 如果残基ID已存在，检查是否包含CA原子
                            existing_has_ca = any(atom.get_name() == "CA" for atom in processed_residues[res_key])
                            current_has_ca = any(atom.get_name() == "CA" for atom in residue)
                            
                            # 如果当前残基有CA而已存在的没有，则替换
                            if current_has_ca and not existing_has_ca:
                                processed_residues[res_key] = residue
                        else:
                            processed_residues[res_key] = residue
                    
                    # 处理过滤后的残基
                    for res_id, residue in sorted(processed_residues.items()):
                        for atom in residue:
                            # 获取原子名称
                            atom_name = atom.get_name()
                            
                            # 如果只提取CA原子，则跳过其他原子
                            if only_ca and atom_name != "CA":
                                continue
                                
                            # 检查原子元素是否为未知元素'X'，如果是则设为'C'
                            if atom.element == 'X' or atom.element == '':
                                atom.element = 'C'  # 将未知元素设为碳原子
                            
                            # 将原子信息添加到列表中
                            atom_coords.append(atom.get_coord())
                            atom_names.append(atom_name)
                            chain_ids.append(chain_id)
                            
                            total_atoms += 1
                            chain_atom_count += 1
                    
                    print(f"从链 {chain_id} 中提取了 {chain_atom_count} 个{'CA' if only_ca else ''}原子")
            
            # 检查是否有提取到原子
            if total_atoms == 0:
                print(f"警告: 从文件 {cif_file} 中未能提取到原子坐标！")
                return None
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_cif), exist_ok=True)
            
            # 创建新的结构，使用与CryFold完全相同的方法
            struct = StructureBuilder()
            struct.init_structure("1")
            struct.init_seg("1")
            struct.init_model("1")
            struct.init_chain("1")  # 使用单一链ID "1"
            
            # 按顺序为每个原子创建一个新的残基，残基ID从0开始连续编号
            for i, (point, atom_name) in enumerate(zip(atom_coords, atom_names)):
                struct.set_line_counter(i)
                # 使用标准的氨基酸命名和编号，从0开始
                struct.init_residue("ALA", " ", i, " ")
                # 设置原子，使用原始原子名称"CA"，元素符号也设置为"CA"
                struct.init_atom(atom_name, point, 0, 1, " ", atom_name, "CA")
            
            # 获取构建的结构
            structure = struct.get_structure()
            
            # 保存为CIF格式
            io = MMCIFIO()
            io.set_structure(structure)
            io.save(output_cif)
            
            print(f"成功提取 {total_atoms} 个{'CA' if only_ca else ''}原子来自 {chain_count} 条链，并合并为单一链保存到: {output_cif}")
            print(f"残基ID从0开始连续编号，符合CryFold格式")
            return output_cif
            
    except Exception as e:
        print(f"处理文件 {cif_file} 时出错: {e}")
        return None

def process_protein(pdb_id, output_dir, convert_to_point_cif=True, only_ca=True, max_retries=3, no_proxy=False, download_method="requests"):
    """
    处理单个蛋白质：下载并可选转换为point.cif
    """
    pdb_id = pdb_id.lower()
    
    # 创建蛋白质目录
    protein_dir = os.path.join(output_dir, f"PDB-{pdb_id.upper()}")
    os.makedirs(protein_dir, exist_ok=True)
    
    # 下载CIF文件
    cif_file = download_cif(pdb_id, output_dir, max_retries, no_proxy, download_method)
    
    if cif_file and convert_to_point_cif:
        # 定义输出point.cif文件路径
        output_cif = os.path.join(protein_dir, f"{pdb_id}_point.cif")
        
        # 转换为point.cif
        return extract_ca_from_cif(cif_file, output_cif, only_ca)
    
    return cif_file

def batch_download_and_process(output_dir, protein_list_file, convert_to_point_cif=True, only_ca=True, max_workers=5, no_proxy=False, download_method="requests"):
    """
    批量下载和处理蛋白质结构
    
    参数:
    - output_dir: 输出目录
    - protein_list_file: 蛋白质ID列表文件，每行一个
    - convert_to_point_cif: 是否将下载的CIF转换为point.cif
    - only_ca: 是否只提取CA原子
    - max_workers: 最大并行下载线程数
    - no_proxy: 是否禁用代理
    - download_method: 下载方法
    """
    # 加载蛋白质ID列表
    with open(protein_list_file, 'r') as f:
        protein_ids = [line.strip() for line in f if line.strip()]
    
    print(f"将处理以下 {len(protein_ids)} 个蛋白质:")
    for pdb_id in protein_ids:
        print(f"  - {pdb_id}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用线程池并行下载和处理
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_pdb = {
            executor.submit(process_protein, pdb_id, output_dir, convert_to_point_cif, only_ca, 3, no_proxy, download_method): pdb_id 
            for pdb_id in protein_ids
        }
        
        # 使用tqdm显示进度
        for future in tqdm(future_to_pdb, total=len(protein_ids), desc="下载和处理进度"):
            pdb_id = future_to_pdb[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                    print(f"成功处理: {pdb_id}")
                else:
                    error_count += 1
                    print(f"处理失败: {pdb_id}")
            except Exception as e:
                error_count += 1
                print(f"处理 {pdb_id} 时发生异常: {e}")
    
    print(f"\n批处理完成！成功: {success_count}, 失败: {error_count}")

def main():
    args = parse_args()
    
    # 打印使用的网络设置
    print(f"下载设置:")
    print(f"  下载方法: {args.download_method}")
    print(f"  {'禁用' if args.no_proxy else '使用'}代理")
    
    batch_download_and_process(
        args.output_dir, 
        args.protein_list, 
        args.convert_to_point_cif,
        args.only_ca,
        args.max_workers,
        args.no_proxy,
        args.download_method
    )

if __name__ == "__main__":
    main()
# python "E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler/trans2cryfold/download_cif_files.py" --output_dir "F:/cifdata" --protein_list "C:/Users/Z/Desktop/2025filtered_proteins.txt" --only_ca --convert_to_point_cif --max_workers 5 --no_proxy --download_method urllib