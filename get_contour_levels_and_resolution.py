import requests
import time
import json
import re
import os
import logging
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emdb_scraper.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 配置 Selenium WebDriver
print("Starting Selenium WebDriver...")
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # 无头模式
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--allow-insecure-localhost")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
print("WebDriver initialized.")

# txt 文件路径
pdb_list_file = "E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler_data/newdateset/pdb_ids.txt"

# 保存结果的 txt 文件路径
output_file_path = "E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler_data/newdateset/contour_levels_and_resolution.txt"
# 保存 JSON 格式结果的文件路径
json_output_file_path = "E:/ZJUT/Research/MrZhouDeepLearning/DiffReaserch/DiffModeler_data/newdateset/contour_levels_and_resolution.json"

# 确保输出目录存在
output_dir = os.path.dirname(output_file_path)
os.makedirs(output_dir, exist_ok=True)

# 读取 PDB ID 列表
with open(pdb_list_file, "r", encoding='utf-8') as f:
    pdb_ids = [line.strip() for line in f]

# 创建结果字典
results_dict = {}

# 从EMDB XML元数据中提取resolution
def extract_resolution_from_xml(emdb_id):
    xml_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdb_id}/header/emd-{emdb_id}-v30.xml"
    result = {"emdb_id": f"EMD-{emdb_id}"}
    
    try:
        response = requests.get(xml_url)
        response.raise_for_status()
        
        # 解析XML
        root = ET.fromstring(response.content)
        
        # 尝试从XML中提取resolution
        resolution_elements = root.findall(".//resolution")
        if resolution_elements:
            for res_element in resolution_elements:
                resolution = res_element.text
                if resolution:
                    result["resolution"] = resolution
                    logging.info(f"Found resolution in XML: {resolution}")
                    break
        
        return result
    except Exception as e:
        logging.error(f"Error extracting resolution from XML for {emdb_id}: {e}")
        return result

# 从EMDB网站获取contour level
def get_contour_level_from_website(emdb_id):
    url = f"https://www.ebi.ac.uk/emdb/{emdb_id}?tab=validation"
    logging.info(f"Accessing URL for contour level: {url}")
    print(f"Accessing URL for contour level: {url}")
    
    retry_attempts = 3
    while retry_attempts > 0:
        try:
            # 使用 WebDriver 打开 URL
            driver.get(url)
            
            # 等待页面加载并查找 contour level 的信息
            print(f"Waiting for contour level element for EMDB ID: {emdb_id}")
            logging.info(f"Waiting for contour level element for EMDB ID: {emdb_id}")
            
            # 尝试使用 XPath 查找 contour level
            contour_element = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="Content"]/div/div/div[2]/div[2]/div/div'))
            )
            contour_text = contour_element.text.strip()
            
            # 从文本中提取 contour level 值
            contour_match = re.search(r"[Cc]ontour\s+[Ll]evel:?\s*([\d\.]+)", contour_text)
            if contour_match:
                contour_level = contour_match.group(1)
                print(f"EMDB ID: {emdb_id}, Contour Level: {contour_level}")
                logging.info(f"EMDB ID: {emdb_id}, Contour Level: {contour_level}")
                return contour_level
            
            # 如果没有通过正则表达式找到，检查文本是否直接就是一个数字
            if re.match(r"^[\d\.]+$", contour_text):
                contour_level = contour_text
                print(f"EMDB ID: {emdb_id}, Contour Level (direct value): {contour_level}")
                logging.info(f"EMDB ID: {emdb_id}, Contour Level (direct value): {contour_level}")
                return contour_level
            
            # 如果文本包含数字，尝试提取第一个数字作为 contour level
            number_match = re.search(r"([\d\.]+)", contour_text)
            if number_match:
                contour_level = number_match.group(1)
                print(f"EMDB ID: {emdb_id}, Contour Level (extracted number): {contour_level}")
                logging.info(f"EMDB ID: {emdb_id}, Contour Level (extracted number): {contour_level}")
                return contour_level
            
            # 如果以上方法都无法提取，记录整个文本并返回空
            logging.info(f"Contour text found but couldn't extract level: {contour_text}")
            return None
            
        except Exception as e:
            retry_attempts -= 1
            print(f"Error fetching contour level for EMDB ID: {emdb_id}, retries left: {retry_attempts}")
            logging.error(f"Error fetching contour level for EMDB ID: {emdb_id}, retries left: {retry_attempts}")
            logging.error(str(e))
            
            if retry_attempts == 0:
                print(f"Failed to fetch contour level for EMDB ID: {emdb_id} after multiple retries.")
                return None
            else:
                time.sleep(2)  # 等待几秒再尝试

# 遍历 PDB ID 列表并获取对应的 contour level 和 resolution 值
try:
    # 清空输出文件，确保是空文件开始
    with open(output_file_path, "w", encoding='utf-8') as f:
        f.write("")  # 清空文件
        
    for pdb_id in pdb_ids:
        print(f"Processing PDB ID: {pdb_id}")
        logging.info(f"Processing PDB ID: {pdb_id}")
        rcsb_api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        
        try:
            # 获取 PDB 数据
            rcsb_response = requests.get(rcsb_api_url)
            rcsb_response.raise_for_status()
            rcsb_data = rcsb_response.json()
            
            print(f"Fetched data for PDB ID: {pdb_id}")
            logging.info(f"Fetched data for PDB ID: {pdb_id}")
            emdb_ids = rcsb_data.get("rcsb_entry_container_identifiers", {}).get("emdb_ids", [])
            
            # 初始化 PDB 结果字典
            results_dict[pdb_id] = {"emdb_data": []}
            
            # 尝试从 PDB 数据中获取分辨率（对于 X-ray 晶体结构）
            try:
                resolution = rcsb_data.get("rcsb_entry_info", {}).get("resolution_combined", [])
                if resolution:
                    results_dict[pdb_id]["pdb_resolution"] = resolution[0]
                    print(f"PDB Resolution: {resolution[0]} A")
            except Exception as e:
                print(f"Error getting PDB resolution: {e}")
            
            if emdb_ids:
                for emdb_id in emdb_ids:
                    # 从原始EMDB ID中提取数字部分
                    emdb_num = emdb_id.replace("EMD-", "")
                    print(f"Processing EMDB ID: {emdb_id}")
                    logging.info(f"Processing EMDB ID: {emdb_id}")
                    
                    # 创建结果字典
                    emdb_result = {"emdb_id": emdb_id}
                    
                    # 从XML获取resolution
                    xml_result = extract_resolution_from_xml(emdb_num)
                    if "resolution" in xml_result:
                        emdb_result["resolution"] = xml_result["resolution"]
                    
                    # 从网站获取contour level
                    contour_level = get_contour_level_from_website(emdb_id)
                    if contour_level:
                        emdb_result["contour_level"] = contour_level
                    
                    # 将 EMDB 结果添加到 PDB 结果中
                    results_dict[pdb_id]["emdb_data"].append(emdb_result)
                    
                    # 保存 contour level 和 resolution 值到文本文件
                    with open(output_file_path, "a", encoding='utf-8') as output_file:
                        contour_str = emdb_result.get("contour_level", "N/A")
                        resolution_str = emdb_result.get("resolution", "N/A")
                        output_file.write(f"{pdb_id}: {emdb_id}, Contour Level: {contour_str}, Resolution: {resolution_str} A\n")
                    
                    # 延迟以避免触发反爬虫机制
                    time.sleep(1)
            else:
                print(f"No EMDB ID found for PDB ID: {pdb_id}")
                
                # 仍然保存 PDB 数据（如果有分辨率）
                with open(output_file_path, "a", encoding='utf-8') as output_file:
                    pdb_resolution = results_dict[pdb_id].get("pdb_resolution", "N/A")
                    output_file.write(f"{pdb_id}: No EMDB ID, PDB Resolution: {pdb_resolution} A\n")
                
        except requests.RequestException as e:
            print(f"Error fetching data for PDB ID: {pdb_id}")
            print(e)
    
    # 保存 JSON 格式的完整结果
    with open(json_output_file_path, "w", encoding='utf-8') as json_file:
        json.dump(results_dict, json_file, indent=2)
    print(f"Results saved to {output_file_path} and {json_output_file_path}")

except Exception as e:
    logging.error(f"Unexpected error in main routine: {e}")
    print(f"Unexpected error: {e}")
finally:
    # 关闭浏览器驱动
    print("Closing Selenium WebDriver...")
    driver.quit()
