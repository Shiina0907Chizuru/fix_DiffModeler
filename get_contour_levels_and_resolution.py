# import requests
# import time
# import json
# import re
# import os
# import logging
# import xml.etree.ElementTree as ET
# from urllib.parse import urlparse
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("emdb_scraper.log", encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )

# # 配置 Selenium WebDriver
# print("Starting Selenium WebDriver...")
# options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # 无头模式
# options.add_argument("--no-sandbox")
# options.add_argument("--disable-dev-shm-usage")
# options.add_argument("--ignore-certificate-errors")
# options.add_argument("--allow-insecure-localhost")
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
# print("WebDriver initialized.")

# # txt 文件路径
# pdb_list_file = "c:/Users/Z/Desktop/20250306.txt"

# # 保存结果的 txt 文件路径
# output_file_path = "c:/Users/Z/Desktop/20250306contour_level&resolution.txt"
# # 保存 JSON 格式结果的文件路径
# json_output_file_path = "c:/Users/Z/Desktop/20250306contour_level&resolution.json"

# # 确保输出目录存在
# output_dir = os.path.dirname(output_file_path)
# os.makedirs(output_dir, exist_ok=True)

# # 读取 PDB ID 列表
# with open(pdb_list_file, "r", encoding='utf-8') as f:
#     pdb_ids = [line.strip() for line in f]

# # 创建结果字典
# results_dict = {}

# # 从EMDB XML元数据中提取resolution
# def extract_resolution_from_xml(emdb_id):
#     xml_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdb_id}/header/emd-{emdb_id}-v30.xml"
#     result = {"emdb_id": f"EMD-{emdb_id}"}
    
#     try:
#         response = requests.get(xml_url)
#         response.raise_for_status()
        
#         # 解析XML
#         root = ET.fromstring(response.content)
        
#         # 尝试从XML中提取resolution
#         resolution_elements = root.findall(".//resolution")
#         if resolution_elements:
#             for res_element in resolution_elements:
#                 resolution = res_element.text
#                 if resolution:
#                     result["resolution"] = resolution
#                     logging.info(f"Found resolution in XML: {resolution}")
#                     break
        
#         return result
#     except Exception as e:
#         logging.error(f"Error extracting resolution from XML for {emdb_id}: {e}")
#         return result

# # 从EMDB网站获取contour level
# def get_contour_level_from_website(emdb_id):
#     url = f"https://www.ebi.ac.uk/emdb/{emdb_id}?tab=validation"
#     logging.info(f"Accessing URL for contour level: {url}")
#     print(f"Accessing URL for contour level: {url}")
    
#     retry_attempts = 3
#     while retry_attempts > 0:
#         try:
#             # 使用 WebDriver 打开 URL
#             driver.get(url)
            
#             # 等待页面加载并查找 contour level 的信息
#             print(f"Waiting for contour level element for EMDB ID: {emdb_id}")
#             logging.info(f"Waiting for contour level element for EMDB ID: {emdb_id}")
            
#             # 尝试使用 XPath 查找 contour level
#             contour_element = WebDriverWait(driver, 30).until(
#                 EC.presence_of_element_located((By.XPATH, '//*[@id="Content"]/div/div/div[2]/div[2]/div/div'))
#             )
#             contour_text = contour_element.text.strip()
            
#             # 从文本中提取 contour level 值
#             contour_match = re.search(r"[Cc]ontour\s+[Ll]evel:?\s*([\d\.]+)", contour_text)
#             if contour_match:
#                 contour_level = contour_match.group(1)
#                 print(f"EMDB ID: {emdb_id}, Contour Level: {contour_level}")
#                 logging.info(f"EMDB ID: {emdb_id}, Contour Level: {contour_level}")
#                 return contour_level
            
#             # 如果没有通过正则表达式找到，检查文本是否直接就是一个数字
#             if re.match(r"^[\d\.]+$", contour_text):
#                 contour_level = contour_text
#                 print(f"EMDB ID: {emdb_id}, Contour Level (direct value): {contour_level}")
#                 logging.info(f"EMDB ID: {emdb_id}, Contour Level (direct value): {contour_level}")
#                 return contour_level
            
#             # 如果文本包含数字，尝试提取第一个数字作为 contour level
#             number_match = re.search(r"([\d\.]+)", contour_text)
#             if number_match:
#                 contour_level = number_match.group(1)
#                 print(f"EMDB ID: {emdb_id}, Contour Level (extracted number): {contour_level}")
#                 logging.info(f"EMDB ID: {emdb_id}, Contour Level (extracted number): {contour_level}")
#                 return contour_level
            
#             # 如果以上方法都无法提取，记录整个文本并返回空
#             logging.info(f"Contour text found but couldn't extract level: {contour_text}")
#             return None
            
#         except Exception as e:
#             retry_attempts -= 1
#             print(f"Error fetching contour level for EMDB ID: {emdb_id}, retries left: {retry_attempts}")
#             logging.error(f"Error fetching contour level for EMDB ID: {emdb_id}, retries left: {retry_attempts}")
#             logging.error(str(e))
            
#             if retry_attempts == 0:
#                 print(f"Failed to fetch contour level for EMDB ID: {emdb_id} after multiple retries.")
#                 return None
#             else:
#                 time.sleep(2)  # 等待几秒再尝试

# # 遍历 PDB ID 列表并获取对应的 contour level 和 resolution 值
# try:
#     # 清空输出文件，确保是空文件开始
#     with open(output_file_path, "w", encoding='utf-8') as f:
#         f.write("")  # 清空文件
        
#     for pdb_id in pdb_ids:
#         print(f"Processing PDB ID: {pdb_id}")
#         logging.info(f"Processing PDB ID: {pdb_id}")
#         rcsb_api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        
#         try:
#             # 获取 PDB 数据
#             rcsb_response = requests.get(rcsb_api_url)
#             rcsb_response.raise_for_status()
#             rcsb_data = rcsb_response.json()
            
#             print(f"Fetched data for PDB ID: {pdb_id}")
#             logging.info(f"Fetched data for PDB ID: {pdb_id}")
#             emdb_ids = rcsb_data.get("rcsb_entry_container_identifiers", {}).get("emdb_ids", [])
            
#             # 初始化 PDB 结果字典
#             results_dict[pdb_id] = {"emdb_data": []}
            
#             # 尝试从 PDB 数据中获取分辨率（对于 X-ray 晶体结构）
#             try:
#                 resolution = rcsb_data.get("rcsb_entry_info", {}).get("resolution_combined", [])
#                 if resolution:
#                     results_dict[pdb_id]["pdb_resolution"] = resolution[0]
#                     print(f"PDB Resolution: {resolution[0]} A")
#             except Exception as e:
#                 print(f"Error getting PDB resolution: {e}")
            
#             if emdb_ids:
#                 for emdb_id in emdb_ids:
#                     # 从原始EMDB ID中提取数字部分
#                     emdb_num = emdb_id.replace("EMD-", "")
#                     print(f"Processing EMDB ID: {emdb_id}")
#                     logging.info(f"Processing EMDB ID: {emdb_id}")
                    
#                     # 创建结果字典
#                     emdb_result = {"emdb_id": emdb_id}
                    
#                     # 从XML获取resolution
#                     xml_result = extract_resolution_from_xml(emdb_num)
#                     if "resolution" in xml_result:
#                         emdb_result["resolution"] = xml_result["resolution"]
                    
#                     # 从网站获取contour level
#                     contour_level = get_contour_level_from_website(emdb_id)
#                     if contour_level:
#                         emdb_result["contour_level"] = contour_level
                    
#                     # 将 EMDB 结果添加到 PDB 结果中
#                     results_dict[pdb_id]["emdb_data"].append(emdb_result)
                    
#                     # 保存 contour level 和 resolution 值到文本文件
#                     contour_str = emdb_result.get("contour_level", "N/A")
#                     resolution_str = emdb_result.get("resolution", "N/A")
#                     with open(output_file_path, "a", encoding='utf-8') as output_file:
#                         output_file.write(f"{pdb_id}: {emdb_id}, Contour Level: {contour_str}, Resolution: {resolution_str} A\n")
                    
#                     # 延迟以避免触发反爬虫机制
#                     time.sleep(1)
#             else:
#                 print(f"No EMDB ID found for PDB ID: {pdb_id}")
                
#                 # 仍然保存 PDB 数据（如果有分辨率）
#                 with open(output_file_path, "a", encoding='utf-8') as output_file:
#                     pdb_resolution = results_dict[pdb_id].get("pdb_resolution", "N/A")
#                     output_file.write(f"{pdb_id}: No EMDB ID, PDB Resolution: {pdb_resolution} A\n")
                
#         except requests.RequestException as e:
#             print(f"Error fetching data for PDB ID: {pdb_id}")
#             print(e)
    
#     # 保存 JSON 格式的完整结果
#     with open(json_output_file_path, "w", encoding='utf-8') as json_file:
#         json.dump(results_dict, json_file, indent=2)
#     print(f"Results saved to {output_file_path} and {json_output_file_path}")

# except Exception as e:
#     logging.error(f"Unexpected error in main routine: {e}")
#     print(f"Unexpected error: {e}")
# finally:
#     # 关闭浏览器驱动
#     print("Closing Selenium WebDriver...")
#     driver.quit()




# 有保险的爬虫
import requests
import time
import json
import re
import os
import logging
import xml.etree.ElementTree as ET
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("contour_levels_fetcher.log"),
        logging.StreamHandler()
    ]
)

# 配置 Chrome WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # 无头模式
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")

try:
    driver = webdriver.Chrome(options=chrome_options)
    logging.info("WebDriver initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize WebDriver: {e}")
    raise

print("WebDriver initialized.")

# txt 文件路径
pdb_list_file = "c:/Users/Z/Desktop/20250315.txt"

# 保存结果的 txt 文件路径
output_file_path = "c:/Users/Z/Desktop/20250315contour_level&resolution.txt"
# 保存 JSON 格式结果的文件路径
json_output_file_path = "c:/Users/Z/Desktop/20250315contour_level&resolution.json"
# 临时文件路径，用于存储处理中的结果
temp_output_file_path = "c:/Users/Z/Desktop/20250315contour_level&resolution_temp.txt"
# 保存没有找到信息的蛋白质
missing_info_file_path = "c:/Users/Z/Desktop/20250306missing_info.txt"

# 确保输出目录存在
output_dir = os.path.dirname(output_file_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取 PDB ID 列表
with open(pdb_list_file, "r") as f:
    pdb_ids = [line.strip() for line in f]

# 创建结果字典
results_dict = {}

# 跟踪没有找到信息的蛋白质
missing_info_proteins = []

# 读取已有的结果文件（如果存在）
existing_results = {}
if os.path.exists(output_file_path):
    try:
        with open(output_file_path, "r", encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    pdb_part = parts[0].split(':')[0].strip()
                    has_contour = "Contour Level: N/A" not in line
                    has_resolution = "Resolution: N/A" not in line
                    
                    if pdb_part not in existing_results:
                        existing_results[pdb_part] = {"has_contour": has_contour, "has_resolution": has_resolution}
        
        print(f"Loaded existing results for {len(existing_results)} PDB IDs")
    except Exception as e:
        print(f"Error reading existing results: {e}")

# 保存已处理结果到 JSON 文件
if os.path.exists(json_output_file_path):
    try:
        with open(json_output_file_path, "r", encoding='utf-8') as f:
            results_dict = json.load(f)
        print(f"Loaded existing JSON results with {len(results_dict)} entries")
    except Exception as e:
        print(f"Error loading existing JSON file: {e}")
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

# 备选方法：从EMDB网站直接提取分辨率
def get_resolution_from_website(emdb_id):
    url = f"https://www.ebi.ac.uk/emdb/{emdb_id}"
    logging.info(f"Accessing URL for resolution: {url}")
    print(f"Accessing URL for resolution (backup method): {url}")
    
    retry_attempts = 3
    while retry_attempts > 0:
        try:
            driver.get(url)
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="Content"]'))
            )
            
            # 尝试不同的XPath定位分辨率信息
            resolution_xpath_options = [
                '//*[contains(text(), "Resolution") and contains(text(), "Å")]',
                '//*[contains(text(), "resolution") and contains(text(), "Å")]',
                '//*[contains(text(), "Resolution")]',
                '//*[contains(text(), "resolution")]'
            ]
            
            for xpath in resolution_xpath_options:
                try:
                    elements = driver.find_elements(By.XPATH, xpath)
                    for element in elements:
                        text = element.text
                        resolution_match = re.search(r"([\d\.]+)\s*Å", text)
                        if resolution_match:
                            resolution = resolution_match.group(1)
                            print(f"EMDB ID: {emdb_id}, Resolution: {resolution} Å (from website)")
                            logging.info(f"EMDB ID: {emdb_id}, Resolution: {resolution} Å (from website)")
                            return resolution
                except:
                    continue
            
            logging.info(f"Could not find resolution for {emdb_id} on website")
            return None
            
        except Exception as e:
            retry_attempts -= 1
            print(f"Error fetching resolution for EMDB ID: {emdb_id}, retries left: {retry_attempts}")
            logging.error(f"Error fetching resolution for EMDB ID: {emdb_id}, retries left: {retry_attempts}")
            logging.error(str(e))
            
            if retry_attempts == 0:
                print(f"Failed to fetch resolution for EMDB ID: {emdb_id} after multiple retries.")
                return None
            else:
                time.sleep(2)

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

# 备选方式：从详情页面获取 contour level
def get_contour_level_from_details(emdb_id):
    url = f"https://www.ebi.ac.uk/emdb/{emdb_id}"
    logging.info(f"Accessing details URL for contour level (backup method): {url}")
    print(f"Accessing details URL for contour level (backup method): {url}")
    
    retry_attempts = 3
    while retry_attempts > 0:
        try:
            driver.get(url)
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="Content"]'))
            )
            
            # 获取页面内容查找 contour 相关文本
            page_text = driver.page_source
            
            # 从页面文本中提取 contour level
            contour_patterns = [
                r"[Cc]ontour\s+[Ll]evel\s*[=:]\s*([\d\.]+)",
                r"[Cc]ontour\s+[Ll]evel\s+of\s+([\d\.]+)",
                r"[Cc]ontour\s+[Ll]evel.*?([\d\.]+)",
                r"[Cc]ontour.*?([\d\.]+)"
            ]
            
            for pattern in contour_patterns:
                contour_match = re.search(pattern, page_text)
                if contour_match:
                    contour_level = contour_match.group(1)
                    print(f"EMDB ID: {emdb_id}, Contour Level: {contour_level} (from details)")
                    logging.info(f"EMDB ID: {emdb_id}, Contour Level: {contour_level} (from details)")
                    return contour_level
            
            print(f"No contour level found in details for EMDB ID: {emdb_id}")
            return None
            
        except Exception as e:
            retry_attempts -= 1
            print(f"Error fetching contour level from details for EMDB ID: {emdb_id}, retries left: {retry_attempts}")
            logging.error(f"Error fetching contour details for EMDB ID: {emdb_id}, retries left: {retry_attempts}")
            logging.error(str(e))
            
            if retry_attempts == 0:
                return None
            else:
                time.sleep(2)

# 使用临时文件存储处理结果
valid_entries = []

# 遍历 PDB ID 列表并获取对应的 contour level 和 resolution 值
try:
    # 只有在临时文件不存在时才创建
    if not os.path.exists(temp_output_file_path):
        with open(temp_output_file_path, "w", encoding='utf-8') as f:
            f.write("")  # 创建空文件
        
    for pdb_id in pdb_ids:
        # 检查是否已经处理过且有完整信息
        if pdb_id in existing_results and existing_results[pdb_id].get("has_contour", False) and existing_results[pdb_id].get("has_resolution", False):
            print(f"Skipping PDB ID: {pdb_id} - already has complete information")
            continue
        
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
            if pdb_id not in results_dict:
                results_dict[pdb_id] = {"emdb_data": []}
            
            # 尝试从 PDB 数据中获取分辨率（对于 X-ray 晶体结构）
            try:
                resolution = rcsb_data.get("rcsb_entry_info", {}).get("resolution_combined", [])
                if resolution:
                    results_dict[pdb_id]["pdb_resolution"] = resolution[0]
                    print(f"PDB Resolution: {resolution[0]} A")
            except Exception as e:
                print(f"Error getting PDB resolution: {e}")
            
            found_valid_info = False  # 标记是否找到有效信息
            
            if emdb_ids:
                for emdb_id in emdb_ids:
                    # 从原始EMDB ID中提取数字部分
                    emdb_num = emdb_id.replace("EMD-", "")
                    print(f"Processing EMDB ID: {emdb_id}")
                    logging.info(f"Processing EMDB ID: {emdb_id}")
                    
                    # 创建结果字典
                    emdb_result = {"emdb_id": emdb_id}
                    
                    # 1. 尝试从XML获取resolution
                    xml_result = extract_resolution_from_xml(emdb_num)
                    if "resolution" in xml_result:
                        emdb_result["resolution"] = xml_result["resolution"]
                    else:
                        # 2. 备选：如果XML没有resolution，尝试从网站获取
                        resolution_from_web = get_resolution_from_website(emdb_id)
                        if resolution_from_web:
                            emdb_result["resolution"] = resolution_from_web
                    
                    # 1. 尝试从validation页面获取contour level
                    contour_level = get_contour_level_from_website(emdb_id)
                    if contour_level:
                        emdb_result["contour_level"] = contour_level
                    else:
                        # 2. 备选：如果validation页面没有contour level，尝试从详情页获取
                        contour_from_details = get_contour_level_from_details(emdb_id)
                        if contour_from_details:
                            emdb_result["contour_level"] = contour_from_details
                    
                    # 将 EMDB 结果添加到 PDB 结果中
                    # 检查是否已有此EMDB ID的数据
                    emdb_exists = False
                    for idx, existing_emdb in enumerate(results_dict[pdb_id]["emdb_data"]):
                        if existing_emdb.get("emdb_id") == emdb_id:
                            # 更新现有数据
                            results_dict[pdb_id]["emdb_data"][idx].update(emdb_result)
                            emdb_exists = True
                            break
                    
                    if not emdb_exists:
                        results_dict[pdb_id]["emdb_data"].append(emdb_result)
                    
                    # 保存 contour level 和 resolution 值到临时文件，并保存所有记录
                    contour_str = emdb_result.get("contour_level", "N/A")
                    resolution_str = emdb_result.get("resolution", "N/A")
                    found_valid_info = True
                    entry = f"{pdb_id}: {emdb_id}, Contour Level: {contour_str}, Resolution: {resolution_str} A"
                    valid_entries.append(entry)
                    with open(temp_output_file_path, "a", encoding='utf-8') as output_file:
                        output_file.write(entry + "\n")
                    
                    # 延迟以避免触发反爬虫机制
                    time.sleep(1)
            else:
                print(f"No EMDB ID found for PDB ID: {pdb_id}")
                
                # 尝试直接从PDB搜索EMDB数据库
                backup_emdb_search_url = f"https://www.ebi.ac.uk/emdb/search/{pdb_id}"
                try:
                    print(f"Attempting backup search for EMDB IDs related to {pdb_id}")
                    driver.get(backup_emdb_search_url)
                    time.sleep(2)  # 等待搜索结果
                    
                    # 尝试找到搜索结果中的EMDB ID
                    search_results = driver.find_elements(By.XPATH, "//*[contains(text(), 'EMD-')]")
                    found_emdb_ids = []
                    
                    for result in search_results:
                        text = result.text
                        emdb_matches = re.findall(r"EMD-\d+", text)
                        found_emdb_ids.extend(emdb_matches)
                    
                    if found_emdb_ids:
                        unique_emdb_ids = list(set(found_emdb_ids))
                        print(f"Found potential EMDB IDs via search: {unique_emdb_ids}")
                        
                        for emdb_id in unique_emdb_ids[:1]:  # 仅处理第一个找到的EMDB ID
                            emdb_num = emdb_id.replace("EMD-", "")
                            
                            # 创建结果字典
                            emdb_result = {"emdb_id": emdb_id}
                            
                            # 获取resolution和contour level
                            xml_result = extract_resolution_from_xml(emdb_num)
                            if "resolution" in xml_result:
                                emdb_result["resolution"] = xml_result["resolution"]
                            else:
                                resolution_from_web = get_resolution_from_website(emdb_id)
                                if resolution_from_web:
                                    emdb_result["resolution"] = resolution_from_web
                            
                            contour_level = get_contour_level_from_website(emdb_id)
                            if contour_level:
                                emdb_result["contour_level"] = contour_level
                            else:
                                contour_from_details = get_contour_level_from_details(emdb_id)
                                if contour_from_details:
                                    emdb_result["contour_level"] = contour_from_details
                            
                            # 将结果添加到PDB结果中
                            results_dict[pdb_id]["emdb_data"].append(emdb_result)
                            
                            # 保存到临时文件，并保存所有记录
                            contour_str = emdb_result.get("contour_level", "N/A")
                            resolution_str = emdb_result.get("resolution", "N/A")
                            found_valid_info = True
                            entry = f"{pdb_id}: {emdb_id} (via search), Contour Level: {contour_str}, Resolution: {resolution_str} A"
                            valid_entries.append(entry)
                            with open(temp_output_file_path, "a", encoding='utf-8') as output_file:
                                output_file.write(entry + "\n")
                    else:
                        # 保存PDB分辨率记录
                        pdb_resolution = results_dict[pdb_id].get("pdb_resolution", "N/A")
                        found_valid_info = True
                        entry = f"{pdb_id}: No EMDB ID, PDB Resolution: {pdb_resolution} A"
                        valid_entries.append(entry)
                        with open(temp_output_file_path, "a", encoding='utf-8') as output_file:
                            output_file.write(entry + "\n")
                
                except Exception as e:
                    print(f"Error in backup EMDB search for {pdb_id}: {e}")
                    # 保存PDB分辨率记录
                    pdb_resolution = results_dict[pdb_id].get("pdb_resolution", "N/A")
                    found_valid_info = True
                    entry = f"{pdb_id}: No EMDB ID, PDB Resolution: {pdb_resolution} A"
                    valid_entries.append(entry)
                    with open(temp_output_file_path, "a", encoding='utf-8') as output_file:
                        output_file.write(entry + "\n")
            
            # 如果没有找到有效信息，则将PDB ID添加到没有找到信息的列表中
            if not found_valid_info:
                missing_info_proteins.append(pdb_id)
                
            # 每处理几个PDB ID就保存一次JSON结果，防止意外中断导致数据丢失
            if pdb_ids.index(pdb_id) % 5 == 0:
                with open(json_output_file_path, "w", encoding='utf-8') as json_file:
                    json.dump(results_dict, json_file, indent=2)
                print(f"Intermediate results saved to {json_output_file_path}")
                
        except requests.RequestException as e:
            print(f"Error fetching data for PDB ID: {pdb_id}")
            print(e)
            # 将出错的PDB ID添加到没有找到信息的列表中
            missing_info_proteins.append(pdb_id)
    
    # 保存 JSON 格式的完整结果
    with open(json_output_file_path, "w", encoding='utf-8') as json_file:
        json.dump(results_dict, json_file, indent=2)
    
    # 按蛋白质ID排序并写入最终文件
    print("Sorting results by protein ID...")
    sorted_entries = sorted(valid_entries, key=lambda x: x.split(':')[0])
    
    with open(output_file_path, "w", encoding='utf-8') as f:
        for entry in sorted_entries:
            f.write(entry + "\n")
    
    # 保存没有找到信息的蛋白质到文件
    with open(missing_info_file_path, "w", encoding='utf-8') as f:
        for pdb_id in sorted(missing_info_proteins):
            f.write(f"{pdb_id}\n")
    
    # 输出排序后的结果到终端
    print("\n===== 蛋白质相关信息（按ID排序）=====")
    for entry in sorted_entries:
        print(entry)
    
    print(f"\nTotal valid proteins with resolution: {len(sorted_entries)}")
    
    # 输出没有找到信息的蛋白质到终端
    print("\n===== 没有找到分辨率信息的蛋白质 =====")
    for pdb_id in sorted(missing_info_proteins):
        print(pdb_id)
    
    print(f"\nTotal proteins without resolution: {len(missing_info_proteins)}")
    print(f"Results saved to {output_file_path} and {json_output_file_path}")
    print(f"Missing information proteins saved to {missing_info_file_path}")

except Exception as e:
    logging.error(f"Unexpected error in main routine: {e}")
    print(f"Unexpected error: {e}")
finally:
    # 关闭浏览器驱动
    print("Closing Selenium WebDriver...")
    driver.quit()