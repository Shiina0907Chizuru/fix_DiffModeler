import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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
pdb_list_file = "D:/桌面/test/pdb.txt"

# 保存 contour level 值的 txt 文件路径
output_file_path = "contour_levels1.txt"

# 读取 PDB ID 列表
with open(pdb_list_file, "r") as f:
    pdb_ids = [line.strip() for line in f]

# 遍历 PDB ID 列表并获取对应的 contour level 值
try:
    for pdb_id in pdb_ids:
        print(f"Processing PDB ID: {pdb_id}")
        rcsb_api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

        try:
            rcsb_response = requests.get(rcsb_api_url)
            rcsb_response.raise_for_status()
            rcsb_data = rcsb_response.json()

            print(f"Fetched data for PDB ID: {pdb_id}")
            emdb_ids = rcsb_data.get("rcsb_entry_container_identifiers", {}).get("emdb_ids", [])

            if emdb_ids:
                for emdb_id in emdb_ids:
                    url = f"https://www.ebi.ac.uk/emdb/{emdb_id}?tab=validation"
                    print(f"Accessing URL: {url}")

                    retry_attempts = 3
                    while retry_attempts > 0:
                        try:
                            # 使用 WebDriver 打开 URL
                            driver.get(url)

                            # 等待页面加载并查找 contour level 的信息
                            print(f"Waiting for contour level element for EMDB ID: {emdb_id}")
                            contour_element = WebDriverWait(driver, 30).until(
                                EC.presence_of_element_located((By.XPATH, '//*[@id="Content"]/div/div/div[2]/div[2]/div/div'))
                            )
                            contour_level = contour_element.text.strip()
                            print(f"EMDB ID: {emdb_id}, Contour Level: {contour_level}")

                            # 保存 contour level 值到文件
                            with open(output_file_path, "a") as output_file:
                                output_file.write(f"{pdb_id}: {contour_level}\n")

                            break  # 成功获取 contour level，跳出重试循环

                        except Exception as e:
                            retry_attempts -= 1
                            print(f"Error fetching data for EMDB ID: {emdb_id}, retries left: {retry_attempts}")
                            print(e)

                            if retry_attempts == 0:
                                print(f"Failed to fetch data for EMDB ID: {emdb_id} after multiple retries.")
                            else:
                                time.sleep(2)  # 等待几秒再尝试

                    # 延迟以避免触发反爬虫机制
                    time.sleep(1)
            else:
                print(f"No EMDB ID found for PDB ID: {pdb_id}")

        except requests.RequestException as e:
            print(f"Error fetching EMDB ID for PDB ID: {pdb_id}")
            print(e)

finally:
    # 关闭浏览器驱动
    print("Closing Selenium WebDriver...")
    driver.quit()
