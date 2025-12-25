"""
下载 Brain Tumor Segmentation Dataset
"""
import requests
import os
import zipfile


def download_file(url, save_path):
    # 发送GET请求下载文件
    print(f"开始下载文件: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 检查是否下载成功

    # 获取文件大小
    file_size = int(response.headers.get('content-length', 0))

    # 写入文件
    with open(save_path, 'wb') as f:
        if file_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # 显示下载进度
                    progress = int(50 * downloaded / file_size)
                    print(f"\r下载进度: [{'=' * progress}{' ' * (50 - progress)}] {downloaded}/{file_size} bytes",
                          end='')
    print("\n下载完成!")


def extract_dataset(zip_path, extract_to):
    """
    解压脑肿瘤图像数据集
    Args:
        zip_path (str): ZIP文件路径
        extract_to (str): 解压目标目录
    """
    # 确保解压目录存在
    os.makedirs(extract_to, exist_ok=True)

    # 解压ZIP文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print(f"正在解压 {zip_path} 到 {extract_to}")
        zip_ref.extractall(extract_to)
        print("解压完成!")


if __name__ == "__main__":
    url = "https://github.com/Zeyi-Lin/UNet-Medical/releases/download/data/Brain.Tumor.Image.DataSet.zip"
    os.makedirs("dataset", exist_ok=True)   # 创建datasets目录
    save_path = "dataset/Brain_Tumor_Image_DataSet.zip"
    extract_directory = "dataset"
    # 检查是否存在Brain.Tumor.Image.DataSet.zip
    if not os.path.exists(save_path):
        download_file(url, save_path)
    # 执行解压
    extract_dataset(save_path, extract_directory)
