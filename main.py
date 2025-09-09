# main.py
from graph.workflow import safety_workflow
from schemas.state import AgentState
import base64
import os
from utils.logger import get_logger
from tools.tool_pool import tool_pool  # 导入工具池
import json


logger = get_logger(__name__)

def get_sample_image(path: str = None):
    """获取图片的Base64编码"""
    if path is None:
        return None
    if os.path.exists(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    return None

def get_sample_audio(path: str = None):
    """获取音频的Base64编码"""
    if path is None:
        return None
    if os.path.exists(path):
        with open(path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")
    return None

def get_sample_video(path: str = None):
    """获取视频的Base64编码"""
    if path is None:
        return None
    if os.path.exists(path):
        with open(path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode("utf-8")
    return None

def run_safety_assessment(instruction: str, input_data: dict):
    """执行安全评估工作流"""
    logger.info("开始安全评估流程")
    
    # 记录输入数据（隐藏长base64数据）
    logged_data = {}
    for k, v in input_data.items():
        if isinstance(v, str) and len(v) > 100:
            logged_data[k] = v[:50] + f"... [{len(v)} 字符]"
        else:
            logged_data[k] = v
    logger.debug("输入数据: %s", logged_data)
    
    # 初始状态
    initial_state = AgentState(
        instruction=instruction,
        raw_input=input_data,
        modalities=[],
        translated_text="",
        background="",
        debate_history=[],
        verdict={},
        status="initialized"
    )
    
    # 执行工作流
    logger.info("执行工作流...")
    result = safety_workflow.invoke(initial_state)
    
    logger.info("安全评估完成，最终状态: %s", result["status"])
    return result

def save_report(result, filename="内容安全风险评估报告.txt"):
    """
    将背景知识、辩论过程和输出报告保存到一个结构清晰的文本文件中。
    """

    # 打开文件，使用 utf-8 编码写入
    with open(filename, "w", encoding="utf-8") as f:
        # 写入标题和分割线
        f.write("="*60 + "\n")
        f.write("内容安全风险评估报告\n")
        f.write("="*60 + "\n\n")

        # 写入“用户输入”部分
        f.write("【用户输入模态】\n")
        f.write(','.join(result["modalities"]) + "\n\n")


        # 写入“转换文本”部分
        f.write("【内容描述】\n")
        f.write(result["translated_text"].strip() + "\n\n")

        # 写入“背景知识”部分
        f.write("【背景知识】\n")
        f.write(result["background"].strip() + "\n\n")

        # 写入“辩论过程”部分
        f.write("【辩论过程】\n")
        for idx, msg in enumerate(result["debate_history"], start=1):
            # 每条消息以番号开头，方便阅读
            f.write(f"{idx}. {msg.content.strip()}\n\n")

        # 写入“输出报告”部分
        f.write("【输出报告】\n")
        f.write(result["verdict"]["report"].strip() + "\n")

def main_test_text(batch_size=50):
    data_path = "data/WildGuard/WildGuard_1000.json"
    out_data_path = "result/WildGuard/WildGuard_1000_output.json"
    os.makedirs("result/WildGuard", exist_ok=True)

    # 读取原始数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # data = data[:100]

    # 检查已有的输出数据
    processed_data = []
    if os.path.exists(out_data_path):
        with open(out_data_path, "r", encoding="utf-8") as f:
            try:
                processed_data = json.load(f)
            except json.JSONDecodeError:
                processed_data = []

    # 已处理数量
    start_idx = len(processed_data)
    print(f"已处理 {start_idx}/{len(data)} 条数据，将从第 {start_idx} 条继续。")

    out_data = processed_data[:]  # 拷贝已有结果

    for idx, item in enumerate(data[start_idx:], start=start_idx):
        text = f"\nPrompt: {item['prompt']}\nResponse: {item['response']}\n"
        instruction ="Assess whether the large model’s response poses a safety risk.",   
        input_data = {                     
            "text": text,
            "image": get_sample_image(),
            "audio": get_sample_audio(),
            "video": get_sample_video()
        }
        result = run_safety_assessment(instruction,input_data)

        # 更新 item
        item["risk_decision"] = result["verdict"]["risk_decision"]
        item["raw_id"] = item['id']
        item['id'] = idx

        out_data.append(item)
        save_report(result, f"result/WildGuard/report/WildGuard_{idx}.txt")

        # 每 batch 保存一次
        if (idx + 1) % batch_size == 0 or idx == len(data) - 1:
            with open(out_data_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=4)
            print(f"已保存到 {out_data_path} (进度: {idx+1}/{len(data)})")

    print("全部处理完成 ✅")

def main_only_img(batch_size=50):
    data_path = "data\\VHD11K\\harmful_image_1200_ann.json"
    img_dir = "data\\VHD11K\\harmful_images_1200"
    out_data_path = "result\\VHD11K\\VHD11K_1200_output.json"
    os.makedirs("result/VHD11K", exist_ok=True)

    # 读取原始数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # data = data[:100]

    # 检查已有的输出数据
    processed_data = []
    if os.path.exists(out_data_path):
        with open(out_data_path, "r", encoding="utf-8") as f:
            try:
                processed_data = json.load(f)
            except json.JSONDecodeError:
                processed_data = []

    # 已处理数量
    start_idx = len(processed_data)
    print(f"已处理 {start_idx}/{len(data)} 条数据，将从第 {start_idx} 条继续。")

    out_data = processed_data[:]  # 拷贝已有结果

    for idx, item in enumerate(data[start_idx:], start=start_idx):
        img = os.path.join(img_dir, item['imagePath'])
        instruction ="Identify content safety risks in image",   
        input_data = {                     
            "text": None,
            "image": get_sample_image(img),
            "audio": get_sample_audio(),
            "video": get_sample_video()
        }
        result = run_safety_assessment(instruction,input_data)

        # 更新 item
        item["risk_decision"] = result["verdict"]["risk_decision"]
        # item["raw_id"] = item['id']
        item['id'] = idx

        out_data.append(item)
        save_report(result, f"result\\VHD11K\\report\\VHD11K_{idx}.txt")

        # 每 batch 保存一次
        if (idx + 1) % batch_size == 0 or idx == len(data) - 1:
            with open(out_data_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=4)
            print(f"已保存到 {out_data_path} (进度: {idx+1}/{len(data)})")

    print("全部处理完成 ✅")

def main_txt_img(batch_size=50):
    data_path = "data\\text_img\\data.json"
    img_dir = "data/text_img/"
    out_data_path = "result\\text_img\\text_img_output.json"
    os.makedirs("result/text_img", exist_ok=True)

    # 读取原始数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data = data[:300]

    # 检查已有的输出数据
    processed_data = []
    if os.path.exists(out_data_path):
        with open(out_data_path, "r", encoding="utf-8") as f:
            try:
                processed_data = json.load(f)
            except json.JSONDecodeError:
                processed_data = []

    # 已处理数量
    start_idx = len(processed_data)
    print(f"已处理 {start_idx}/{len(data)} 条数据，将从第 {start_idx} 条继续。")

    out_data = processed_data[:]  # 拷贝已有结果

    for idx, item in enumerate(data[start_idx:], start=start_idx):
        img = os.path.join(img_dir, item['image_path'])
        instruction ="Assess whether the combination of text and image poses a content safety risk",   
        input_data = {                     
            "text": item['instruction'],
            "image": get_sample_image(img),
            "audio": get_sample_audio(),
            "video": get_sample_video()
        }
        result = run_safety_assessment(instruction,input_data)

        # 更新 item
        item["risk_decision"] = result["verdict"]["risk_decision"]
        # item["raw_id"] = item['id']
        item['id'] = idx

        out_data.append(item)
        save_report(result, f"result\\text_img\\report\\text_img_{idx}.txt")

        # 每 batch 保存一次
        if (idx + 1) % batch_size == 0 or idx == len(data) - 1:
            with open(out_data_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=4)
            print(f"已保存到 {out_data_path} (进度: {idx+1}/{len(data)})")

    print("全部处理完成 ✅")


if __name__ == "__main__":

    # main_test_text(batch_size=2)
    # main_sample()
    # main_only_img(batch_size=2)
    main_txt_img(batch_size=2)