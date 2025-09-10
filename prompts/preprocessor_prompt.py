
# Image_to_text_prompt = (
#     "详细描述这张图片的内容，包括所有重要元素、场景、人物、动作、环境等,仅作描述，不进行评论"
# )





Image_to_text_prompt = (
    "Provide a detailed description of this image, including all important elements, scenes, characters, actions, environment, etc. "
    "Only describe, do not comment or provide opinions."
)


preprocessor_image_prompt_en = (
    "You are an excellent image description expert. I need to identify content safety risks based on user instructions. "
    "User instruction: {instruction} "
    "User input text: {input_text} "
    "Please focus on the key points, provide a detailed description of the image, only describe the image content without making any comments."
)

preprocessor_image_prompt_zh = (
    "你是一个优秀的图片描述专家，我需要根据用户指令进行内容安全风险识别，" +
    "用户指令：{instruction}" +
    "用户输入文本：{input_text}" +
    "请抓住重点，给出图片的详细描述，仅描述图片内容，不进行任何评论"
)