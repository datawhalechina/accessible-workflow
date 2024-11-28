# 使用nest_asyncio确保异步稳定性
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 从环境变量中读取值
deep_seek_url = os.getenv("DEEPSEEK_BASE_URL")
deep_seek_api_key = os.getenv("DEEPSEEK_API_KEY")
deep_seek_default_model = os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat")

# 打印环境变量
# print(f"DEEPSEEK_BASE_URL: {deep_seek_url}")
# print(f"DEEPSEEK_API_KEY: {deep_seek_api_key}")
# print(f"DEEPSEEK_DEFAULT_MODEL: {deep_seek_default_model}")

import Agently
agent = (
    Agently.create_agent()
        .set_settings("current_model", "OAIClient")
        .set_settings("model.OAIClient.url", deep_seek_url)
        .set_settings("model.OAIClient.auth", { "api_key": deep_seek_api_key })
        .set_settings("model.OAIClient.options", { "model": deep_seek_default_model })
)

result = agent.input(input("[请输入您的要求]: ")).start()
print("[回复]: ", result)