## 本教程会带给你

1. 了解工作流对大模型进行高质量工作的辅助意义
    
2. 学会复现吴恩达博士的翻译工作流开源项目
    
3. 了解构成大模型工作流系统的关键元素
    
4. 学会搭建一个更复杂的业务场景工作流


代码能力要求：**中**，AI/数学基础要求：**低**

1. 有编程基础的同学
    
    1. 能够自己动手实现一套复杂的大模型工作流
        
2. 没有编程基础的同学
    
    1. 可以关注和理解工作流对于大模型应用的意义、关键元素和构建思路
        
    2. 不需要复杂编程知识，可以尝试复现简单的翻译工作流
        

  

## 基础环境

```Python
# 安装所需要使用的包
!pip install openai langgraph Agently==3.3.4.5 mermaid-python nest_asyncio

# 因为本课使用的langgraph可能需要依赖langchain 0.2.10版本，但其他课件依赖langchain 0.1.20版本
# 请学习完本课之后对langchain进行降级，以免在其他课程出现运行错误
#!pip install langchain==0.1.20
#!pip install langchain-openai==0.1.6
#!pip install langchain-community==0.0.38
```

```Python
# 使用nest_asyncio确保异步稳定性
import nest_asyncio
nest_asyncio.apply()
```

```Python
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 从环境变量中读取值
deep_seek_url = os.getenv("DEEPSEEK_BASE_URL")
deep_seek_api_key = os.getenv("DEEPSEEK_API_KEY")
deep_seek_default_model = os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat")

# 打印环境变量
#print(f"DEEPSEEK_BASE_URL: {deep_seek_url}")
#print(f"DEEPSEEK_API_KEY: {deep_seek_api_key}")
#print(f"DEEPSEEK_DEFAULT_MODEL: {deep_seek_default_model}")
```

  

试一试

# 直接请求模型的效果：

```Python
#试一试
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
```