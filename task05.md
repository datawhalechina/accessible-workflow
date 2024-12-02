## **使用LangGraph和Agently** **Workflow****分别复现这个****工作流**

## **LangGraph**

LangGraph手册：[https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)

```Python
import openai
from ENV import deep_seek_url, deep_seek_api_key, deep_seek_default_model
from langgraph.graph import StateGraph, START, END
import os

# 模型请求准备
client = openai.OpenAI(
    api_key = deep_seek_api_key,
    base_url =deep_seek_url
)
default_model = deep_seek_default_model

def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = default_model,
    temperature: float = 0.3,
    json_mode: bool = False,
):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        top_p=1,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

# 定义传递的信息结构
from typing import TypedDict, Optional
class State(TypedDict):
    source_lang: str
    target_lang: str
    source_text: str
    country: Optional[str] = None
    translation_1: Optional[str] = None
    reflection: Optional[str] = None
    translation_2: Optional[str] = None

# 创建一个工作流对象
workflow = StateGraph(State)

# 定义三个工作块
"""
获取state中的信息：state.get("key_name")
更新state中的信息：return { "key_name": new_value }
"""
def initial_translation(state):
    source_lang = state.get("source_lang")
    target_lang = state.get("target_lang")
    source_text = state.get("source_text")

    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

    prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""

    translation = get_completion(prompt, system_message=system_message)

    print("[初次翻译结果]: \n", translation)

    return { "translation_1": translation }

def reflect_on_translation(state):
    source_lang = state.get("source_lang")
    target_lang = state.get("target_lang")
    source_text = state.get("source_text")
    country = state.get("country") or ""
    translation_1 = state.get("translation_1")
    
    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    additional_rule = (
        f"The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}."
        if country != ""
        else ""
    )
    
    prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
{additional_rule}

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    reflection = get_completion(prompt, system_message=system_message)

    print("[初次翻译结果]: \n", reflection)

    return { "reflection": reflection }

def improve_translation(state):
    source_lang = state.get("source_lang")
    target_lang = state.get("target_lang")
    source_text = state.get("source_text")
    translation_1 = state.get("translation_1")
    reflection = state.get("reflection")
    
    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."

    prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""

    translation_2 = get_completion(prompt, system_message)

    print("[初次翻译结果]: \n", translation_2)

    return { "translation_2": translation_2 }

# 规划执行任务
## 节点（node）注册
workflow.add_node("initial_translation", initial_translation)
workflow.add_node("reflect_on_translation", reflect_on_translation)
workflow.add_node("improve_translation", improve_translation)
## 连接节点
workflow.set_entry_point("initial_translation")
#workflow.add_edge(START, )
workflow.add_edge("initial_translation", "reflect_on_translation")
workflow.add_edge("reflect_on_translation", "improve_translation")
workflow.add_edge("improve_translation", END)

# 开始执行
app = workflow.compile()
result = app.invoke({
    "source_lang": "English",
    "target_lang": "中文",
    "source_text": """Translation Agent: Agentic translation using reflection workflow
This is a Python demonstration of a reflection agentic workflow for machine translation. The main steps are:

Prompt an LLM to translate a text from source_language to target_language;
Have the LLM reflect on the translation to come up with constructive suggestions for improving it;
Use the suggestions to improve the translation.
Customizability
By using an LLM as the heart of the translation engine, this system is highly steerable. For example, by changing the prompts, it is easier using this workflow than a traditional machine translation (MT) system to:

Modify the output's style, such as formal/informal.
Specify how to handle idioms and special terms like names, technical terms, and acronyms. For example, including a glossary in the prompt lets you make sure particular terms (such as open source, H100 or GPU) are translated consistently.
Specify specific regional use of the language, or specific dialects, to serve a target audience. For example, Spanish spoken in Latin America is different from Spanish spoken in Spain; French spoken in Canada is different from how it is spoken in France.
This is not mature software, and is the result of Andrew playing around with translations on weekends the past few months, plus collaborators (Joaquin Dominguez, Nedelina Teneva, John Santerre) helping refactor the code.

According to our evaluations using BLEU score on traditional translation datasets, this workflow is sometimes competitive with, but also sometimes worse than, leading commercial offerings. However, we’ve also occasionally gotten fantastic results (superior to commercial offerings) with this approach. We think this is just a starting point for agentic translations, and that this is a promising direction for translation, with significant headroom for further improvement, which is why we’re releasing this demonstration to encourage more discussion, experimentation, research and open-source contributions.

If agentic translations can generate better results than traditional architectures (such as an end-to-end transformer that inputs a text and directly outputs a translation) -- which are often faster/cheaper to run than our approach here -- this also provides a mechanism to automatically generate training data (parallel text corpora) that can be used to further train and improve traditional algorithms. (See also this article in The Batch on using LLMs to generate training data.)

Comments and suggestions for how to improve this are very welcome!"""
})

print(result)
```

  

```Python
# 绘制流程图
from mermaid import Mermaid
Mermaid(app.get_graph().draw_mermaid())
```

  

```Python
Agently Workflow

- Agently官网：[Agently.cn](https://Agently.cn)
- Agently Workflow与LangGraph的详细比较：[点击查看](https://github.com/Maplemx/Agently/blob/main/playground/constrast_between_Agently_workflow_and_LangGraph.ipynb)
- Agently Workflow详细教程：[点击查看](https://agently.cn/guides/workflow/index.html)
```

```Python
import json
from ENV import deep_seek_url, deep_seek_api_key, deep_seek_default_model
import Agently
import os 

# 将模型请求配置设置到agent工厂，后续工厂创建的agent对象都可以继承这个配置
agent_factory = (
    Agently.AgentFactory()
        .set_settings("current_model", "OAIClient")
        .set_settings("model.OAIClient.url", deep_seek_url)
        .set_settings("model.OAIClient.auth", { "api_key": deep_seek_api_key })
        .set_settings("model.OAIClient.options", { "model": deep_seek_default_model })
)

# 创建工作流
workflow = Agently.Workflow()

# 定义关键处理节点
## 首次翻译
@workflow.chunk()
def initial_translation(inputs, storage):
    source_lang = storage.get("source_lang")
    target_lang = storage.get("target_lang")
    source_text = storage.get("source_text")

    # 创建一个翻译agent来执行任务
    translate_agent = agent_factory.create_agent()
    
    # 给翻译agent设置system信息
    translate_agent.set_agent_prompt(
        "role",
        f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
    )

    # 向翻译agent发起翻译任务请求
    translation_1 = (
        translate_agent
        .input(
f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""
        )
        .start()
    )

    # 保存翻译结果
    storage.set("translation_1", translation_1)
    # 保存翻译agent备用
    storage.set("translate_agent", translate_agent)
    return {
        "stage": "initial translation",
        "result": translation_1
    }

## 反思优化
@workflow.chunk()
def reflect_on_translation(inputs, storage):
    source_lang = storage.get("source_lang")
    target_lang = storage.get("target_lang")
    source_text = storage.get("source_text")
    country = storage.get("country", "")
    translation_1 = storage.get("translation_1")

    # 创建一个反思agent来执行任务
    reflection_agent = agent_factory.create_agent()

    # 给反思agent设置system信息
    reflection_agent.set_agent_prompt(
        "role",
        f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."
    )

    additional_rule = (
        "The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}."
        if country != ""
        else ""
    )

    # 向反思agent发起反思任务
    reflection = (
        reflection_agent
            .input(
f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
{additional_rule}

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""
            )
            .start()
    )

    # 保存反思结果
    storage.set("reflection", reflection)
    return {
        "stage": "reflection",
        "result": reflection
    }

## 二次翻译
@workflow.chunk()
def improve_translation(inputs, storage):
    source_lang = storage.get("source_lang")
    target_lang = storage.get("target_lang")
    source_text = storage.get("source_text")
    translation_1 = storage.get("translation_1")
    reflection = storage.get("reflection")

    # 使用保存下来的翻译agent
    translate_agent = storage.get("translate_agent")

    # 直接发起二次翻译任务
    translation_2 = (
        translate_agent
            .input(
f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""
            )
            .start()
    )

    # 保存二次翻译结果
    storage.set("translation_2", translation_2)
    return {
        "stage": "improve translation",
        "result": translation_2
    }

# 连接工作块
(
    workflow
        .connect_to("initial_translation")
        .connect_to("reflect_on_translation")
        .connect_to("improve_translation")
        .connect_to("end")
)

# 添加过程输出优化
@workflow.chunk_class()
def output_stage_result(inputs, storage):
    print(f"[{ inputs['default']['stage'] }]:\n", inputs["default"]["result"])
    return

(
    workflow.chunks["initial_translation"]
        .connect_to("@output_stage_result")
        .connect_to("reflect_on_translation.wait")
)
(
    workflow.chunks["reflect_on_translation"]
        .connect_to("@output_stage_result")
        .connect_to("improve_translation.wait")
)
(
    workflow.chunks["improve_translation"]
        .connect_to("@output_stage_result")
)

# 启动工作流
result = workflow.start(storage = {
    "source_lang": "English",
    "target_lang": "中文",
    "source_text": """Translation Agent: Agentic translation using reflection workflow
This is a Python demonstration of a reflection agentic workflow for machine translation. The main steps are:

Prompt an LLM to translate a text from source_language to target_language;
Have the LLM reflect on the translation to come up with constructive suggestions for improving it;
Use the suggestions to improve the translation.
Customizability
By using an LLM as the heart of the translation engine, this system is highly steerable. For example, by changing the prompts, it is easier using this workflow than a traditional machine translation (MT) system to:

Modify the output's style, such as formal/informal.
Specify how to handle idioms and special terms like names, technical terms, and acronyms. For example, including a glossary in the prompt lets you make sure particular terms (such as open source, H100 or GPU) are translated consistently.
Specify specific regional use of the language, or specific dialects, to serve a target audience. For example, Spanish spoken in Latin America is different from Spanish spoken in Spain; French spoken in Canada is different from how it is spoken in France.
This is not mature software, and is the result of Andrew playing around with translations on weekends the past few months, plus collaborators (Joaquin Dominguez, Nedelina Teneva, John Santerre) helping refactor the code.

According to our evaluations using BLEU score on traditional translation datasets, this workflow is sometimes competitive with, but also sometimes worse than, leading commercial offerings. However, we’ve also occasionally gotten fantastic results (superior to commercial offerings) with this approach. We think this is just a starting point for agentic translations, and that this is a promising direction for translation, with significant headroom for further improvement, which is why we’re releasing this demonstration to encourage more discussion, experimentation, research and open-source contributions.

If agentic translations can generate better results than traditional architectures (such as an end-to-end transformer that inputs a text and directly outputs a translation) -- which are often faster/cheaper to run than our approach here -- this also provides a mechanism to automatically generate training data (parallel text corpora) that can be used to further train and improve traditional algorithms. (See also this article in The Batch on using LLMs to generate training data.)

Comments and suggestions for how to improve this are very welcome!"""
})

# 打印执行结果
#print(workflow.storage.get("translation_1"))
#print(workflow.storage.get("reflection"))
#print(workflow.storage.get("translation_2"))
print(json.dumps(result, indent=4, ensure_ascii=False))
```