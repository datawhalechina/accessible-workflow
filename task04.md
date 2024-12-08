## 开源翻译工作流项目的具体内容:

项目中三个关键代码，第一个是initial translation，做了两件关键事情，一是说明模型当前扮演的角色，二是写了system的message，给定原始语种和目标语种，给出翻译任务并要求根据原始语种给出目标语种的翻译，不要做无关解释，接下来给定source text，用户以特定方式给出，工作结果是列出翻译结果；

第二个代码做了一轮对上一轮翻译的反思，除说明是语言学家外，还提供原始文本和翻译，目标是改进翻译，给出改进意见，这里有编码冗余，进行了是否有国别的判断，特定地区时语种要更符合当地口语化表达，需仔细阅读原始文本和提供的翻译内容并给出相关建议，结果是一堆改进建议；

第三个代码是回到最初始的语言学家，看到第一轮的原始文本、出发文本和改进建议，进行最终翻译；

主调用函数的输入变量包括原始语种、目标语种、原始文本和是否有特定国别的处理，通过多轮函数调用和参数传递构建起最简单的三步直线链条，整个思路强调在模型使用过程中，很难通过直接依赖大模型和prompt的组合完成高质量工作，可考虑利用特定场景下的步骤编排、输入输出传递的方式完成更好任务；

## 工作流复线及框架的使用:

读完代码后看整个项目思路，是让模型完成首轮翻译后自我反思修正，提升文本最终翻译质量；

探讨用框架完成工作流复线的问题，什么时候应该使用框架，什么时候不应该使用或没必要使用，如存在复杂逻辑、数据等待、中间状态存储，需要做更复杂条件判断，手写代码难维护，编码过程中思路整理和结构体系需要更好的书写规范结构时，就需要使用框架；

项目地址：[https://github.com/andrewyng/translation-agent](https://github.com/andrewyng/translation-agent)

- 项目基本思路：

- 让模型在完成首轮翻译之后，通过自我反思后修正的工作流优化翻译结果，以提升最终文本翻译的质量

- 关键步骤：

1. 第一步：

- 输入信息：**原始文本语言(source_lang)** 、**翻译目标语言(target_lang)** 和 **原始文本(source_text)**

- 角色设定：以翻译文本为任务目标的语言学家

- 输出结果：基于所有输入信息，对 **原始文本(source_text)** 进行 **第一轮翻译的结果(translation_1)**

2. 第二步：

- 输入信息：**原始文本语言(source_lang)** 、**翻译目标语言(target_lang)** 、 **原始文本(source_text)** 和 **第一轮翻译结果(translation_1)**

- 角色设定：以阅读原始文本和翻译文本，并给出翻译改进意见为任务目标的语言学家

- 输出结果：基于所有输入信息，对 **第一轮翻译结果(translation_1)** 提出的 **改进意见反思(reflection)**

3. 第三步：

- 输入信息：**原始文本语言(source_lang)** 、**翻译目标语言(target_lang)** 、 **原始文本(source_text)** 、 **第一轮翻译结果(translation_1)** 和 **改进意见反思(reflection)**

- 角色设定：以翻译文本为任务目标的语言学家（和第一步相同）

- 输出结果：基于所有输入信息，给出的**第二轮优化后翻译结果(translation_2)**

- 关键代码文件：[https://github.com/andrewyng/translation-agent/blob/main/src/translation_agent/utils.py](https://github.com/andrewyng/translation-agent/blob/main/src/translation_agent/utils.py)

```Python

def one_chunk_initial_translation(
    source_lang: str, target_lang: str, source_text: str
) -> str:
    """
    Translate the entire text as one chunk using an LLM.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text (str): The text to be translated.

    Returns:
        str: The translated text.
    """

    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

    translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""

    prompt = translation_prompt.format(source_text=source_text)

    translation = get_completion(prompt, system_message=system_message)

    return translation


def one_chunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    country: str = "",
) -> str:
    """
    Use an LLM to reflect on the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        country (str): Country specified for target language.

    Returns:
        str: The LLM's reflection on the translation, providing constructive criticism and suggestions for improvement.
    """

    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    if country != "":
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

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

    else:
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \

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

    prompt = reflection_prompt.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        translation_1=translation_1,
    )
    reflection = get_completion(prompt, system_message=system_message)
    return reflection


def one_chunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    reflection: str,
) -> str:
    """
    Use the reflection to improve the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        reflection (str): Expert suggestions and constructive criticism for improving the translation.

    Returns:
        str: The improved translation based on the expert suggestions.
    """

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

    return translation_2


def one_chunk_translate_text(
    source_lang: str, target_lang: str, source_text: str, country: str = ""
) -> str:
    """
    Translate a single chunk of text from the source language to the target language.

    This function performs a two-step translation process:
    1. Get an initial translation of the source text.
    2. Reflect on the initial translation and generate an improved translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The text to be translated.
        country (str): Country specified for target language.
    Returns:
        str: The improved translation of the source text.
    """
    translation_1 = one_chunk_initial_translation(
        source_lang, target_lang, source_text
    )

    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, translation_1, country
    )
    translation_2 = one_chunk_improve_translation(
        source_lang, target_lang, source_text, translation_1, reflection
    )

    return translation_2

```
