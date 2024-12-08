# **Agently** **Workflow**

```Python
- Agently官网：[Agently.cn](https://Agently.cn)

- Agently Workflow与LangGraph的详细比较：[点击查看]

(https://github.com/Maplemx/Agently/blob/main/playground/constrast_between_Agently_workflow_and_LangGraph.ipynb)
- Agently Workflow详细教程：[点击查看](https://agently.cn/guides/workflow/index.html)
```

## 客服工作流的工作流程

![Alt](img_04.jpg "工作流程图")

```Python

workflow = Agently.Workflow()
@workflow.chunk()
def user_input(inputs, storage):
    storage.set("user_input", input("[请输入您的要求]: "))
    return

@workflow.chunk()
def judge_intent_and_quick_reply(inputs, storage):
    result = (
        agent
            .input(storage.get("user_input"))
            .output({
                "user_intent": ("闲聊 | 售后问题 | 其他", "判断用户提交的{input}内容属于给定选项中的哪一种"),
                "quick_reply": (
                    "str",
"""如果{user_intent}=='闲聊'，那么直接给出你的回应；
如果{user_intent}=='售后问题'，那么请用合适的方式告诉用户你已经明白用户的诉求，安抚客户情绪并请稍等你去看看应该如何处理；
如果{user_intent}=='其他'，此项输出null""")
            })
            .start()
    )
    storage.set("reply", result["quick_reply"])
    return result["user_intent"]

@workflow.chunk()
def generate_after_sales_reply(inputs, storage):
    storage.set("reply", (
        agent
            .input(storage.get("user_input"))
            .instruct(
"""请根据{input}的要求，以一个专业客户服务人员的角色给出回复，遵循如下模板进行回复：
亲爱的客户，感谢您的耐心等待。
我理解您希望{{复述客户的要求}}，是因为{{复述客户要求提出要求的理由}}，您的心情一定非常{{阐述你对客户心情/感受的理解}}。
{{给出对客户当前心情的抚慰性话语}}。
我们会尽快和相关人员沟通，并尽量进行满足。请留下您的联系方式以方便我们尽快处理后与您联系。
"""
)
            .start()
    ))
    return

@workflow.chunk()
def generate_other_topic_reply(inputs, storage):
    storage.set("reply", "我们好像不应该聊这个，还是回到您的问题或诉求上来吧。")
    return

@workflow.chunk_class()
def reply(inputs, storage):
    print("[回复]: ", storage.get("reply"))
    return

(
    workflow
        .connect_to("user_input")
        .connect_to("judge_intent_and_quick_reply")
        .if_condition(lambda return_value, storage: return_value=="闲聊")
            .connect_to("@reply")
            .connect_to("end")
        .elif_condition(lambda return_value, storage: return_value=="售后问题")
            .connect_to("@reply")
            .connect_to("generate_after_sales_reply")
            .connect_to("@reply")
            .connect_to("user_input")
        .else_condition()
            .connect_to("generate_other_topic_reply")
            .connect_to("@reply")
            .connect_to("END")
)

workflow.start()
pass

```
