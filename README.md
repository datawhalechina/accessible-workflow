# 项目名称
accessible-workflow

## 安装所需要使用的包
*注: 建议 python 版本在3.10及以上 *
```
!pip install openai langgraph Agently==3.3.4.5 mermaid-python nest_asyncio
```

因为本课使用的langgraph可能需要依赖langchain 0.2.10版本，如果与本地依赖langchain版本不同
请学习完本课之后对langchain自行进行处理，以免在影响本地开发和学习

```
!pip install langchain==0.1.20
!pip install langchain-openai==0.1.6
!pip install langchain-community==0.0.38
```

## 关于key

使用的是 [https://www.deepseek.com/] API,要到官方申请API调用权限，注册送500万 token

## 目录

- [task01](task01.md) 课程介绍，试一试
- [task02](task02.md) 如何区别工作流和智能体
- [task03](task03.md) 将问题工作流化
- [task04](task04.md) 翻译工作流原理讲解
- [task05](task05.md) 分别使用LangGraph和Agently重现翻译工作流

![Alt](img_03.png "deepseek官网")

## Roadmap

*注：说明当前项目的规划，并将每个任务通过 Issue 形式进行对外进行发布。*

## 参与贡献

- 如果你想参与到项目中来欢迎查看项目的 [Issue]() 查看没有被分配的任务。
- 如果你发现了一些问题，欢迎在 [Issue]() 中进行反馈🐛。
- 如果你对本项目感兴趣想要参与进来可以通过 [Discussion]() 进行交流💬。

如果你对 Datawhale 很感兴趣并想要发起一个新的项目，欢迎查看 [Datawhale 贡献指南](https://github.com/datawhalechina/DOPMC#%E4%B8%BA-datawhale-%E5%81%9A%E5%87%BA%E8%B4%A1%E7%8C%AE)。

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| 刍荛 | 项目负责人 | 刍荛 |

*注：表头可自定义，但必须在名单中标明项目负责人*

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

*注：默认使用CC 4.0协议，也可根据自身项目情况选用其他协议*
