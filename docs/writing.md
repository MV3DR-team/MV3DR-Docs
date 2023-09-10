---
layout: default
title: 如何编辑本文档
nav_order: A
---

# 如何编辑本文档

这篇文章来讲如何编辑本文档。

## 文档基本信息

本文档通过 [Jekyll] 将 markdown 文件转化为静态网页，采用 [Just the Docs] 主题，将文件托管至 Github 并通过 Github Pages 部署。要编辑文档内容或布局并在本地预览，需要先[安装 Jekyll]。关于 Github Pages 的更多信息，参考 [Github 文档]。

安装好 Jekyll 后，将本文档的 [Github 项目]克隆至本地（推荐使用 [Github Desktop] 操作，减小学习 git 的门槛，后续更便捷）。

所有文档内容全部存放于 docs 文件夹下，每个 .md 文件的文件名即为最终部署到网页的 URL。比如本篇文章的文件名为 `writing.md`，最终网页地址为 `https://mv3dr-team.github.io/MV3DR-Docs/docs/writing.html`。如果要修改 URL，则需要在文件头添加 permalink（关于文件头在后面有更详细介绍）。

## 使用 markdown 写作

### markdown

markdown（文件后缀名：.md）是一种轻量化文本标记语言，通过简单的语法进行排版，让人们更多地关注内容本身。

推荐使用 VSCode 和 [Markdown Preview Enhanced 插件]进行 .md 写作。

Markdown 语法非常简洁，网上教程也很多，可以参考[此 b 站视频]（这个视频就是上面写的推荐方法），也可以自行搜索其他教程。



### Typora

需要收费的markdown文件编辑软件，可以使用早期的版本，或者对新版本使用一些特殊的方法来正常使用软件。

Typora可以做到实时渲染（VsCode插件也可以），并且能使用快捷键来减少格式代码的输入，更加方便快捷。

同时，你可以直接复制图片在typora里查看。但需要注意的是。你在复制项目到其他路径的时候，一定要查看文件的路径是否正确，毕竟Typora的图片也引用的本地路径的图片路径。

完成的文档可以使用pdf导出，便于不支持使用md文件的情况下正常阅读。



### kramdown

本网站采用 kramdown 渲染，kramdown 在原本 markdown 的基础上添加了新的语法（也兼容了 markdown 本身的语法），得到更丰富的展现。可以参考 [kramdown 和 markdown 较大的差异比较]，也可以参考 [kramdown 官方文档]。

{: .note}
以上内容不需全学，只要会最基本的语法即可开始写作，之后有问题再现查文档。

不过，这里要强调一点，一定要**编写专业的文档**，这需要你遵守一些排版上的规范，这里有一个[简单的参考]。

## 一篇文章的结构

本文档的一篇文章就是一个 .md 文件，文件头是一段 YAML 代码，一般长这样：

```yml
---
layout: default
title: 如何编辑本文档
nav_order: A
---
```
`layout` 是文档的布局，default 布局是 Just the Docs 已经定义好的（就是你现在在网页上看到的这个样子），直接用即可。`title` 是文章的标题，即显示在旁栏的名字。`nav_order` 是这篇文章在旁栏的顺序，这个值可以是任何数字、字母或字符串，其他文章基本上都是数字，这篇文章是字母（相当于附录 A），字母会排在数字的后面，所以这篇文章在旁栏最底部（相对地，“主页” 的 `nav_order` 是 0，所以在第一个，我们暂时没有 `nav_order` 为负数的文章）。

如果你觉得文章的 URL 太长，可以使用 `permalink`，这么写：

```yml
---
layout: default
title: 如何编辑本文档
permalink: w
nav_order: A
---
```

这样这篇文章的 URL 就变为了 `https://mv3dr-team.github.io/MV3DR-Docs/docs/w` 了。不过还是推荐 permalink 尽量起可读的名字。

关于其他的 YAML 参数，参考 Jekyll 的[Front Matter]。

YAML 头之后是文章的主体，使用 markdown 语法写作即可。除了 kramdown 提供的之外，Just the Docs 还提供了额外的模块，参考 [UI Components]（按钮、标签、表格等等），这些东西的配置都关系到 `_config.yml` 文件，可以看看。

## 文档的总体结构

文档总体结构就是一个个文件夹，每个文件夹可能有子文件夹，每个文件夹内就是当前主题的文章，这个关系到网站旁栏的展示结构。想要自己构建文档的层次结构，参考 Just the Docs 文档中 [Navigation Structure] 一节。

## 在本地预览网页效果

如果想在提交前先在本地看看最终渲染出的网页是什么样子，则在文档工程的本地文件夹中打开命令行，并输入：

```bash
bundle install
bundle exec jekyll serve
```

这里有可能会有错误，你有可能要安装 bundle。~~我忘了具体要不要安装了，总之你先试试（bushi~~

如果没问题就可以在 `localhost:4000` 预览了，而且这个预览是实时的！（除非你修改了 `_config.yml`，这时候得重新输入。）

## 提交修改

当在本地预览没有问题时，就可以 commit & push 了，push 完成后等待一段时间网站就渲染好了，输入链接就可以看到更改了。这部分通过学习 git 可以有更好的了解。

## 额外的学习资料

1. 最重要的就是 Just the Docs 的[文档](https://just-the-docs.com/)和其 [Github 仓库](https://github.com/just-the-docs/just-the-docs)，如果想修改布局什么的就得学习一下。
2. 描述算法肯定得写数学公式，建议学习一下 LaTeX 的基本写法，可以参考 [LaTeX 公式保姆级教程 (以及其中的各种细节)](https://www.bilibili.com/video/BV1no4y1U7At)。
3. [Jekyll 文档](https://www.jekyll.com.cn/docs/)，有些 Just the Docs 文档没说的东西就得在这里面查了。
4. [Github 新手够用指南](https://www.bilibili.com/video/BV1e541137Tc)。

[Jekyll]: https://www.jekyll.com.cn/
[Just the Docs]: https://just-the-docs.com/
[安装 Jekyll]: https://jekyllcn.com/docs/installation/
[Github 文档]: https://docs.github.com/zh/pages/getting-started-with-github-pages/about-github-pages
[Github 项目]: https://github.com/MV3DR-team/MV3DR-Docs
[Github Desktop]: https://desktop.github.com/
[Markdown Preview Enhanced 插件]: https://shd101wyy.github.io/markdown-preview-enhanced/#/zh-cn/
[此 b 站视频]: https://www.bilibili.com/video/BV1si4y1472o
[kramdown 和 markdown 较大的差异比较]: https://gohom.win/2015/11/06/Kramdown-note/
[kramdown 官方文档]: https://kramdown.gettalong.org/syntax.html
[Front Matter]: https://jekyllrb.com/docs/front-matter/
[UI Components]: https://just-the-docs.com/docs/ui-components
[Navigation Structure]: https://just-the-docs.com/docs/navigation-structure/
[简单的参考]: https://xie.infoq.cn/article/69feb60ca6fba4ae0c8adeef6