# ctags

ctags有一个选项可以指定文件使用的语言：langmap。

比如，指定.cu文件中语言为C++可以加入选项：--langmap=c++:+.cu。


于是，可以使用以下方式来为cuda项目建立tag：

ctags -R --langmap=c++:+.cu *


为了避免每次都添加一长串选项，可以将这些选项直接写入ctags的配置文件中。

打开$HOME/.ctags（如果没有则创建），在其中添加如下两行：

```sh
--langmap=c++:+.cu
--langmap=c++:+.cuh
```
这样，每次使用ctags时都会添加这两个选项。


