
# Make 

```shell
SRCS=main.cpp solve.c
OBJS=$(patsubst %.o,%.c,$(patsubst %.o,%.cpp,$(SRCS)))
```
使用patsubst将 .c 和 .cpp文件替换为 .o文件

（1）Makefile中的 符号 $@, $^, $< 的意思：

　　$@  表示目标文件

　　$^  表示所有的依赖文件

　　$<  表示第一个依赖文件

　　$?  表示比目标还要新的依赖文件列表

（2）wildcard、notdir、patsubst的意思：

　　wildcard : 扩展通配符

　　notdir ： 去除路径

　　patsubst ：替换通配符
