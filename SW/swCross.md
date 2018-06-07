# sw共享内存

<!-- toc -->

```C
#define cross_p 0x6000000000
char *temp = (char*)cross_p;
char *temp2 = (char*)cross_p + temp_size;
if(rank % 4 == 0){
	graph_reg(temp, 4*temp_size);
	graph_reg(temp, 4*temp2_size);
}
```

使用crosssize的时候要分配到指定的地址,cross size最大16G，每个核组这样还剩4G，host size + share size 小于3.2G(大约80%可用），第一次使用这段内存时，每个节点内的4个进程中任意一个进程调用一下上面的方法, 不然会变慢。

程序提交时，需要在提交命令里面加-crosssize 