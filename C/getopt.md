## 参数解析

<!-- toc -->


```C
 #include <unistd.h>
       int getopt(int argc, char * const argv[],const char *optstring);
       extern char *optarg;
       extern int optind, opterr, optopt;
 
       #include <getopt.h>
       int getopt_long(int argc, char * const argv[], const char *optstring, 
        const struct option *longopts, int *longindex);
       int getopt_long_only(int argc, char * const argv[], const char *optstring, 
        const struct option *longopts, int *longindex);
```


介绍几个全局变量
```C
optarg; //指向当前选项参数（如果有）的指针
optind,  //再次调用 getopt() 时的下一个 argv 指针的索引
opterr,  //指定getopt、getopt_long、getopt_long_only是否在遇到错误时将错误输出到标准输出流
optopt;  //最后一个未知选项
```

## getopt
```C
  int getopt(int argc, char * const argv[],const char *optstring);
```
argc和argv与main函数的两个参数相匹配的

optstring是一个字符串，形式如"a: b:: c: d "，分别表示选项 程序支持的命令行选项有-a、-b、-c、-d，冒号含义如下：

只有一个字符，不带冒号——只表示选项， 如-c

一个字符，后接一个冒号——表示选项后面带一个参数，如-a 100

一个字符，后接两个冒号——表示选项后面带一个可选参数，即参数可有可无，如果带参数，则选项与参数直接不能有空格，形式应该如-b200

```C
#include <stdio.h>                                                              
#include <unistd.h>
 
int main(int argc, char* argv[])
{
    int opt;
    char *optstring = "a:b::cd:";
        
    while ((opt = getopt(argc, argv, optstring)) != -1) 
    {   
        printf("opt = %c\n", opt);<span style="white-space:pre">    </span> //输出选项名称
        printf("optarg = %s\n", optarg); //输出选项后接的参数
        printf("optind = %d\n", optind); //输出当前命令行参数下一个参数的下标
        printf("argv[optind-1] = %s\n\n", argv[optind-1]); //
    }   
 
    return 0;
}
```


## getopt_long
```C
       int getopt_long(int argc, char * const argv[],
                  const char *optstring,
                  const struct option *longopts, int *longindex);
```
longopts 定义如下

```C
           struct option {
               const char *name;
               int         has_arg;
               int        *flag;
               int         val;
           };
```

name：长选项名字 

has_arg:是否需要参数。值有三种情况

```C
    # define no_argument        0    //不需要参数  
    # define required_argument  1    //必须指定参数  
    # define optional_argument  2    //参数可选  
```
no_argument(或者是0)时  ——参数后面不跟参数值

required_argument(或者是1)时 ——参数输入格式为：--参数 值 或者 --参数=值。

optional_argument(或者是2)时  ——参数输入格式只能为：--参数=值。

flag和val相互依赖，主要分两种情况：

（1）、flag为NULL，val值用于确定该长选项，所以需要为长选项指定唯一的val值。这里也为长选项和短选项建立了桥梁。

（2）、flag不为NULL，则将val值存放到flag所指向的存储空间，用于标识该长选项出现过。


- 程序中使用短选项，则返回短选项字符（如‘n'），当需要参数是，则在返回之前将参数存入到optarg中。

- 程序中使用长选项，返回值根据flag和val确定。当flag为NULL，则返回val值。所以根据val值做不同的处理，这也说明了val必须唯一。当
val值等于短选项值，则可以使用短选项解析函数解析长选项；当flag不为NULL，则将val值存入flag所指向的存储空间，getopt_long返回0

- 出现未定义的长选项或者短选项，getopt_long返回？

- 解析完毕，getopt_long返回-1

```C
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
 
#define VAL1    0
#define VAL2    1
#define VAL3    2
 
int main(int argc, char* argv[])
{
    int opt;
    int this_option_optind = optind ? optind : 1;
    char *optstring = "a:b::cd:";
    struct option long_options[] = {
        {"lopt1", no_argument,          0, VAL1},
        {"lopt2", required_argument,    0, VAL2},
        {"lopt3", optional_argument,    0, VAL3},
        {"lopt4", no_argument,          0, VAL1},
        {0,0,0,0}
    };
    int option_index = 0;
 
 
    while ((opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1)
    {
        
        switch(opt)
        {
            case VAL1:
            case VAL2:
            case VAL3:
                printf("option %s", long_options[option_index].name);
                if (optarg)
                    printf(" with arg %s", optarg);
                    printf("\n");
                break;
            case 'a':
            case 'b':
            case 'c':
            case 'd':
                printf("opt = %c\n", opt);   //输出选项名称
                printf("optarg = %s\n", optarg); //输出选项后接的参数
                printf("optind = %d\n", optind); //输出当前命令行参数下一个参数的下标
                printf("argv[optind-1] = %s\n\n", argv[optind-1]); //
            default:
                exit(-1);
                break;
                
        }
    }
 
 
    return 0;
}
```

## getopt_long_only

```C
  int getopt_long_only(int argc, char * const argv[], const char *optstring, 
    const struct option *longopts, int *longindex);
```
getopt_long_only与getopt_long参数表和功能基本相同，主要差别的地方在于长选项参数的解析。

在getopt_long中，长选项参数需要由双横杠开始，即--name， 而-name会被解析成-n,-a,-m和-e在optstring中匹配

在getopt_long_only中，--name和-name都被解析成长选项参数。

 
```C++

#include <iostream>
#include <iomanip>
#include <getopt.h>


enum mum_t { MUM, MAM, MEM };

int   min_len          = 20;
int   sparseMult       = 1;
mum_t type             = MAM;
bool  rev_comp         = false;
bool  _4column         = false;
bool  nucleotides_only = false;
bool  forward          = true;
bool  setRevComp       = false;
bool  setBoth          = false;
bool  automatic        = true;
bool  automaticSkip    = true;
bool  automaticKmer    = true;
bool  suflink          = true;
bool  child            = false;
int   kmer             = 0;
bool  print_length     = false;
bool  printSubstring   = false;
bool  printRevCompForw = false;
int   K                = 1;
int   num_threads      = 1;
int   query_threads    = 0;
size_t max_chunk       = 50000;

int main(int argc, char* argv[]) {

  // Collect parameters from the command line.
  std::string save;
  std::string load;

  while (1) {
    static struct option long_options[] = {
      {"l", 1, 0, 0}, // 0
      {"mumreference", 0, 0, 0}, // 1
      {"b", 0, 0, 0}, // 2
      {"maxmatch", 0, 0, 0}, // 3
      {"mum", 0, 0, 0}, // 4
      {"mumcand", 0, 0, 0},  // 5
      {"F", 0, 0, 0}, // 6
      {"k", 1, 0, 0}, // 7
      {"threads", 1, 0, 0}, // 8
      {"n", 0, 0, 0}, // 9
      {"qthreads", 1, 0, 0}, // 10
      {"suflink", 1, 0, 0}, // 11
      {"child", 1, 0, 0}, // 12
      {"skip", 1, 0, 0}, // 13
      {"L", 0, 0, 0}, // 14
      {"r", 0, 0, 0}, // 15
      {"s", 0, 0, 0}, // 16
      {"c", 0, 0, 0}, // 17
      {"kmer", 1, 0, 0}, // 18
      {"save", 1, 0, 0}, // 19
      {"load", 1, 0, 0}, // 20
      {"max-chunk", 1, 0, 0}, // 21
      {"version", 0, 0, 0}, // 22
      {0, 0, 0, 0}
    };
    int longindex = -1;
    int c = getopt_long_only(argc, argv, "", long_options, &longindex);
    if(c == -1) break; // Done parsing flags.
    else if(c == '?') { // If the user entered junk, let him know.
      std::cerr << "Invalid parameters." << std::endl;
      usage(argv[0]);
    }
    else {
       //Branch on long options.
      switch(longindex) {
      case 0: min_len = atol(optarg); break;
      case 1: type = MAM; break;
      case 2: setBoth = true; break;
      case 3: type = MEM; break;
      case 4: type = MUM; break;
      case 5: type = MAM; break;
      case 6: _4column = true; break;
      case 7: K = atoi(optarg); break;
      case 8: num_threads = atoi(optarg); break;
      case 9: nucleotides_only = true; break;
      case 10: query_threads = atoi(optarg) ; break;
      case 11: suflink = atoi(optarg) > 0;    automatic = false; break;
      case 12: child = atoi(optarg) > 0;      automatic = false; break;
      case 13: sparseMult = atoi(optarg); automaticSkip = false; break;
      case 14: print_length = true; break;
      case 15: setRevComp = true; break;
      case 16: printSubstring = true; break;
      case 17: printRevCompForw = true; break;
      case 18: kmer = atoi(optarg); automaticKmer = false; break;
      case 19: save = optarg; break;
      case 20: load = optarg; break;
      case 21: max_chunk = atoi(optarg); break;
      case 22:
#ifdef VERSION
        std::cout << VERSION << '\n';
#else
        std::cout << "<unknown version>\n";
#endif
        exit(0);
      default: break;
      }
    }
  }
  return 0
}

```

[参考1](https://blog.csdn.net/windeal3203/article/details/9049967)

[参考2](https://blog.csdn.net/pengrui18/article/details/8078813)