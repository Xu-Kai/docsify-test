## C++ Union

union主要是共享内存，分配内存以其最大的结构或对象为大小，即sizeof最大的。

```C++
#include <iostream>
#include <cstdint>
union S
{
    std::int32_t n;     // 占用 4 字节
    std::uint16_t s[2]; // 占用 4 字节
    std::uint8_t c;     // 占用 1 字节
};                      // 整个联合体占用 4 字节
 
int main()
{
    S s = {0x12345678}; // 初始化首个成员， s.n 现在是活跃成员
    // 于此点，从 s.s 或 s.c 读取是未定义行为
    std::cout << std::hex << "s.n = " << s.n << '\n';
    s.s[0] = 0x0011; // s.s 现在是活跃成员
    // 在此点，从 n 或 c 读取是 UB 但大多数编译器都对其定义
    std::cout << "s.c is now " << +s.c << '\n' // 11 或 00 ，依赖平台
              << "s.n is now " << s.n << '\n'; // 12340011 或 00115678
}
```

