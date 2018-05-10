## Thread, functtion, bind, lambda  expression

<!-- toc -->

## 使用thread
```C++
#include <thread.h>
void f1(int& n){
  n++;
}
int main(){
  int n = 0;
  std::thread t1(f1, &n);
  t1.join()
}
```


## function & bind
* function

```C++

#include < functional>  
   
std::function< size_t(const char*)> print_func;  
   
/// normal function -> std::function object  
size_t CPrint(const char*) { ... }  
print_func = CPrint;  
print_func("hello world"):  

```
* bind

```C++
#include < functional>  
   
int Func(int x, int y);  
auto bf1 = std::bind(Func, 10, std::placeholders::_1);  
bf1(20); ///< same as Func(10, 20)  
   
class A  
{  
public:  
    int Func(int x, int y);  
};  
   
A a;  
auto bf2 = std::bind(&A::Func, a, std::placeholders::_1, std::placeholders::_2);  
bf2(10, 20); ///< same as a.Func(10, 20)  
   
std::function< int(int)> bf3 = std::bind(&A::Func, a, std::placeholders::_1, 100);  
bf3(10); ///< same as a.Func(10, 100)  
```

## Lambda expressions

Syntax

* \[ captures \] <tparams>(optional)(c++20) ( params ) specifiers exception attr -> ret requires(optional)(c++20) { body }  

* \[ captures \] ( params ) -> ret { body } 

* \[ captures \] ( params ) { body } 

* \[ captures \] { body } 

Explanation


* captures  - a comma-separated list of zero or more captures, optionally beginning with a capture-default.

Capture list can be passed as follows (see below for the detailed description):


[a,&b] where a is captured by copy and b is captured by reference.

\[this\] captures the current object (\*this) by reference

[&] captures all automatic variables used in the body of the lambda by reference and current object by reference if exists

[=] captures all automatic variables used in the body of the lambda by copy and current object by reference if exists

[] captures nothing

A lambda expression can use a variable without capturing it if the variable is a non-local variable or has static or thread local storage duration (in which case the variable cannot be captured), or is a reference that has been initialized with a constant expression.

A lambda expression can read the value of a variable without capturing it if the variable has const non-volatile integral or enumeration type and has been initialized with a constant expression, or is constexpr and trivially copy constructible.

Structured bindings cannot be captured.(since C++17)

* <tparams>(C++20)  - a template parameter list (in angle brackets), used to provide names to the template parameters of a generic lambda (see ClosureType::operator() below). Like in a template declaration, the template parameter list may be followed by an optional requires-clause, which specifies the constraints on the template arguments.

* params  - The list of parameters, as in named functions, except that default arguments are not allowed (until C++14). 

If auto is used as a type of a parameter, the lambda is a generic lambda. (since C++14)

* specifiers  - Optional sequence of specifiers.The following specifiers are allowed:

  - mutable: allows body to modify the parameters captured by copy, and to call their non-const member functions
  - constexpr: explicitly specifies that the function call operator is a constexpr function. When this specifier is not present, the function call operator will be constexpr anyway, if it happens to satisfy all constexpr function requirements(since C++17)

* exception - provides the exception specification or the noexcept clause for operator() of the closure type

* attr  - provides the attribute specification for operator() of the closure type

* ret - Return type. If not present it's implied by the function return statements (or void if it doesn't return any value)

* requires  - adds a constraint to operator() of the closure type

* body  - Function body

