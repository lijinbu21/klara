windows下，控制台不便调试，全局重定向std::cout到文件中，这样不需要修改每个函数中的输出代码
### 方法：全局重定向 `std::cout` 到文件

1. **在程序开始时重定向 `std::cout` 到文件**。
2. **在程序结束时恢复 `std::cout` 到控制台**。

```
#include <iostream>
#include <fstream>
#include <streambuf>

int main() {
    // 保存原始的 cout 缓冲区
    std::streambuf* original_cout_buffer = std::cout.rdbuf();

    // 打开文件并将 cout 重定向到该文件
    std::ofstream debug_file("debug_output.txt");
    if (!debug_file.is_open()) {
        std::cerr << "Failed to open debug file." << std::endl;
        return 1;
    }
    std::cout.rdbuf(debug_file.rdbuf());

    // 程序中的所有 std::cout 输出将被重定向到 debug_output.txt 文件
    std::cout << "This is a test message." << std::endl;

    // 你的其他代码...
    // 调用不同函数，所有 std::cout 输出都会写入文件

    // 程序结束时，恢复 std::cout 的输出到控制台
    std::cout.rdbuf(original_cout_buffer);
    debug_file.close();

    return 0;
}

```
不需要提前创建文件，`std::ofstream`会自动创建文件。如果文件 `"debug_output.txt"` 不存在，`std::ofstream` 会创建一个新文件；如果文件已经存在，则会清空文件内容并覆盖。如果你希望追加内容而不是覆盖，可以使用 `std::ios::app` 模式。
```
std::ofstream debug_file("debug_output.txt", std::ios::out);  // 覆盖模式（默认）
std::ofstream debug_file("debug_output.txt", std::ios::app);  // 追加模式
```