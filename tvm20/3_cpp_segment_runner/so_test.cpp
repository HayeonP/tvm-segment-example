#include <iostream>
#include <dlfcn.h>

int main() {
    const char* so_path = "./resnet18.so";
    //const char* so_path = "/home/rubis/workspace/tvm-segment-20/build/libtvm_allvisible.so";
    // RTLD_LAZY: 필요할 때 심볼 로딩
    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "dlopen 실패: " << dlerror() << std::endl;
        return 1;
    } else {
        std::cout << "dlopen 성공!" << std::endl;
        // 사용이 끝나면 반드시 닫아줌
        dlclose(handle);
    }
    return 0;
}
