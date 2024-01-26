#include "structs/device_mat.cuh"
#include <thread>
#include <chrono>


int main() {
    sde::Size size(1000000000, 10000);
    sde::DeviceMat<float> de(size);

    while(1) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return 0;
}