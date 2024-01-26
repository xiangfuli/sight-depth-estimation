#include <iostream>
#include "cuda_runtime.h"
#include "structs/cuda_exception.cuh"

namespace sde {

class Size {
public:
    Size(size_t height, size_t width):height_(height), width_(width) {}
    size_t height_;
    size_t width_;

    std::string toString() const {
        return  "w * h: [" + std::to_string(width_) + ", " + std::to_string(height_) + "]";
    }

    bool operator== (const Size & s ) {
        return s.height_ == this->height_ && s.width_ == this->width_;
    }
};

template <typename Element>
class DeviceMat {
public:
    __host__ DeviceMat(Size &size):
    size_(size) {
        // assign the needed memory section
        cudaError ce = cudaMallocPitch(
            &this->elements_,
            &this->pitch_,
            this->size_.width_,
            this->size_.height_
        );

        std::stringstream description;

        if(ce != cudaSuccess) {
            throw CudaException("DeviceMat: unable to allocate pitched memory with size " + this->size_.toString(), ce);
        }

        this->stride_ = this->pitch_ / sizeof(Element);

        ce = cudaMalloc(
            &device_mat_,
            sizeof(*this)
        );
        if(ce != cudaSuccess) {
            throw CudaException("DeviceMat: unable to allocate memory for mat object.", ce);
        }

        ce = cudaMemcpy(
            this->device_mat_,
            this,
            sizeof(*this),
            cudaMemcpyHostToDevice
        );
        if(ce != cudaSuccess) {
            throw CudaException("DeviceMat: unable to copy mat memory from host to device.", ce);
        }
    }

    __host__
    ~DeviceMat() noexcept(false) {
        cudaError ce = cudaFree(this->elements_);
        if(ce != cudaSuccess) {
            throw CudaException("DeviceMat: unable to free mat memory on the device.", ce);
        }
        ce = cudaFree(this->device_mat_);
        if(ce != cudaSuccess) {
            throw CudaException("DeviceMat: unable to free mat object memory on the device.", ce);
        }
    }

    __host__
  void zero()
  {
    cudaError ce = cudaMemset2D(
          this->data_,
          this->pitch_,
          0,
          this->width_*sizeof(Element),
          this->height_);
    if(ce != cudaSuccess) {
      throw CudaException("DeviceMat: unable to set value to zeros.", ce);
    }
  }

  __host__
  DeviceMat<Element> & operator=(const DeviceMat<Element> &other_image)
  {
    if(this != &other_image)
    {
        static_assert(this->size_ == other_image.size_);
        cudaError ce = cudaMemcpy2D(this->data_,
                                this->pitch_,
                                other_image.data_,
                                other_image.pitch_,
                                this->width_*sizeof(Element),
                                this->height_,
                                cudaMemcpyDeviceToDevice);
                            
      if(ce != cudaSuccess) {
        throw CudaException("DeviceMat operator '=': unable to copy data from another mat.", ce);
      } 
    }
    return *this;
  }

    Size size_;
    size_t pitch_;
    size_t stride_;
    Element *elements_;
    DeviceMat<Element> *device_mat_;
};

}