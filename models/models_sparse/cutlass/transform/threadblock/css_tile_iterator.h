#pragma once

#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"

#include "stdio.h"


namespace cutlass {
namespace transform {
namespace threadblock {

template<typename SEMEShape>
class CSSAddressIteratorA;


template<typename SEMEShape_>
class CSSAddressIteratorA {
    public:
        using SEMEShape = SEMEShape_;

        float* address;
        int height;
        int width;
        int add_pos;
        int length;
        int* flags;
        int flag_base;
        int flag_pos;
        bool is_over;

        CUTLASS_DEVICE
        CSSAddressIteratorA(int blockIdx_x, int threadId_x, float* global_address ,int matrix_width, int matrix_height, int* global_flags, int flags_width){
            this->flag_base = blockIdx_x*flags_width;
            this->length = global_flags[this->flag_base];
            this->flag_pos = threadId_x/SEMEShape::kM + 1;
            this->flags = global_flags;
            this->height = matrix_height;
            this->width = matrix_width;
            int add_pos = blockIdx_x*SEMEShape::kM + threadId_x%SEMEShape::kM;
            if (add_pos >= matrix_width){
                this->is_over = true;
                // printf("Cross the border: matrix_width=%d, threadIdx.x=%d.\n", matrix_width, threadId_x);
            }
            else{
                this->is_over = false;
            }
            //this->address = reinterpret_cast<float *>(global_address + add_pos*sizeof(float));
            this->address = global_address + add_pos;
        }

        CUTLASS_DEVICE
        CSSAddressIteratorA(){}

        CUTLASS_DEVICE
        CSSAddressIteratorA& operator++(){
            this->flag_pos+=2;
            return *this;
        }

        CUTLASS_DEVICE
        CSSAddressIteratorA operator++(int){
            CSSAddressIteratorA old = *this;
            operator++();
            return old;
        }

        CUTLASS_DEVICE
        int getLength(){
            return this->length;
        }

        CUTLASS_DEVICE
        float* get() const{
            return this->address + this->flags[this->flag_pos + this->flag_base]*this->width;
            //return this->address;
        }

        CUTLASS_DEVICE
        bool valid(){
            if(this->is_over)
                return false;
            if(this->flag_pos > this->length )
                return false;
            else
                return true;
        }
};


template<typename SEMEShape_>
class CSSIteratorA{
    public:
        using SEMEShape = SEMEShape_;
        using Fragment = cutlass::Array<float, 4>;
        using AddressIterator = CSSAddressIteratorA<SEMEShape>;

        AddressIterator add_itr;

        CUTLASS_DEVICE
        CSSIteratorA(int blockIdx_x, int threadId_x, float* global_address ,int matrix_width, int matrix_height, int* global_flags, int flags_width){
            this->add_itr = AddressIterator(blockIdx_x, threadId_x, global_address, matrix_width, matrix_height, global_flags, flags_width);
        }

        CUTLASS_DEVICE
        CSSIteratorA(){}

        CUTLASS_DEVICE
        void load(Fragment &frag){
            for(int i=0; i<4; i++){
                if(add_itr.valid())
                    cutlass::arch::global_load<float,sizeof(float)>(frag[i], add_itr.get(), true);
                else
                    frag[i] = 0;
                add_itr++;
            }
            // for(int i=0; i<4; i++){
            //     cutlass::arch::global_load<float,sizeof(float)>(frag[i], add_itr.get(), add_itr.valid());
            //     add_itr++;
            // }
        }

        CUTLASS_DEVICE
        bool valid(){
            return this->add_itr.valid();
        }

        CUTLASS_DEVICE
        int getLength(){
            return this->add_itr.getLength();
        }


};


template<typename SEMEShape_>
class CSSAddressIteratorB {
    public:
        using SEMEShape = SEMEShape_;

        float* address;
        int height;
        int width;
        int add_pos;
        int length;
        int* flags;
        int flag_base;
        int flag_pos;
        bool is_over;

        CUTLASS_DEVICE
        CSSAddressIteratorB(int blockIdx_x, int blockIdx_y, int threadId_x, float* global_address ,int matrix_width, int matrix_height, int* global_flags, int flags_width){
            this->flag_base = blockIdx_x*flags_width;
            this->length = global_flags[this->flag_base];
            this->flag_pos = threadId_x/SEMEShape::kM + 1;
            this->flags = global_flags;
            this->height = matrix_height;
            this->width = matrix_width;
            int add_pos = blockIdx_y*SEMEShape::kM + threadId_x%SEMEShape::kM;
            if(add_pos >= matrix_width){
                this->is_over = true;
                // printf("Cross the border: threadIdx.x=%d.\n", threadId_x);
            } 
            else{
                this->is_over = false;
            }
            //this->address = reinterpret_cast<float *>(global_address + add_pos*sizeof(float));
            this->address = global_address + add_pos;
        }

        CUTLASS_DEVICE
        CSSAddressIteratorB(){}

        CUTLASS_DEVICE
        CSSAddressIteratorB& operator++(){
            this->flag_pos+=2;
            return *this;
        }

        CUTLASS_DEVICE
        CSSAddressIteratorB operator++(int){
            CSSAddressIteratorB old = *this;
            operator++();
            return old;
        }

        CUTLASS_DEVICE
        int getLength(){
            return this->length;
        }

        CUTLASS_DEVICE
        float* get() const{
            return this->address + this->flags[this->flag_pos + this->flag_base]*this->width;
            //return this->address;
        }

        CUTLASS_DEVICE
        bool valid(){
            if(this->is_over)
                return false;
            if(this->flag_pos > this->length )
                return false;
            else
                return true;
        }
};


template<typename SEMEShape_>
class CSSIteratorB{
    public:
        using SEMEShape = SEMEShape_;
        using Fragment = cutlass::Array<float, 4>;
        using AddressIterator = CSSAddressIteratorB<SEMEShape>;

        AddressIterator add_itr;

        CUTLASS_DEVICE
        CSSIteratorB(int blockIdx_x, int blockIdx_y, int threadId_x, float* global_address ,int matrix_width, int matrix_height, int* global_flags, int flags_width){
            this->add_itr = AddressIterator(blockIdx_x, blockIdx_y, threadId_x, global_address, matrix_width, matrix_height, global_flags, flags_width);
        }

        CUTLASS_DEVICE
        CSSIteratorB(){}

        CUTLASS_DEVICE
        void load(Fragment &frag){
            for(int i=0; i<4; i++){
                if(add_itr.valid())
                    cutlass::arch::global_load<float,sizeof(float)>(frag[i], add_itr.get(), true);
                else
                    frag[i] = 0;
                add_itr++;
            }
            // for(int i=0; i<4; i++){
            //     cutlass::arch::global_load<float,sizeof(float)>(frag[i], add_itr.get(), add_itr.valid());
            //     add_itr++;
            // }
        }

};

template<typename SEMEShape_>
class CSSAddressIteratorA_B_base {
    public:
        using SEMEShape = SEMEShape_;

        float* address;
        int height;
        int width;
        int add_pos;
        int length;
        int* flags;
        int flag_base;
        int flag_pos;
        bool is_over;

        CUTLASS_DEVICE
        CSSAddressIteratorA_B_base(int blockIdx_x, int blockIdx_y, int threadId_x, float* global_address ,int matrix_width, int matrix_height, int* global_flags, int flags_width){
            this->flag_base = blockIdx_y*flags_width;
            this->length = global_flags[this->flag_base];
            this->flag_pos = threadId_x/SEMEShape::kM + 1;
            this->flags = global_flags;
            this->height = matrix_height;
            this->width = matrix_width;
            int add_pos = blockIdx_x*SEMEShape::kM + threadId_x%SEMEShape::kM;
            if (add_pos >= matrix_width){
                this->is_over = true;
                // printf("Cross the border: matrix_width=%d, threadIdx.x=%d.\n", matrix_width, threadId_x);
            }
            else{
                this->is_over = false;
            }
            //this->address = reinterpret_cast<float *>(global_address + add_pos*sizeof(float));
            this->address = global_address + add_pos;
        }

        CUTLASS_DEVICE
        CSSAddressIteratorA_B_base(){}

        CUTLASS_DEVICE
        CSSAddressIteratorA_B_base& operator++(){
            this->flag_pos+=2;
            return *this;
        }

        CUTLASS_DEVICE
        CSSAddressIteratorA_B_base operator++(int){
            CSSAddressIteratorA_B_base old = *this;
            operator++();
            return old;
        }

        CUTLASS_DEVICE
        int getLength(){
            return this->length;
        }

        CUTLASS_DEVICE
        float* get() const{
            return this->address + this->flags[this->flag_pos + this->flag_base]*this->width;
            //return this->address;
        }

        CUTLASS_DEVICE
        bool valid(){
            if(this->is_over)
                return false;
            if(this->flag_pos > this->length )
                return false;
            else
                return true;
        }
};

template<typename SEMEShape_>
class CSSIteratorA_B_base{
    public:
        using SEMEShape = SEMEShape_;
        using Fragment = cutlass::Array<float, 4>;
        using AddressIterator = CSSAddressIteratorA_B_base<SEMEShape>;

        AddressIterator add_itr;

        CUTLASS_DEVICE
        CSSIteratorA_B_base(int blockIdx_x, int blockIdx_y, int threadId_x, float* global_address ,int matrix_width, int matrix_height, int* global_flags, int flags_width){
            this->add_itr = AddressIterator(blockIdx_x, blockIdx_y, threadId_x, global_address, matrix_width, matrix_height, global_flags, flags_width);
        }

        CUTLASS_DEVICE
        CSSIteratorA_B_base(){}

        CUTLASS_DEVICE
        void load(Fragment &frag){
            for(int i=0; i<4; i++){
                if(add_itr.valid())
                    cutlass::arch::global_load<float,sizeof(float)>(frag[i], add_itr.get(), true);
                else
                    frag[i] = 0;
                add_itr++;
            }
            // for(int i=0; i<4; i++){
            //     cutlass::arch::global_load<float,sizeof(float)>(frag[i], add_itr.get(), add_itr.valid());
            //     add_itr++;
            // }
        }

        CUTLASS_DEVICE
        bool valid(){
            return this->add_itr.valid();
        }

        CUTLASS_DEVICE
        int getLength(){
            return this->add_itr.getLength();
        }


};


template<typename SEMEShape_>
class CSSAddressIteratorB_B_base {
    public:
        using SEMEShape = SEMEShape_;

        float* address;
        int height;
        int width;
        int add_pos;
        int length;
        int* flags;
        int flag_base;
        int flag_pos;
        bool is_over;

        CUTLASS_DEVICE
        CSSAddressIteratorB_B_base(int blockIdx_x, int blockIdx_y, int threadId_x, float* global_address ,int matrix_width, int matrix_height, int* global_flags, int flags_width){
            this->flag_base = blockIdx_y*flags_width;
            this->length = global_flags[this->flag_base];
            this->flag_pos = threadId_x/SEMEShape::kM + 1;
            this->flags = global_flags;
            this->height = matrix_height;
            this->width = matrix_width;
            int add_pos = blockIdx_y*SEMEShape::kM + threadId_x%SEMEShape::kM;
            if(add_pos >= matrix_width){
                this->is_over = true;
                // printf("Cross the border: threadIdx.x=%d.\n", threadId_x);
            } 
            else{
                this->is_over = false;
            }
            //this->address = reinterpret_cast<float *>(global_address + add_pos*sizeof(float));
            this->address = global_address + add_pos;
        }

        CUTLASS_DEVICE
        CSSAddressIteratorB_B_base(){}

        CUTLASS_DEVICE
        CSSAddressIteratorB_B_base& operator++(){
            this->flag_pos+=2;
            return *this;
        }

        CUTLASS_DEVICE
        CSSAddressIteratorB_B_base operator++(int){
            CSSAddressIteratorB_B_base old = *this;
            operator++();
            return old;
        }

        CUTLASS_DEVICE
        int getLength(){
            return this->length;
        }

        CUTLASS_DEVICE
        float* get() const{
            return this->address + this->flags[this->flag_pos + this->flag_base]*this->width;
            //return this->address;
        }

        CUTLASS_DEVICE
        bool valid(){
            if(this->is_over)
                return false;
            if(this->flag_pos > this->length )
                return false;
            else
                return true;
        }
};

template<typename SEMEShape_>
class CSSIteratorB_B_base{
    public:
        using SEMEShape = SEMEShape_;
        using Fragment = cutlass::Array<float, 4>;
        using AddressIterator = CSSAddressIteratorB_B_base<SEMEShape>;

        AddressIterator add_itr;

        CUTLASS_DEVICE
        CSSIteratorB_B_base(int blockIdx_x, int blockIdx_y, int threadId_x, float* global_address ,int matrix_width, int matrix_height, int* global_flags, int flags_width){
            this->add_itr = AddressIterator(blockIdx_x, blockIdx_y, threadId_x, global_address, matrix_width, matrix_height, global_flags, flags_width);
        }

        CUTLASS_DEVICE
        CSSIteratorB_B_base(){}

        CUTLASS_DEVICE
        void load(Fragment &frag){
            for(int i=0; i<4; i++){
                if(add_itr.valid())
                    cutlass::arch::global_load<float,sizeof(float)>(frag[i], add_itr.get(), true);
                else
                    frag[i] = 0;
                add_itr++;
            }
            // for(int i=0; i<4; i++){
            //     cutlass::arch::global_load<float,sizeof(float)>(frag[i], add_itr.get(), add_itr.valid());
            //     add_itr++;
            // }
        }

};


}
}
}