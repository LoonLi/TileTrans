#include <torch/extension.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "stdio.h"
#include <stdexcept>

#define NUM_STREAMS 4

int create_flag = 0;
const unsigned int workspace_size = 268435456;
float* workspace_A{nullptr};
float* workspace_B{nullptr};
float* workspace_C{nullptr};

unsigned int iDivUp(const unsigned int& a, const unsigned int& b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void initiate_workspace(){
    if(create_flag == 0){
      cudaMalloc(&workspace_A, workspace_size*sizeof(float));
      cudaMalloc(&workspace_B, workspace_size*sizeof(float));
      cudaMalloc(&workspace_C, workspace_size*sizeof(float));
      create_flag = 1;
    }
  }

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    int lda,
    float *B,
    int ldb,
    float beta,
    float *C,
    int ldc,
    float* D,
    int ldd,
    int* flags,
    int which_base,
    cudaStream_t stream = nullptr) {
  
  
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using RowMajor = cutlass::layout::RowMajor;
  
    using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                    ColumnMajor,  // Layout of A matrix
                                                    float,        // Data-type of B matrix
                                                    RowMajor,  // Layout of B matrix
                                                    float,        // Data-type of C matrix
                                                    RowMajor,     // Layout of C matrix
                                                    float>;  
    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {D, ldd},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

    
    float* wkp_A = A;
    float* wkp_B = B;
    cutlass::Status status = gemm_operator(args, flags, wkp_A, wkp_B, M, N, K, which_base, nullptr, stream);

  
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }

    return cudaSuccess;
}

torch::Tensor gen_flag(torch::Tensor matrix, int tile_width, int tile_height){
    int matrix_width = matrix.sizes()[1];
    int matrix_height = matrix.sizes()[0];
    
    int w_iteration = (matrix_width + tile_width - 1)/tile_width;
    int h_iteration = (matrix_height + tile_height - 1)/tile_height;
    torch::Tensor flag = torch::zeros({h_iteration, w_iteration+1}, torch::kInt);
    
    for(int i=0; i<h_iteration; i++){
        for(int j=0; j<w_iteration; j++){
            for(int l=0; l<tile_height; l++){
                if(i*tile_height+l >= matrix_height){
                    break;
                }
                for(int m=0; m<tile_width; m++){
                    if(j*tile_width+m >= matrix_width){
                        break;
                    }
                    if(*(matrix[i*tile_height+l][j*tile_width+m]!=0).data<bool>()){
                        flag[i][j] = 1;
                        break;
                    }
                }
                if(*(flag[i][j]==1).data<bool>()){
                    break;
                }
            }
        }
    }

    int tmp_pos_element;
    int longest = 0;

    int* flags = flag.data<int>();
    int flags_height = h_iteration;
    int flags_width = w_iteration;
  
    for(int i=0; i<flags_height; i++){  
        int tmp_next_element = flags[i*(flags_width + 1)];
        flags[i*(flags_width + 1)] = 0;
        
        for(int j=1; j<(flags_width+1); j++){
        tmp_pos_element = tmp_next_element;
        tmp_next_element = flags[i*(flags_width + 1) + j];
        if(tmp_pos_element != 0){
            flags[i*(flags_width + 1)]+=1;
            flags[i*(flags_width + 1) + flags[i*(flags_width + 1)]] = j-1;
            if(flags[i*(flags_width + 1)] > longest)
                longest = flags[i*(flags_width + 1)];
            }
        }
    }

    return flag;
}


torch::Tensor r2c(torch::Tensor matrix){
    int matrix_width = matrix.sizes()[1];
    int matrix_height = matrix.sizes()[0];
    float* matrix_base = matrix.data<float>();
    
    torch::Tensor matrix_c = torch::zeros({matrix_width, matrix_height}, torch::kFloat32);
    float* matrix_c_base = matrix_c.data<float>();

    for(int i=0; i<matrix_height; i++){
        for(int j=0; j<matrix_width; j++){
            matrix_c_base[i + j*matrix_height] = matrix_base[j + i*matrix_width];
        }
    }

    return matrix_c;
}


void matmul_with_flags(torch::Tensor A_c, torch::Tensor B, torch::Tensor C, torch::Tensor flags){
    int M = A_c.sizes()[1];
    int K = A_c.sizes()[0];
    int N = B.sizes()[1];
    int lda = M;
    int ldb = N;
    int ldc = N;
    float alpha = 1;
    float beta = 0;

    float* A_c_base = A_c.data<float>();
    float* B_base = B.data<float>();
    float* C_base = C.data<float>();
    int* flags_base = flags.data<int>();

    CutlassSgemmNN(M, N, K, alpha, A_c_base, lda, B_base, ldb, beta, C_base, ldc, workspace_C, ldc, flags_base, 0);
}


__global__ void matrix_transpose_d(float* matrix_in, int height, int width, float* matrix_out){
    for(int i = blockDim.x*blockIdx.x + threadIdx.x; i<height; i += blockDim.x*gridDim.x){
        for(int j=blockDim.y*blockIdx.y + threadIdx.y; j<width; j+= blockDim.y*gridDim.y){
            int in_pos = i*width + j;
            if(in_pos<height*width){
                int out_pos = j*height + i;
                matrix_out[out_pos] = matrix_in[in_pos];
            }
        }
    }
}

void matrix_transpose(torch::Tensor input, float* wkp){
    int height = input.sizes()[0];
    int width = input.sizes()[1];
    
    const dim3 block(32,32);
    const dim3 grid(iDivUp(height, 32), iDivUp(width, 32));
    matrix_transpose_d<<< grid, block >>>(input.data<float>(), height, width, wkp);
}

__global__ void build_linear_bias_matrix_d(float* bias_matrix, float* bias_vector, int output_dim, int batch_size){
    for(int i = blockDim.x*blockIdx.x + threadIdx.x; i<batch_size; i += blockDim.x*gridDim.x){
        for(int j=blockDim.y*blockIdx.y + threadIdx.y; j<output_dim; j+= blockDim.y*gridDim.y){
            if(i<batch_size && j<output_dim){
                int out_pos = i*output_dim + j;
                bias_matrix[out_pos] = bias_vector[j];
            }
        }
    }
}

void build_linear_bias_matrix(torch::Tensor bias, int batch_size, float* wkp){
    int output_dim = bias.sizes()[0];
    
    const dim3 block(1,32);
    const dim3 grid(batch_size, iDivUp(output_dim, 32));
    build_linear_bias_matrix_d<<< grid, block >>>(wkp, bias.data<float>(), output_dim, batch_size);
}

torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor flags){
    int M = input.sizes()[0];
    int K = input.sizes()[1];
    int N = weight.sizes()[0];
    int lda = M;
    int ldb = N;
    int ldc = N;
    float alpha = 1;
    float beta = 1;

    initiate_workspace();
    int* flags_base = flags.data<int>();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0).requires_grad(false);
    torch::Tensor output = torch::zeros({M, N}, options);

    matrix_transpose(input, workspace_A);
    matrix_transpose(weight, workspace_B);
    build_linear_bias_matrix(bias, M, workspace_C);

    CutlassSgemmNN(M, N, K, alpha, workspace_A, lda, workspace_B, ldb, beta, output.data<float>(), ldc, workspace_C, ldc, flags_base, 1);
    return output;
}


__global__ void im2col_d(
    float* input, 
    int input_dim, 
    int input_height, 
    int input_width, 
    float* output, 
    int output_height, 
    int output_width, 
    int filter_height, 
    int filter_width, 
    int stride, 
    int padding
){
    for(int row = blockIdx.x*blockDim.x + threadIdx.x; row<input_dim*filter_height*filter_width; row += blockDim.x * gridDim.x){
        if(row < input_dim*filter_height*filter_width){
            int k = row/(filter_height*filter_width);
            int m = (row - k*filter_height*filter_height)/filter_height;
            int n = row - k*filter_height*filter_width - m*filter_height;
            int i = (blockIdx.y*blockDim.y + threadIdx.y)/output_width;
            int j = (blockIdx.y*blockDim.y + threadIdx.y)%output_width;
            int outPos = row*output_height*output_width + blockIdx.y*blockDim.y + threadIdx.y;
            int inPos_height = i*stride + m;
            int inPos_width = j*stride + n;
            if( inPos_height < padding || inPos_height >= padding + input_height || inPos_width < padding || inPos_width >= padding + input_width ){
                output[outPos] = 0;
            }
            else{
                int inPos = k*input_height*input_width + (inPos_height-padding)*input_width + inPos_width - padding;
                output[outPos] = input[inPos];
            }
        }
    }
}


void im2col(
    float* input, 
    int input_dim, 
    int input_height, 
    int input_width, 
    float* output, 
    int output_height, 
    int output_width, 
    int filter_height, 
    int filter_width, 
    int stride, 
    int padding,
    cudaStream_t stream = nullptr){
  
    const dim3 block(1, 32);
    const dim3 grid(iDivUp(input_dim*filter_height*filter_width,1),iDivUp(output_height*output_width,32));
    //printf("(output_height, output_width)=(%d,%d)\n", output_height, output_width);
    im2col_d<<< grid , block, 0, stream>>>(input, input_dim, input_height, input_width, output, output_height, output_width, filter_height, filter_width, stride, padding);
  
}



torch::Tensor im2col_tensor(torch::Tensor input, int filter_height, int filter_width, int stride, int padding){
    int batch = input.sizes()[0];
    int in_dim = input.sizes()[1];
    int in_height = input.sizes()[2];
    int in_width = input.sizes()[3];
    
    int kernal_size = filter_height;
    int pad = kernal_size/2;

    int padded_height = padding*2 + 1*(in_height - 1) + 1;
    int padded_width = padding*2 + 1*(in_width - 1) + 1;

    int out_height = (padded_height - 2*pad - 1)/stride + 1;
    int out_width = (padded_width - 2*pad - 1)/stride + 1;
    torch::Tensor output = torch::zeros({in_dim*kernal_size*kernal_size, out_height*out_width}, torch::kFloat32).to(torch::kCUDA);
    
    im2col(input[0].data<float>(), in_dim, in_height, in_width, output.data<float>(), out_height, out_width, kernal_size, kernal_size, stride, padding);

    return output;
}

__global__ void built_bias_matix_d(float* bias, float* bias_matrix, int out_dim, int out_height, int out_width){
    int pos_h = blockIdx.x*blockDim.x + threadIdx.x;
    int pos_w = blockIdx.y*blockDim.y + threadIdx.y;
    if( pos_h<out_dim*out_height && pos_w<out_width ){
        int k = pos_h/out_height;
        int pos = pos_h*out_width + pos_w;
        bias_matrix[pos] = bias[k];
    }
}

void built_bias_matrix(float* bias, float* bias_matrix, int out_dim, int out_height, int out_width){
    dim3 block(32,32);
    dim3 grid(iDivUp(out_dim*out_height,32),iDivUp(out_width,32));
    built_bias_matix_d <<<grid, block>>> (bias, bias_matrix, out_dim, out_height, out_width);
}


torch::Tensor conv_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor weight_c, torch::Tensor flags, int stride, int padding){
    int batch = input.sizes()[0];
    int in_dim = input.sizes()[1];
    int in_height = input.sizes()[2];
    int in_width = input.sizes()[3];
    
    int kernal_size = weight.sizes()[2];
    int pad = kernal_size/2;
  
    int padded_height = padding*2 + 1*(in_height - 1) + 1;
    int padded_width = padding*2 + 1*(in_width - 1) + 1;
  
    int out_dim = weight.sizes()[0];
    int out_height = (padded_height - 2*pad - 1)/stride + 1;
    int out_width = (padded_width - 2*pad - 1)/stride + 1;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0).requires_grad(false);
    torch::Tensor output = torch::zeros({batch, out_dim, out_height, out_width}, options);
    
    initiate_workspace();

    const unsigned int M = out_dim;
    const unsigned int K = kernal_size*kernal_size*in_dim;
    const unsigned int N = out_height*out_width;
    const unsigned int lda = M;
    const unsigned int ldb = N;
    const unsigned int ldc = N;
    const unsigned int ldd = ldc;
    float alpha = 1;
    float beta = 1;
  
    //fi2col(weight, out_dim, in_dim, kernal_size, kernal_size, workspace_A);

    built_bias_matrix(bias.data<float>(), workspace_C, out_dim, out_height, out_width);
    for(int i=0; i<batch; i++){
      im2col(input[i].data<float>(), in_dim, in_height, in_width, workspace_B, out_height, out_width, kernal_size, kernal_size, stride, padding);
      CutlassSgemmNN(M, N, K, alpha, weight_c.data<float>(), lda, workspace_B, ldb, beta, output[i].data<float>(), ldc, workspace_C, ldd, flags.data<int>(), 0);
    }

    return output;
}

void freeMemory(){
    cudaFree(workspace_A);
    cudaFree(workspace_B);
    cudaFree(workspace_C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gen_flag", &gen_flag, "gen_flag");
    m.def("r2c", &r2c, "r2c");
    m.def("matmul_with_flags", &matmul_with_flags, "matmul_with_flags");
    m.def("im2col", &im2col_tensor, "im2col");
    m.def("free", &freeMemory, "free");
    m.def("conv_forward", &conv_forward, "conv_forward");
    m.def("linear_forward", &linear_forward, "linear_forward");
    m.def("matrix_transpose", &matrix_transpose, "matrix_transpose");
}