#include <torch/extension.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include"stdio.h"

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

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}
  
/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}


/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  float* D,
  int ldd) {


  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  cutlass::layout::RowMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  cutlass::layout::RowMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  cutlass::layout::RowMajor,  // Layout of C matrix
                                                  float>;

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {D, ldd},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //
  
  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
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
  int padding){

  const dim3 block(1, 32);
  const dim3 grid(iDivUp(input_dim*filter_height*filter_width,1),iDivUp(output_height*output_width,32));
  //printf("(output_height, output_width)=(%d,%d)\n", output_height, output_width);
  im2col_d<<< grid , block>>>(input, input_dim, input_height, input_width, output, output_height, output_width, filter_height, filter_width, stride, padding);

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

torch::Tensor conv_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding){
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
    const unsigned int lda = K;
    const unsigned int ldb = N;
    const unsigned int ldc = N;
    const unsigned int ldd = ldc;
    float alpha = 1;
    float beta = 1;

    built_bias_matrix(bias.data<float>(), workspace_C, out_dim, out_height, out_width);
    for(int i=0; i<batch; i++){
      im2col(input[i].data<float>(), in_dim, in_height, in_width, workspace_B, out_height, out_width, kernal_size, kernal_size, stride, padding);
      CutlassSgemmNN(M, N, K, alpha, weight.data<float>(), lda, workspace_B, ldb, beta, output[i].data<float>(), ldc, workspace_C, ldd);
    }
  
    return output;
}

void freeMemory(){
    cudaFree(workspace_A);
    cudaFree(workspace_B);
    cudaFree(workspace_C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("free", &freeMemory, "free");
    m.def("conv_forward", &conv_forward, "conv_forward");
}