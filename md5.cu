#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Define the s array and K constants
__constant__ uint32_t s[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

__constant__ uint32_t K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
    0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
    0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
    0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
    0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
    0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};

__device__ uint32_t leftrotate(uint32_t x, uint32_t c) {
    return (x << c) | (x >> (32 - c));
}

__global__ void md5_kernel(uint8_t *d_msg, size_t msg_len, uint8_t *d_digest) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 64 >= msg_len)
        return;

    uint8_t *msg_chunk = d_msg + idx * 64;

    // Process the message in 512-bit chunks
    uint32_t *chunk = (uint32_t *)msg_chunk;
    uint32_t digest_state[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};

    uint32_t A = digest_state[0];
    uint32_t B = digest_state[1];
    uint32_t C = digest_state[2];
    uint32_t D = digest_state[3];

    for (int j = 0; j < 64; j++) {
        uint32_t F, g;
        if (j < 16) {
            F = (B & C) | ((~B) & D);
            g = j;
        } else if (j < 32) {
            F = (D & B) | ((~D) & C);
            g = (5 * j + 1) % 16;
        } else if (j < 48) {
            F = B ^ C ^ D;
            g = (3 * j + 5) % 16;
        } else {
            F = C ^ (B | (~D));
            g = (7 * j) % 16;
        }

        F = F + A + K[j] + chunk[g];
        A = D;
        D = C;
        C = B;
        B = B + leftrotate(F, s[j]);
    }

    digest_state[0] += A;
    digest_state[1] += B;
    digest_state[2] += C;
    digest_state[3] += D;

    // Write the results to global memory
    for (int i = 0; i < 4; i++) {
        d_digest[idx * 16 + i * 4] = digest_state[i] & 0xFF;
        d_digest[idx * 16 + i * 4 + 1] = (digest_state[i] >> 8) & 0xFF;
        d_digest[idx * 16 + i * 4 + 2] = (digest_state[i] >> 16) & 0xFF;
        d_digest[idx * 16 + i * 4 + 3] = (digest_state[i] >> 24) & 0xFF;
    }
}

void md5(uint8_t *msg, size_t len, uint8_t *digest) {
    // Allocate memory on the GPU
    uint8_t *d_msg, *d_digest;
    size_t padded_len = (len + 1 + 8 + 63) & ~63;
    cudaError_t err;

    err = cudaMalloc(&d_msg, padded_len);
    if (err != cudaSuccess) {
        printf("CUDA malloc d_msg failed: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc(&d_digest, ((padded_len + 63) / 64) * 16);
    if (err != cudaSuccess) {
        printf("CUDA malloc d_digest failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Prepare the message
    uint8_t *padded_msg = (uint8_t *)malloc(padded_len);
    memcpy(padded_msg, msg, len);
    padded_msg[len] = 0x80;
    memset(padded_msg + len + 1, 0, padded_len - len - 1 - 8);
    uint64_t bit_len = len * 8;
    memcpy(padded_msg + padded_len - 8, &bit_len, 8);

    // Copy the message to the GPU
    err = cudaMemcpy(d_msg, padded_msg, padded_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy to device failed: %s\n", cudaGetErrorString(err));
        return;
    }
    free(padded_msg);

    // Launch the kernel with enough threads to cover the entire message
    int blockSize = 256;
    int numBlocks = (padded_len / 64 + blockSize - 1) / blockSize;
    md5_kernel<<<numBlocks, blockSize>>>(d_msg, padded_len, d_digest);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Copy the digest back to the host
    err = cudaMemcpy(digest, d_digest, ((padded_len + 63) / 64) * 16,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA memcpy to host failed: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaFree(d_msg);
    cudaFree(d_digest);
}

int main(int argc, char *argv[]) {
    uint8_t *initial_msg;
    size_t initial_len;

    if (argc > 1) {
        initial_msg = (uint8_t *)argv[1];
        initial_len = strlen((char *)initial_msg);
    } else {
        printf("Abort. Please specify a input\n");
        return 1;
    }

    uint8_t digest[16];
    md5(initial_msg, initial_len, digest);

    for (int i = 0; i < 16; i++) {
        printf("%02x", digest[i]);
    }
    printf("\n");
    return 0;
}
