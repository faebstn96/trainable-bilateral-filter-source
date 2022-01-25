#pragma once

#include <string>
#include <stdexcept>

#include <cuda_runtime_api.h>


// makes the error checking calls wait until all previous kernels finished. helpful to find the kernel that caused an error.
#ifndef NDEBUG
#define MCR_STRICT_DEBUGGING
#endif


inline void cuda_error_check_impl(cudaError err,
	const char *file, const int line)
{
	#ifdef MCR_STRICT_DEBUGGING
	// do not overwrite return codes that do not set the global error state
	if (err == cudaSuccess)
	{
		// wait for asynchronous errors
		err = cudaDeviceSynchronize();
	}
	#endif

	if (err != cudaSuccess)
	{
        auto full_message = std::string(file) + ", line " + std::to_string(line) + ": " + std::string("CUDA ERROR ") + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
        throw std::runtime_error(full_message); // set a breakpoint here in release builds to have the call stack available
	}
}
inline void cuda_error_check_impl(const std::string& errorMessage,
	const char *file, const int line)
{
	auto err = cudaPeekAtLastError();
	#ifdef MCR_STRICT_DEBUGGING
	if (err == cudaSuccess)
	{
		// wait for asynchronous errors
		err = cudaDeviceSynchronize();
	}
	#endif

	if (err != cudaSuccess)
	{
        auto full_message = std::string(file) + ", line " + std::string("CUDA ERROR ") + cudaGetErrorName(err) + ", " + errorMessage + ": " + cudaGetErrorString(err);
        throw std::runtime_error(full_message); // set a breakpoint here in release builds to have the call stack available
	}
}

#define cuda_error_check(msg_or_err_code) ::cuda_error_check_impl(msg_or_err_code,__FILE__,__LINE__)
