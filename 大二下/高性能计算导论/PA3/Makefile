# DO NOT MODIFY THIS FILE

benchmark: apsp.h cuda_utils.h apsp.cu apsp_ref.cu benchmark.cu
	# GTX 1080 = 61, Tesla P100 = 60
	nvcc -gencode arch=compute_60,code=sm_60 \
		 -gencode arch=compute_61,code=sm_61 \
		 -Xcompiler -Wall,-g,-O3 -Xptxas -O3 -o $@ apsp.cu apsp_ref.cu benchmark.cu
