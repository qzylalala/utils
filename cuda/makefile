sgemm:
	nvcc sgemm.cu -o sgemm --generate-line-info -arch=compute_86 -code=sm_86 -lcublas -lcurand --ptxas-options=-v
	cuobjdump -ptx sgemm > sgemm.ptx
	cuobjdump -sass sgemm > sgemm.sass

sgemm_tensorcore:
	nvcc sgemm_tensorcore.cu -o sgemm_tensorcore -arch=compute_86 -code=sm_86 -lcublas -lcurand --ptxas-options=-v

clean:
	rm -f sgemm sgemm_tensorcore *.sass *.ptx