default: conv

OBJS = main.o bmpfuncs.o serialConv.o helper.o hostFE.o

conv: $(OBJS)
	nvcc -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC -Wno-unused-result' -O3 -o $@ $(OBJS)

hostFE.o: hostFE.cu
	nvcc --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC -Wno-unused-result' -O3 -c $< -o $@

%.o: %.c
	nvcc --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC -Wno-unused-result' -O3 -c $< -o $@

clean:
	rm -f conv *.o output.bmp ref.bmp