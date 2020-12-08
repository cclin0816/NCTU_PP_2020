all:
	g++ -mavx2 -O3 -o matmul main.cc matmul.cc -I/openmpi/include -pthread -Wl,-rpath -Wl,/openmpi/lib -Wl,--enable-new-dtags -L/openmpi/lib -lmpi -Wno-unused-result
clean:
	rm matmul 