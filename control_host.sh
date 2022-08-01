git clone https://github.com/RhysHuo/Fused_Multicores_01.git
cd Fused_Multicores_01
cp host.cpp ..
cp host.h ..
cd ..
rm -rf Fused_Multicores_01
g++ -g -std=c++14 -I$XILINX_XRT/include -L${XILINX_XRT}/lib/ -I/mnt/scratch/rhyhuo/HLS_arbitrary_Precision_Types/include -o host.exe host.cpp host.h -lOpenCL -pthread -lrt -lstdc++

