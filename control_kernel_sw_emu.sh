git clone https://github.com/RhysHuo/Fused_Multicores_01.git
cd Fused_Multicores_01
cp kernelMatrixmult.cpp ..
cp kernelMatrixmult.h ..
cd ..
rm -rf Fused_Multicores_01
v++ -t sw_emu --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 -c -k kernelMult -o'kernelMult.sw_emu.xo' kernelMatrixmult.cpp kernelMatrixmult.h xcl2.hpp
v++ -t sw_emu --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 --link kernelMult.sw_emu.xo --config kernelMult.cfg -o'kernelMult.sw_emu.xclbin'
