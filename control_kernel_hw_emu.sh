git clone https://github.com/RhysHuo/Fused_Multicores_01.git
cd Fused_Multicores_01
cp kernelMatrixmult.cpp ..
cp kernelMatrixmult.h ..
cd ..
rm -rf Fused_Multicores_01
v++ -t hw_emu --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 -c -k mmult_top -o'kernelmult.hw_emu.xo' kernelMatrixmult.cpp kernelMatrixmult.h xcl2.hpp
v++ -t hw_emu --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 --link kernelmult.hw_emu.xo --config mmult_top.cfg -o'kernelmult.hw_emu.xclbin'
