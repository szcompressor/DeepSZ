#for SZ compression and decompression script
# 1st arg should be lenth of the data array
# 2ed arg should be layer number

import os
import sys

folder = os.path.exists("./SZ_compress_script")
if not folder:
    os.makedirs("./SZ_compress_script")

bash_line = "./SZ_compress_script/fc" + str(sys.argv[2]) + "_script.sh"
output = open(bash_line, "w")

wr_line = "echo start > ./data/compression_ratios_fc" + str(sys.argv[2]) + ".txt\n"
output.writelines(wr_line)

for i in range(1,10):
    wr_line_1 = "./SZ/build/bin/sz -z -f -c ./SZ/example/sz.config -M ABS -A " + str(i) + "E-3 -i ./data/fc" + str(sys.argv[2]) +"-data-o.dat -1 " + str(sys.argv[1]) + "\n"
    wr_line_2 = "./SZ/build/bin/sz -x -f -s ./data/fc" + str(sys.argv[2]) + "-data-o.dat.sz -1 " + str(sys.argv[1]) + "\n"
    wr_line_3 = "mv -f ./data/fc" + str(sys.argv[2]) + "-data-o.dat.sz.out ./data/fc" + str(sys.argv[2]) + "-data-" +str(i) + "E-3.dat\n"
    wr_line_4 = "echo " + str(i) + "E-3 >> ./data/compression_ratios_fc" + str(sys.argv[2]) + ".txt\n"
    wr_line_5 = "wc -c ./data/fc" + str(sys.argv[2]) + "-data-o.dat.sz >> ./data/compression_ratios_fc" + str(sys.argv[2]) + ".txt\n"
    output.writelines(wr_line_1)
    output.writelines(wr_line_2)
    output.writelines(wr_line_3)
    output.writelines(wr_line_4)
    output.writelines(wr_line_5)
