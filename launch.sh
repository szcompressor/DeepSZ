source ./build_SZ.sh

python ./extract_weights_and_compression.py    #extract weights compress and decompress
python ./reassemble_and_test.py 6  #test on accuracy degradation with different compression ratio
python ./reassemble_and_test.py 7
python ./reassemble_and_test.py 8
python ./optimize.py   #optimize compression config for each layer and reconstruct model
