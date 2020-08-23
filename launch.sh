python2 ./extract_weights_and_compression.py    #extract weights compress and decompress
python2 ./reassemble_and_test.py 6  #test on accuracy degradation with different compression ratio
python2 ./reassemble_and_test.py 7
python2 ./reassemble_and_test.py 8
python2 ./optimize.py   #optimize compression config for each layer and reconstruct model
