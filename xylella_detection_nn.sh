#! /bin/bash

# run -c 1 -m 5 -t 35:30 -o Out/run_models.out -e Error/run_models.err "python run_models.py"
# run -c 1 -m 5 -t 0:30 -o Out/code_cnn.out -e Error/code_cnn.err "python xylella_detection_cnn.py"

# run -c 1 -m 5 -t 00:30 -o Out/code.out -e Error/code.err "python xylella_detection_nn.py --cost_sensitive"
# run -c 1 -m 5 -t 35:30 -o Out/code.out -e Error/code.err "python xylella_detection_nn.py --dropout"
# run -c 1 -m 5 -t 35:30 -o Out/code.out -e Error/code.err "python xylella_detection_nn.py --L2_regularizer"

# run -c 1 -m 5 -t 0:30 -o Out/code.out -e Error/code.err "python xylella_detection_nn.py --oversampling"
# run -c 1 -m 5 -t 35:30 -o Out/code.out -e Error/code.err "python xylella_detection_nn.py --oversampling --dropout"
# run -c 1 -m 5 -t 35:30 -o Out/code.out -e Error/code.err "python xylella_detection_nn.py --oversampling --L2_regularizer"

# run -c 1 -m 5 -t 15:30 -o Out/code.out -e Error/code.err "python xylella_detection_nn.py --undersampling"
# run -c 1 -m 5 -t 15:30 -o Out/code_small.out -e Error/code_small.err "python xylella_detection_nn.py --oversampling --dropout"
# run -c 1 -m 5 -t 15:30 -o Out/code.out -e Error/code.err "python xylella_detection_nn.py --undersampling --L2_regularizer"

# run -c 1 -m 5 -t 15:30 -o Out/code.out -e Error/code.err "python xylella_detection_nn.py --oversampling --undersampling"
# run -c 1 -m 5 -t 15:30 -o Out/code.out -e Error/code.err "python xylella_detection_nn.py --oversampling --undersampling --dropout"
# run -c 1 -m 5 -t 15:30 -o Out/code.out -e Error/code.err "python xylella_detection_nn.py --oversampling --undersampling --L2_regularizer"





# run -c 1 -m 5 -t 0:10 -e Error/results_summary.err "python results_summary.py"
# run -c 1 -m 50 -t 0:10 -o Out/gpu.out -e Error/gpu.err "python gpu_test.py"

# run -g 1 -m 24 -t 1:30 -o Out/code_gpu.out -e Error/code_gpu.err "python xylella_detection_nn.py --oversampling"
