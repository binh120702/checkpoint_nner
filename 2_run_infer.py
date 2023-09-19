import subprocess, os
import json
from tqdm import tqdm
import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from pipeline_setup import CHECKPOINT, DATA_PATH, TMP_DATA, TMP_INFER, INFER_BATCH_SIZE

TMP_DATA_FOLDER = os.path.join(os.getcwd(), TMP_DATA)
TMP_INFER_FOLDER = os.path.join(os.getcwd(), TMP_INFER)

def run(json_data):
    
    N = len(json_data)
    it = (N + INFER_BATCH_SIZE - 1)  // INFER_BATCH_SIZE
    for id in range(it):
        data_i = os.path.join(TMP_DATA_FOLDER, str(id) + '.json')
        
        CMD = f"CUDA_VISIBLE_DEVICES=0 python main.py \
        --version sagri \
        --model SpanAttModelV3 \
        --bert_name_or_path dmis-lab/biobert-base-cased-v1.1 \
        --learning_rate 3e-5 \
        --batch_size 1 \
        --gradient_accumulation_steps 128 \
        --train_epoch 41 \
        --score tri_affine \
        --truncate_length 192 \
        --word \
        --word_dp 0.2 \
        --char \
        --pos \
        --use_context \
        --warmup_ratio 0.0 \
        --att_dim 200 \
        --bert_before_lstm \
        --lstm_dim 1024 \
        --lstm_layer 2 \
        --encoder_learning_rate 5e-4 \
        --max_span_count 30 \
        --share_parser \
        --subword_aggr max \
        --init_std 1e-2 \
        --dp 0.2 \
        --fixed_output {TMP_INFER_FOLDER} \
        --infer_checkpoint {CHECKPOINT} \
        --infer {data_i} \
        --infer_id {id}" 
        
        subprocess.run(CMD, shell=True, check=True)

if __name__ == "__main__":
    try:
        os.system(f"rm -rf {TMP_INFER_FOLDER}")
    except BaseException:
        pass
    json_data = json.load(open(DATA_PATH))
    run(json_data)

