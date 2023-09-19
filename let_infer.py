import subprocess, os
import json
from tqdm import tqdm
import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
import torch

DATA_PATH = "/workspace/checkpoint_nner/data/AGRIS_AZ0.json"
CHECKPOINT = "/workspace/epoch34.pth"



TMP_DATA = os.path.join(os.getcwd(), 'tmp.json')
FIXED_OUT = os.path.join(os.getcwd(), 'output', 'tmp_out')
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
    --fixed_output tmp_out \
    --infer_checkpoint {CHECKPOINT} \
    --infer {TMP_DATA}" 

def setupSpacy():
    nlp = spacy.load("en_core_web_sm")
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # ✅ Commented out regex that splits on hyphens between letters:
            # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer
    return nlp

def writeToFileAsJson(data, path):
    js_data = json.dumps(data, indent=4)
    with open(path, "w") as f:
        f.write(js_data)

def getEntsFromFile(infer_path):
    with open(infer_path, "r") as f:
            pred = f.readlines() 
    pr = pred[0]
    ents_pred = pr.split('|')
    ents = []
    for ent_pred in ents_pred:
        if ent_pred == '\n':
            continue
        l = int(ent_pred.split(',')[0])
        r = int((ent_pred.split(',')[1]).split(' ')[0])
        typ = ent_pred.split(' ')[1]
        typ = typ.replace('\n', '')
        ents.append({'posStart': l, 'posEnd': r,  'label': typ})
    return ents

def genDatasetForTriaffineFromCollectionThayHungFull(json_data):
    
    data = []
    nlp = setupSpacy()
    collection = []
    
    for article in json_data:
        
        sent = []
        for token in nlp(article['abstract']):
            sent.append((token.text, token.pos_))
            
        collection.append(sent)
    
    for id, abstract in tqdm(enumerate(collection)):
            
        sentJson = {}
        
        tokens = []
        entities = []
        relations = []
        org_id = []
        pos = []
        ltokens = []
        rtokens = []
            
        for token_, pos_ in abstract:
            tokens.append(token_)
            pos.append(pos_)
            
        sentJson['tokens'] = tokens
        sentJson['entities'] = entities
        sentJson['relations'] = relations 
        sentJson['org_id'] = org_id
        sentJson['pos'] = pos
        sentJson['ltokens'] = ltokens
        sentJson['rtokens'] = rtokens
        
        
        
        writeToFileAsJson(data=[sentJson], path=TMP_DATA)
        process = subprocess.run(CMD + ' --infer_id ' + str(id), shell=True, check=True)
        # process.wait()
        # process.kill()
        torch.cuda.empty_cache()
        
        infer_path = os.path.join(FIXED_OUT, f"infer_predict_{id}.txt")
        ents = getEntsFromFile(infer_path)
        for ent in ents:
            ent['text'] = sentJson['tokens'][ent['posStart']:ent['posEnd']]
        
        json_data[id]['tokens'] = sentJson['tokens']
        json_data[id]['sciEntity'] = ents
        
    return json_data

json_data = json.load(open(DATA_PATH))
json_data = genDatasetForTriaffineFromCollectionThayHungFull(json_data)
writeToFileAsJson(json_data, os.path.join(os.getcwd, 'final.json'))

