import subprocess, os
import json
from tqdm import tqdm
import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from pipeline_setup import CHECKPOINT, DATA_PATH, TMP_DATA, TMP_INFER, INFER_BATCH_SIZE, FINAL, FINAL_NAME

TMP_INFER_FOLDER = os.path.join(os.getcwd(), TMP_INFER)
FINAL_RESULT_PATH = os.path.join(os.getcwd(), FINAL)

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
    ent_per_line = []
    for id in range(len(pred)):
        pr = pred[id]
        ents_pred = pr.split('|')
        ents = []
        for ent_pred in ents_pred:
            if ent_pred == '\n':
                continue
            l = int(ent_pred.split(',')[0])
            r = int((ent_pred.split(',')[1]).split(' ')[0])
            typ = ent_pred.split(' ')[1]
            typ = typ.replace('\n', '')
            ents.append({'l': l, 'r': r,  'label': typ})
        ent_per_line.append(ents)
    return ent_per_line

def convertDataBackward(json_data):
    
    nlp = setupSpacy()
    collection = []
    
    for article in json_data:
        
        sents = []
        sent = []
        for token in nlp(article['abstract']):
            sent.append((token.text, token.pos_))
            if token.text == '.':
                sents.append(sent)
                sent = []
            
        collection.append(sents)
    
    N = len(json_data)
    total_batch = (N + INFER_BATCH_SIZE - 1)  // INFER_BATCH_SIZE
    
    sents = []
    pushed = 0
    batch_num = 0
    line = 0
    ent_per_line = getEntsFromFile(os.path.join(TMP_INFER_FOLDER, 'infer_predict_0.txt'))
    
    for id, abstract in tqdm(enumerate(collection)):
        abs_tok = []
        abs_ents = []
        for sent in abstract:
            
            tokens = []
            for token_, pos_ in sent:
                tokens.append(token_)
                
            
            pre_len = len(abs_tok)
            for ent in ent_per_line[line]:
                abs_ents.append({
                    'posStart': pre_len + ent['l'],
                    'posEnd': pre_len + ent['r'],
                    'text': ' '.join(tokens[ent['l']:ent['r']]),
                    'label': ent['label']
                })
            
            abs_tok += tokens
            line += 1
        
        json_data[id]['tokens'] = abs_tok
        json_data[id]['sciEntity'] = abs_ents 
        
        pushed += 1
        if pushed == INFER_BATCH_SIZE:
            pushed = 0
            batch_num += 1
            line = 0
            ent_per_line = getEntsFromFile(os.path.join(TMP_INFER_FOLDER,'infer_predict_' +  str(batch_num) + '.txt'))
    return

if __name__ == "__main__":
    json_data = json.load(open(DATA_PATH))
    convertDataBackward(json_data)
    try:
        os.mkdir(FINAL_RESULT_PATH)
    except BaseException:
        pass
    writeToFileAsJson(json_data, os.path.join(FINAL_RESULT_PATH, FINAL_NAME))

