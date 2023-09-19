import subprocess, os
import json
from tqdm import tqdm
import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from pipeline_setup import CHECKPOINT, DATA_PATH, TMP_DATA, INFER_BATCH_SIZE

TMP_DATA_FOLDER = os.path.join(os.getcwd(), TMP_DATA)

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

def genDatasetForTriaffine(json_data):
    
    nlp = setupSpacy()
    collection = []
    
    for article in tqdm(json_data):
        
        sents = []
        sent = []
        for token in nlp(article['abstract']):
            sent.append((token.text, token.pos_))
            if token.text == '.':
                sents.append(sent)
                sent = []
        collection.append(sents)
    
    sents = []
    pushed = 0
    batch_num = 0
    for id, abstract in enumerate(collection):
        for sent in abstract:
            
            sentJson = {}
                
            tokens = []
            entities = []
            relations = []
            org_id = []
            pos = []
            ltokens = []
            rtokens = []
                
            for token_, pos_ in sent:
                tokens.append(token_)
                pos.append(pos_)
                
            sentJson['tokens'] = tokens
            sentJson['entities'] = entities
            sentJson['relations'] = relations 
            sentJson['org_id'] = org_id
            sentJson['pos'] = pos
            sentJson['ltokens'] = ltokens
            sentJson['rtokens'] = rtokens
            
            
            if len(tokens) > 60:
                sentJson['tokens'] = ["."]
                sentJson['pos'] = ["PUNCT"]
                
            
            sents.append(sentJson)
            
        pushed += 1
        if pushed == INFER_BATCH_SIZE or id + 1 == len(collection):
            writeToFileAsJson(data=sents, path=os.path.join(TMP_DATA_FOLDER, str(batch_num) + '.json'))
            sents = []
            pushed = 0
            batch_num += 1
    return 

if __name__ == "__main__":
    try:
        os.system(f"rm -rf {TMP_DATA_FOLDER}")
    except BaseException:
        pass
    try:
        os.mkdir(TMP_DATA_FOLDER)
    except BaseException:
        pass
    json_data = json.load(open(DATA_PATH))
    genDatasetForTriaffine(json_data)

