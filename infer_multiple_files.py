import os 
import subprocess

DATA_FOLDER = '/workspace/checkpoint_nner/data/sagri/spec'

for i in os.listdir(DATA_FOLDER):
    
    data = os.path.join(DATA_FOLDER, i)
    CONFIG = f"DATA_PATH = '{data}'  \
        \nCHECKPOINT = '/workspace/epoch34.pth' \
        \nTMP_DATA = 'tmp_data'   # default \
        \nTMP_INFER = 'tmp_infer' # default \
        \nFINAL = 'final_result' # default \
        \nFINAL_NAME = '{i}' \
        \nINFER_BATCH_SIZE = 20 "
        
    f = open("pipeline_setup.py", "w")
    f.write(CONFIG)
    f.close()
    subprocess.run(['python', '1_convert_forward.py'], check=True)
    subprocess.run(['python', '2_run_infer.py'], check=True)
    subprocess.run(['python', '3_convert_backward.py'], check=True)
    
