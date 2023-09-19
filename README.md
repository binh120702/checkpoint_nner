# Install dependencies

First, you need to install virtualenv:

```bash
$ pip install virtualenv
```

Then, create a virtual environment and go to it:

```bash
$ virtualenv nner
$ cd nner
```

Activate the virtual environment:

```bash
$ source bin/activate
```

Clone the repository:

```bash
$ git clone https://github.com/binh120702/checkpoint_nner
```

Go to the repository:

```bash
$ cd checkpoint_nner
```

Install build-essential:

```bash
$ sudo apt-get install build-essential -y
```

Install the dependencies:

```bash
$ pip install -r requirements.txt
$ python -m spacy download en_core_web_sm
```

Download the model checkpoint:

```bash
$ gdown 1AEO0Zek7_i1PZLpwGYeSrsd7HkbSl-Zv
```

The model checkpoint is now stored inside `checkpoint_nner`.

# Usage

## For inference/labeling data

First, put all the data you want to label inside a `data` folder. Then, replace the `DATA_FOLDER` and `CHECKPOINT_PATH` var in the code `infer_multiple_files.py` by the path to the `data` folder and *checkpoint* path.

Make sure that the `data` folder only contains the data you want to label. The data should be in the following *json* format:

```json
[
    {
        "abstract": "This is a sentence." ,
        ...
    },
    {
        "abstract": "This is another sentence." ,
        ...
    }
]
```

You can also modify the `infer_multiple_files.py` code to change the *batch size* (number of abstracts to label at once). The default values is 50.

Then, run the following command:

```bash
$ python infer_multiple_files.py
```

The output will be stored inside the `final_result` folder with the same name as the input file.
