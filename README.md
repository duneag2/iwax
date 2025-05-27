# iWAX: Interpretable Wav2vec-AASIST-XGBoost Framework for Voice Spoofing Detection

## Usage
### 1. Train Wav2vec 2.0-AASIST Backbone
First, train the Wav2vec 2.0-AASIST model in the `aasist` directory. This step follows the training process of [AASIST GitHub](https://github.com/clovaai/aasist/)
Run the following command from the `aasist` directory:

`python main.py --config ./config/wav2vec2_aasist.conf`

Make sure the training completes successfully and note the directory where the trained model is saved. This will be used in the next step.

### 2. Train iWAX
Once the backbone model is trained, you can proceed to train the iWAX classifier:

`python iwax.py --save_dir /path/to/save/the/final/model -- w2v2 /path/to/w2v2/you/trained/in/step1`

`--save_dir`: Directory where the final iWAX model will be saved.

`--w2v2`: Path to the directory or checkpoint of the trained Wav2vec2-AASIST model.

### 3. Train iWAX with the Sinc Filter
To apply the Sinc filter before classification, use the `iwax_sinc.py` script:

`python iwax_sinc.py --low_freq 128 --high_freq 8000 --time 2 --save_dir /path/to/save/the/final/model -- w2v2 /path/to/w2v2/you/trained/in/step1`

`--low_freq`: Lower cutoff frequency (in Hz) for the Sinc filter.

`--high_freq`: Upper cutoff frequency (in Hz).

`--time`: `n` for `n/4`.
