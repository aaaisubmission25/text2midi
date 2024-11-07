# Text2MIDI: Generating Symbolic Music from Captions
This repository provides the code and additional results for generating MIDI files from text descriptions, as proposed in our paper *Text2MIDI: Generating Symbolic Music from Captions*. Text2MIDI is an end-to-end model leveraging a large language model (LLM) encoder and a transformer-based autoregressive decoder to translate natural language captions into symbolic music (MIDI format). This approach aims to make music generation accessible to both musicians and non-technical users through descriptive text prompts.

## Installation Guide
To get started, clone the repository and set up the required environment.
```bash
git clone https://github.com/aaaisubmission25/text2midi.git
cd text2midi
conda create -n text2midi python=3.10
pip install -r requirements.txt
```
## User Guide
To train our model, we use accelerate library. Please check/fix the confiugration in config/config.yaml. To train the model from scratch:
```bash
cd model
accelerate launch --multi_gpu --num_processes=4 train_accelerate.py --config ../config.yaml
```
To generate after training is finished: 
```bash
cd model
python transformer_model.py
```
## Results on Complete MidiCaps Test Set
After further analysis on the full MidiCaps test set, we found that Text2midi outperforms MuseCoco across **all** evaluated metrics. Below is a summary of these results:
| Metric      | Text2MIDI | MidiCaps | MuseCoco |
|-------------|-----------|----------|----------|
| CR (↑)      | 2.1560    | 3.4326   | 2.1214   |
| CLAP (↑)    | 0.2204    | 0.2593   | 0.2090   |
| TB (%) (↑)  | 34.03     | -        | 33.89    |
| TBT (%) (↑) | 66.90     | -        | 66.63    |
| CK (%) (↑)  | 15.36     | -        | 15.23    |
| CKD (%) (↑) | 15.80     | -        | 15.73    |



