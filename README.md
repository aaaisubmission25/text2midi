# Text2MIDI: Generating Symbolic Music from Captions
This repository provides the code and pre-trained model for generating MIDI files from text descriptions, as proposed in our paper *Text2MIDI: Generating Symbolic Music from Captions*. Text2MIDI is an end-to-end model leveraging a large language model (LLM) encoder and a transformer-based autoregressive decoder to translate natural language captions into symbolic music (MIDI format). This approach aims to make music generation accessible to both musicians and non-technical users through descriptive text prompts.

## Installation Guide
To get started, clone the repository and set up the required environment.
```bash
git clone https://github.com/aaaisubmission25/text2midi.git
cd text2midi
conda create -n text2midi python=3.10
pip install -r requirements.txt
```
