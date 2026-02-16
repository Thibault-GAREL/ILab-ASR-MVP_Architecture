# MVP ASR Architecture

The goal is to find the best architecture for the best ASR!

We want to reach: **Efficiency**, **Quickness** and **Frugality**!

Here is the model router architecture idea :

<p align="center">
  <img src="img/Model_Router_V2.png" alt="Model_Router_V2" style="border-radius:8px">
</p>

<details>
<summary>See the old architecture idea</summary>
<p align="center">
  <img src="img/Model_Router_V3.png" alt="model_router_architecture" style="border-radius:8px">
</p>

</details>

---

## Router

The goal is to create a Neural Network able to choose the n best models for a specific batch :

### Inputs :

#### The audio batch :

To link with the Neural Network, we can have some problemes : different resolutions and durations.

Different strategy is possible to solve it :

1. For the resolution :

- Choose a unique frequence (16 kHz) and resample all the audio to this value. But the router will learn some structural noize.

2. For the duration :

- Padding (shorter audio / add some 0 or noize) / Troncation (longer audio / cut) ==> simple but info lost / the duration are similar
- Window cut (cut audio in fix frame with overlaps) ==> cut during a word ?
- Time/Frequency features (STFT, Log-Mel spectrogram, MFCC) to have a time \* frequency image. ==> imposed a human vision + little info lost
- Global pooling (any cut) ==> crushed the temporal structure

My opinion : Log-Mel spectrogram + CNN + global or attention pulling and fixing the time of each batch with a big overlaps and make a ROVER during this timing

#### Some metadata :

##### Needed :

- **SNR** (Signal-to-Noise Ratio)
- **Duration** of the batch

##### Need to discuss with the group :

- **Languages** (if provided and see how to map)
- **Resolution** (8Hz (phones) to 44.1 kHz (studio)) ==> + or - robust models

##### Possible :

- **Bit Depth** : (ex: 16-bit, 24-bit) ==> sound dynamics
- **Field / theme** : (ex: medical, legal, conversation) ==> Simplify the process for the router
- **Duration** : be less energy-consumption (batch or total ?)

#### Idea for the overlaps

For the overlaps, I got an idea :

<p align="center">
  <img src="img/ROVER_during_the_overlaps.png" alt="ROVER_during_the_overlaps" style="border-radius:8px">
</p>

#### Add a confiance base in the model router

### Outputs

![Inside the router](/img/inside_the_router.png)
Let's dive inside the router.

We thought of a 2 step classification _(method and model selection)_ processed simultaneously.
On one side, the router will select the most suitable method based on the input audio batch, with softmax distribution.

On the other side, the router will select one or multiple models from the entire pool of models based on the choosen method :

- if a method requiring multiple models _(Corrector, Confidence Based Ensemble, Rover, MoE, CoE)_, the router will pick a selection of $n$ models that will be used to process the input batch together.
- if the router choose a single model method, then it will select the more suitable model from the models pool, to process the input batch.

---

## Standardisation for each model

For each model, there is a `model_type_name.py` in the folder "models" (either an API or a local processing) that creates the standardised JSON as output.

For each model, make a function **"infer"** to compute the model and to create the output.

---

## Standardisation for the **JSON** output of an STT model

```bash
Batch :
- Batch ID (or the path + name)
- Info on the batch (language, total duration of the original audio, context…)
- Text transcription of the batch
- Timestamp for the start and the end of the batch

Model :
- ASR Model Name
- Size Model / Category
- Diarisation or not (bool) ==> Deleting or not "Speaker 0" for the WER test
- Compute duration
- Output: Text transcription of the model
```

<details>
<summary>See what it looks like</summary>

```bash
{
  "batch": {
    "batch_id": "batch_001",
    "info": {
      "language": "English",
      "total_duration_seconds": 360,
      "context": "Meeting recording with multiple speakers"
    },
    "true_transcription": "Hello everyone",
    "timestamps": {
      "start_sec": 0,
      "end_sec": 34
    }
  },
  "model": {
    "asr_model_name": "Whisper-XL",
    "size_category": 0 or 1, "small/large"
    "diarisation": true,
    "compute_duration_seconds": 45,
    "output_transcription": "Hello everyone",
    "segments": [
          {
            "segment_id": 1,
            "speaker": "Speaker 1",
            "start_sec": 0.0,
            "end_sec": 12.4,
            "text": "Good morning everyone."
          },
          {
            "segment_id": 2,
            "speaker": "Speaker 2",
            "start_sec": 12.5,
            "end_sec": 28.9,
            "text": "Today we will review the quarterly financial results."
          }
        ]
  }
}
```

</details>

## WER test

Here is a link for the [WER test](https://huggingface.co/spaces/evaluate-metric/wer) !

## Optimisation of the architecture (after)

- Word Boosting
- MoE
- FlashAttention
- vLLM
- Distillation
- Quantization
- Zipformer
- Pruning
- Merging
- Steering
- Parallelization of the batch, frequency...
- Diarisation in parallel

---

## Installation

### Prerequisites

- Python 3.8 or higher

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Thibault-GAREL/Rover_architecture.git
   cd Rover_architecture
   code .
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env with your own values
   # Update API keys, model paths, and other configuration as needed
   ```

4. **Run the application**
   ```bash
   python src/main.py
   ```

---

## 🚀 MVP Implementation - ROVER System

### What is ROVER?

ROVER (Recognizer Output Voting Error Reduction) is a technique that combines multiple speech recognition models to produce more accurate transcriptions through intelligent voting. Instead of relying on a single model, ROVER aligns the outputs of multiple models at the word level and selects the best word based on confidence scores.

### Current Implementation

A functional MVP has been implemented with:

- **Base Model Infrastructure**: Standardized interface for ASR models with JSON output format
- **Whisper Models**: Support for multiple Whisper variants (tiny, base, small, medium, large-v3)
- **ROVER Voting System**: Word-level alignment and confidence-weighted voting
- **Automatic Deduplication**: Removes repetitions and cleans output text

### Quick Start

**Test a single Whisper model:**

```bash
python models/example_whisper.py
```

**Test ROVER with 3 models:**

```bash
python rover/example_rover.py
```

### Performance Example

Testing with 3 Whisper models on 22s of French audio:

| Model     | Time  | Confidence | Agreement |
| --------- | ----- | ---------- | --------- |
| Tiny      | 1.2s  | 76.6%      | -         |
| Base      | 2.0s  | 82.4%      | -         |
| Small     | 5.3s  | 91.2%      | -         |
| **ROVER** | **-** | **~87%**   | **72.2%** |

<!--
Add confiance in the Router

-->
