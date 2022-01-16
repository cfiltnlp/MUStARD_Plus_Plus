# MUStARD++

This repository was created as part of our submission 'Emotion recognition in Sarcasm: A multimodal dataset creation and
evaluation' to LREC-2022

Our multimodal dataset consists of dialogs from sit-coms each of which is presented as a combination of the main 'utterance' and the 'context' in which it was uttered. There are 1202 instances (utterance+context) out of which 601 are sarcastic and 601 are non-sarcastic. Each utterance is annotated with the following information


| Column            |                   Description                         |
| -------------     | ----------------------------------------------------- |
| Sarcasm           | 0 or 1 indicating presence or absence of sarcasm      |
| Sarcasm_Type      | If sarcastic, indicates the type of sarcasm else None |
| Implicit_Emotion  | The hidden emotion associated with an instance        |
| Explicit_Emotion  | The surfact emotion associated with an instance       |
| Valence           | Level of pleasantness (1-9)                           |
| Arousal           | Level of perceived intensity  (1-9)                   |


The textual transcipts of this data with the associated annotations is made available in the form of a csv file uploaded in the repo. To access the videos associated with the utterances and their corresponding contexts, visit this [link](https://drive.google.com/drive/folders/1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW?usp=sharing).
