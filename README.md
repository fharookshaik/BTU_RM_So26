# BTU Coursework - Research Module - SoSe26

FG Graphische Systeme \
Prof. Dr. Douglas W. Cunningham \
Mentor: Prashant Varadarajan

**Topic:  Structural Segmentation with Transformers (Extension 2)**

Paper:  [To Catch A Chorus, Verse, Intro, or Anything Else Analyzing a Song with Structural Functions](/Papers/Wang%20et%20al.%20-%202022%20-%20To%20Catch%20A%20Chorus,%20Verse,%20Intro,%20or%20Anything%20Else%20Analyzing%20a%20Song%20with%20Structural%20Functions.pdf)
- This paper adapts SpecTNT (Spectrogram Transformer with interleaved Time–Frequency attention) for structural function classification (e.g., intro, verse, chorus, bridge) from spectrogram input.


**Course Objectives:**

- Reimplement the SpecTNT baseline for functional structure analysis.
- Train and evaluate on a structural dataset (e.g., SALAMI, RWC-Pop).
- Extend the system by applying a temporal smoothing decoder, such as a Viterbi algorithm or simple transition-penalty smoothing, on top of the model’s framewise predictions.
- Keep the model architecture unchanged so that the only modification is post-processing.
- Compare baseline vs. smoothed predictions using macro F1 and structural segmentation metrics, and analyse whether smoothing improves temporal consistency of predicted sections.

---

Datasets Used: 

1. SALAMI-pop
2. Harmonix Set
3. RWC-pop
4. Isophonics (MIREX 2009 Collection)

---
**Harmonix Dataset(Nieto et al., ISMIR 2019)**

Dataset Link: https://github.com/urinieto/harmonixset

- **Size:** 912 Western pop songs.
- **Annotations:** Beats, downbeats, and functional segment annotations (the main reason it is used here).
- **Special handling:**
  - Original audio not publicly released with the dataset.
  - Authors searched for matching audio versions themselves and manually refined the annotations.

- **Usage in the paper:**
  - Primary dataset for the 4-fold cross-validation ablation study.
  - Always included in the training set during all cross-dataset experiments.

- **Implementation note:** You will need to source the audio yourself (YouTube, Spotify, etc.) and align it with the provided annotations. This is the largest and most “modern” pop dataset used.

### TODO:

- [ ] Prepare Dataset
- [ ] 