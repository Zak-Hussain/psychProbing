## psychProbing

This repository contains the code for the paper:

Hussain, Z., Mata, R., Newell, Ben R., & Wulff, D. U. (2024). Probing the contents of text, brain, and behavior data toward improving human-LLM alignment. *arXiv preprint arXiv:XXXXXX*.

```
@article{hussain2024probing,
  title={Probing the contents of text, brain, and behavior data toward improving human-LLM alignment},
  author={Zak Hussain and Rui Mata and Ben R. Newell and Dirk U. Wulff},
  journal={arXiv},
  year={2024}
  url={https://arxiv.org/XX}
}
```

### Representations

In order to run the code, you will need to download the representations from the [here](XX) and place them in 
`data/raw/embeds` (they are too large to share on GitHub unfortunately). The original sources of the representations are as follows:

**Text**: 
- [`CBOW_GoogleNews`](https://code.google.com/archive/p/word2vec/) ('GoogleNews-vectors-negative300.bin.gz') 
- [`fastText_CommonCrawl`](https://fasttext.cc/docs/en/english-vectors.html) ('crawl-300d-2M.vec.zip')
- [`fastText_Wiki_News`](https://fasttext.cc/docs/en/english-vectors.html)('wiki-news-300d-1M.vec.zip)
- [`fastTextSub_OpenSub`](https://github.com/jvparidon/subs2vec/) ('English, en, OpenSubtitles')
- [`GloVe_CommonCrawl`](https://nlp.stanford.edu/projects/glove/) ('glove.840B.300d.zip')
- [`GloVe_Twitter`](https://nlp.stanford.edu/projects/glove/) ('glove.twitter.27B.zip')
- [`GloVe_Wikipedia`](https://nlp.stanford.edu/projects/glove/) ('glove.6B.zip')
- [`LexVec_CommonCrawl`](https://github.com/alexandres/lexvec) ('Word Vectors (2.2GB)')
- [`morphoNLM`](https://nlp.stanford.edu/~lmthang/morphoNLM/) ('HSMN+csmRNN')
- [`spherical_text_Wikipedia`](https://github.com/yumeng5/Spherical-Text-Embedding) ('300-d')

**Brain**:
- [`microarray`](https://figshare.com/s/94962977e0cc8b405ef3) ('results/tungsten/word_projections.pickle')
- [`EEG_speech`](https://github.com/DS3Lab/cognival)('cognival-vectors/eeg_speech/naturalspeech_scaled.txt')
- [`EEG_text`](https://github.com/DS3Lab/cognival)('cognival-vectors/eeg_text/zuco_scaled.txt')
- [`fMRI_speech_hyper_align`](https://github.com/DS3Lab/cognival)('cognival-vectors/fmri/harry-potter/1000-random-voxels/', further processed with ['hyper alignment'](https://hypertools.readthedocs.io/en/latest/hypertools.align.html)) 
- [`fMRI_text_hyper_align`](https://github.com/DS3Lab/cognival)('cognival-vectors/fmri/alice/', further processed with ['hyper alignment'](https://hypertools.readthedocs.io/en/latest/hypertools.align.html))
- [`eye_tracking`](https://github.com/DS3Lab/cognival)('cognival-vectors/eye-tracking/all_scaled.txt')

**Behavior**:
- [`PPMI_SVD_SWOW`](https://smallworldofwords.org/en/project/research) ('SWOW-EN18', further processed with PPMI and SVD transformations)
- [`SGSoftMaxInput_SWOW`](https://smallworldofwords.org/en/project/research) ('SWOW-EN18', further processed with Skip-Gram Softmax embedding algorithm)
- [`SGSoftMaxOutput_SWOW`](https://smallworldofwords.org/en/project/research) ('SWOW-EN18', further processed with Skip-Gram Softmax embedding algorithm)
- [`PPMI_SVD_SouthFlorida`](http://w3.usf.edu/FreeAssociation/) ('Appendix A. The normed cues, their targets and related information', further processed with PPMI and SVD transformations)
- [`PPMI_SVD_EAT`](http://w3.usf.edu/FreeAssociation/) ('ea-thesaurus.json', further processed with PPMI and SVD transformations)
- [`THINGS`](https://osf.io/z2784/) ('spose_embedding_49d_sorted.txt' and 'items1854names.tsv')
- [`feature_overlap`](https://github.com/doomlab/shiny-server/blob/master/wn_double/double_words.csv)
- [`norms_sensorimotor`](https://osf.io/rwhs6/files/osfstorage) ('Lancaster_sensorimotor_norms_for_39707_words.csv')
- [`compo_attribs`](https://www.neuro.mcw.edu/index.php/resources/brain-based-semantic-representations/) ('word_ratings.zip')
- [`SVD_sim_rel`](XX) (further processed with SVD transformation)


