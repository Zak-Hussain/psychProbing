## psychProbing

This repository contains the code for the paper:

Hussain, Z., Mata, R., Newell, B. R., & Wulff, D. U. (2024). Probing the contents of semantic representations from text, behavior, and brain data using the psychNorms metabase. arXiv. https://arxiv.org/abs/2412.04936

```
@misc{hussain2024probingcontentssemanticrepresentations,
      title={Probing the contents of semantic representations from text, behavior, and brain data using the psychNorms metabase}, 
      author={Zak Hussain and Rui Mata and Ben R. Newell and Dirk U. Wulff},
      year={2024},
      eprint={2412.04936},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.04936}, 
}
```

### Environment setup

1. To set up a conda environment, you can use the `environment.yml` file in the root directory of this repository. 
2. **Before running any other code, make sure to run `code/setup.py` to download/generate the necessary data files. Please note that
to reduce the download size of the representations, we have already subsetted many of them to their intersection with the psychNorms vocabulary.**
3. For licensing reasons, you will need to manually download `SWOW-EN18 [80Mb]` from the [Small World of Words](https://smallworldofwords.org/en/project/research), unzip it, and move `SWOW-EN.R100.20180827.csv` into `data/free_assoc/`.
4. To obtain the representations that we trained ourselves, you will need to run the notebooks in `code/embed_training/`. 
5. You can then run the analyses in the following order: `code/rsa`, `code/rca`, `code/figures`, running the code files in the order implied by the numbering within each directory.

### Representations

The original sources of the representations are as follows:

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
- [`PPMI_SVD_EAT`](https://github.com/dariusk/ea-thesaurus) ('ea-thesaurus.json', further processed with PPMI and SVD transformations)
- [`THINGS`](https://osf.io/z2784/) ('spose_embedding_49d_sorted.txt' and 'items1854names.tsv')
- [`feature_overlap`](https://github.com/doomlab/shiny-server/blob/master/wn_double/double_words.csv) ('double_words.csv')
- [`norms_sensorimotor`](https://osf.io/rwhs6/files/osfstorage) ('Lancaster_sensorimotor_norms_for_39707_words.csv')
- [`compo_attribs`](https://www.neuro.mcw.edu/index.php/resources/brain-based-semantic-representations/) ('word_ratings.zip')
- `SVD_sim_rel`: 'AG203', 'BakerVerb', 'MartinezAldana', 'MC30', 'MEN3000', 'RG65', 'SimLex999', 'SimVerb3500', 'SL7576sem', 'SL7576vis', 'WP300', 'YP130', 'Atlasify240', 'GM30', 'MT287', 'MT771', 'Rel122',
       'RW2034', 'WordSim353', 'Zie25', 'Zie30' (datasets were combined, min-max scaled and then processed with SVD transformation).

Note: `compo_attribs` has been renamed to 'experiential attributes' in the paper and figures to be consistent with the terminolgy in the psychNorms metabase.

### Norms

Information on the norms used in our analysis can be found in the [psychNorms repository](https://github.com/Zak-Hussain/psychNorms), and 
in the metadata file in `data/psychNorms/psychNorms_metadata.csv`.

