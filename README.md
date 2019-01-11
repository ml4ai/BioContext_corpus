# Biological Container Context Corpus

This repository is the home of the public release of the Biological Container Context Corpus.

The `corpus_data` directory contains three subdirectories, one for each of the annotators, which in turn contains the annotation files for each of the documents annotated by a particular annotator.

The annotations files have the format `$PMCID_$type.tsv`, where the `$PMCID` is substituted for the id of the paper, and `$type` is substituted for the annotation type of the file.

The available annotation file types are:

- REACH
- Events
- Context
- Grounding

The mieaning of the fields in each file is described below.

## REACH

This file contains the annotations made a a domain expert where he or her associated a specific biochemical event to a specific context type. The events in this file were automatically extracted by REACH. If an event was not automatically extracted by information extraction, it was manually annotated by the human and still used as a data point.

The first column represents the __line number__ of the full text. The second column represents the __word interval__ of the the annotated event in the line and the third column is a _comma separated list_ of the context types associated to it by the human annotator.

## Events

This file contains the annotations made a a domain expert where he or her associated a specific biochemical event to a specific context type. The events of this file were detected manually by the annotator and pooled with the event annotations of REACH for training. If an event was not automatically extracted by information extraction, it was manually annotated by the human and still used as a data point.

The first column represents the __line number__ of the full text. The second column represents the __word interval__ of the the annotated event in the line and the third column is a _comma separated list_ of the context types associated to it by the human annotator.

## Context

In a format similar to the _events_ file, this file contains the context mentions, automatically detected by REACH. The format is similar:

The first column represents the __line number__ of the full text. The second column represents the __word interval__ of the the annotated event in the line and the third column is the _context type_ of the word. If the word is human, then the context type would be _taxonomy:9606_.

## Grouding

The content of this file is a list with all the _context mention types_ detected by REACH. Not all those context types will be associated to a specific event, just a subset of those.

---

## Full text of the articles

An aditional file exists with the lines and words tokenized which are also output of the REACH IE system. We don't include them in the repository to avoid violating any copy right, However, they're not necessary for training and evaluation. If you're interested in those files, please contact the author for instructions on how to generate them.

---

For the main project page, visit https://ml4ai.github.io/BioContext/ 

