"""
Three models under consideration for sentence-classification (i.e. PD/HC classification)

1. BERT + basic classifier, this look _only_ at the text and nothing else at the first instance
2. Use ELMO/BERT embeddings and feed these into a classifier along with other information
3. Contextual string embeddings (most recent, has excellent performance)
"""
