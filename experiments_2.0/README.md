Scripts

Retrieve data
python experiments_2.0/save_data_hypotheses.py http://localhost:7200/repositories/coda experiments/cat_moderators.json experiments_2.0/data/entry

Prep data
python experiments/prep_data.py experiments_2.0/data/entry experiments_2.0/data/llm llm
python experiments/prep_data.py experiments_2.0/data/entry experiments_2.0/data/classification classification
python experiments/prep_data.py experiments_2.0/data/entry experiments_2.0/data/lp lp

Classification

- save embeddings
python experiments/hp_kg_embed/save_embedding_classification.py ./experiments_2.0/data/classification ./experiments_2.0/data/embeds

- search hp
python experiments/classification/search_hp_classification.py ./experiments_2.0/data/classification/ ./experiments_2.0/data/embeds/ ./experiments_2.0/classification/hp_search


Link Prediction task

DL-based

- save data
python experiments/hp_bn_lp/save_data_bn.py experiments_2.0/data/lp/ experiments_2.0/data/bn/ data/vocab.csv 

- search hp
python experiments/hp_bn_lp/search_hp_bn_lp.py experiments_2.0/data/bn/ experiments_2.0/hp_bn_lp/ data/vocab.csv

AnyBURL

- save data
python experiments/anyburl/save_data_anyburl.py ./experiments_2.0/data/bn/ ./experiments_2.0/data/anyburl ./data/vocab.csv