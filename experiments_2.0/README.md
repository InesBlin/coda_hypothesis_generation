Scripts

Retrieve data
python experiments_2.0/save_data_hypotheses.py http://localhost:7200/repositories/coda experiments/cat_moderators.json experiments_2.0/data/entry

Prep data
python experiments/prep_data.py experiments_2.0/data/entry experiments_2.0/data/llm llm
python experiments/prep_data.py experiments_2.0/data/entry experiments_2.0/data/classification classification
python experiments/prep_data.py experiments_2.0/data/entry experiments_2.0/data/lp lp

Classification
python experiments/hp_kg_embed/save_embedding_classification.py ./experiments_2.0/data/embeds