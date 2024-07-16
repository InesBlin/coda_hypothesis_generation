# Automated Hypothesis Generation on Human Cooperation

This is the code we submit together with the paper [TO-ADD], submitted to [TO-ADD]. The code contains the followings:
- The core code in the `src` folder, for general usage
- All elements related to our experiments, including:
    * The data and models, cf. folders `data.zip` and `models`
    * Thorough details on the experiements in the folder `experiments`. For more clarity we provide a [README.md](./experiments/README.md) for the experiments in that folder.


## Installing a virtualenv + Set-Up

We used Python 3.10.8. If you plan to use the OpenAI API, you need to add your API key. To do, create a file `src/settings.py` and add your key:
```python
API_KEY_GPT = "your-key"
```

Installing rpy2 can cause problems with conda, hence we recommend to first install it before installing all the other dependencies.

```bash
conda install rpy2
pip install -r requirements.txt
python setup.py install
cd kglab && python setup.py install
```

## Code Structure

- `src/helpers`: various helpers used throughout the code
- `src/knowledge.py`: human-readable hypothesis generation
- `src/hg`: all classes related to hypothesis generation