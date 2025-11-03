# NFTM -  State of the Art

The State of Neural Field Turing Machine Research Available on Springer Open.

## Setup - Virtual environment

1. **Create the environment in python**
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```
2. **Install dependencies** 
   ```bash
   pip install --upgrade pip
   pip install springernature-api-client requests beautifulsoup4
   ```

## Code configuration
Before running any file, it is mandatory to create an account in Open Access API in order to obtain the API key that needs to be replace in the variable found in the file *ETL.py* defined as *apiKey*.

Additionally, edit the keywords that works as the labels for every retrieved query.


## Visualization
For a deep understanding of the Metadata related to each paper, the file *dataVis.ipynb* helps to achieve this task.  We recommend to run it in a notebook environment.

```bash
jupyter lab
# or
jupyter notebook
   ```

The original version of the current repository is found here: https://github.com/filiperusso/BDMA/tree/main/UPC2025/BDS   