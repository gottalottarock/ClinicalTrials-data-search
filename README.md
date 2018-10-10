# ClinicalTrials-data-search
CLI for searching for studies in the data from ClinicalTrials by name drugs and research phase for B-Cell Lymphoma disease.

## Usage

```console
usage:  python search.py [-h] -d DRUGS [-p {1,2,3} [{1,2,3} ...]] [--debug] [-s]

optional arguments:
  -h, --help            show help message and exit
  -d DRUGS, --drugs DRUGS
                        Drugs query to search
  -p {1,2,3} [{1,2,3} ...], --phases {1,2,3} [{1,2,3} ...]
                        Possible phases
  -s, --similarity      
                        print similarity to the query for each result 
  --debug               
                        -s + print word's weights in query, docs embaddings for each result
```
