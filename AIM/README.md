# Synthetic Tabular Schemas Generation via AIM

This code implements the generation of tabular schemas of the bioRxiv dataset using AIM. The AIM package is provided by [Ryan Mckenna](https://scholar.google.com/citations?user=qv5vhKEAAAAJ&hl=en).

## Setup

```
pip install git+https://github.com/ryan112358/private-pgm.git
```

## Example Scripts

- Performs generation assuming a $\rho$ value is supplied. 

    Note that the $\rho$ value can be obtained using our privacy accounting code at [privacy_accounting](../privacy_accounting/).

    ```
    python main.py --rho 0.06 --pgm_iters 2000 --num_gen 5000
    ```

- Perform generation assuming an $\varepsilon$ value is supplied.
    ```
    python main.py --eps 4 --pgm_iters 5000 --num_gen 5000
    ```
