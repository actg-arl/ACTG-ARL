# Privacy Accounting for Two-Stage DP Mechanisms

This code implements privacy accounting for two two-stage DP mechanisms:

- DP-SGD + DP-SGD

- AIM + DP-SGD

This implementation is built on the following packages: 
[dp-accounting](https://pypi.org/project/dp-accounting/) 
and
[prv-accountant].

## Setup

```
pip install -r requirements.txt
```

## Example Scripts

### Accounting for DP-SGD + DP-SGD

The code accepts the target $(\varepsilon,\delta)$, the dataset size $N$, the list of batch size values and the list of training iterations to use for each stage, the list of epsilon values to spend for the first stage, and outputs a dataframe of configurations, where each configuration describes $(b_1,T_1,\sigma_1,b_2,T_2,\sigma_2)$.

Accounting is done through PRV Accountant.

```
bash scripts/run_privacy_analysis_composed_dpsgd-dpsgd.sh
```


### Accounting for AIM + DP-SGD


The code accepts the target  $(\varepsilon,\delta)$ as well as the configuration to be used for the second stage of DP-SGD: $(N, b, T, \sigma)$, and computes the parameter $\rho$ to be used for the first AIM stage.

Accounting is done through RDP Accountant.

```
bash scripts/run_privacy_analysis_composed_aim-dpsgd.sh
```
