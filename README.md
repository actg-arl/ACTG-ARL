# Differentially Private Conditional Text Generation with RL-Boosted Control

This repository contains code for our project "Differentially Private Conditional Text Generation with RL-Boosted Control".

## Overview

This repository implements

1. [annotation](./annotation/): Extracting features (schemas) from texts according to a given schema, via prompting an LLM (Gemini)

2. [privacy_accounting](./privacy_accounting/): Privacy accounting for 2 two-stage DP mechanisms: composition of DP-SGD and DP-SGD, composition of AIM + DPSGD.

3. [AIM](./AIM/): Generating tabular schemas via AIM.

4. [DPSFT](./DPSFT/): Fine-tuning an LLM (gemma-3-1b-pt) via differentially private supervised fine-tuning, and generating synthetic texts. Supports both baseline SFT and our conditional SFT.

5. [RL](./RL/): Improving the instruction following capability of the conditional generation module via our proposed anchored RL algorithm.

6. [evaluation](./evaluation/): Evaluating the quality of synthetic texts via multiple metrics, including MAUVE, feature divergence, domain classification.

Each module contains a separate README file with detailed instructions.