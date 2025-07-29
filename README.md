# BaryGNN

This is the basic project for BaryGNN, focused on barycenter pooling methods for Graph Neural Networks (GNNs).

## Environment Setup (conda + uv)

To set up the development environment on a new machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/matfain/BaryGNN.git
   cd BaryGNN
   ```

2. **Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)** if not already installed.

3. **Create and activate the conda environment:**
   ```bash
   conda create -y -n barygnn python=3.11
   conda activate barygnn
   ```

4. **Install [uv](https://astral.sh/docs/uv/):**
   ```bash
   pip install uv
   ```

5. **Install all project dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```
   - If you want to regenerate the requirements file from the project definition:
     ```bash
     uv pip compile pyproject.toml --output-file requirements.txt
     uv pip install -r requirements.txt
     ```

6. **You're ready to go!**

If you encounter any issues, make sure your conda environment is active and Python version is 3.11.