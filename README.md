# DA6401 - Assignment 2

## Important Links
- **Link to GitHub Repo** : <https://github.com/AkharanCS/CH21B009_DL_Assignment2>
- **Link to wandb report** : <https://api.wandb.ai/links/ch21b009-indian-institute-of-technology-madras/gmq8fbz9>

## Directories
- **part_A**: contains all the codes and configurations required for Part-A of the assignment.
- **part_B**: contains all the codes and configurations required for Part-B of the assignment.

## Files in part_A 
- **`A_Q1.py`**: contains the CNN class which includes the architecture and relevant methods of the convolutional neural network.
- **`A_Q2.py`**: contains code for performing the wandb sweep required in Part-A Q2.
- **`A_Q4.py`**: contains code for running the best network configurations on inaturalist_12K as required in Part-A Q4.
- **`config.yaml`**: contains the hyperparameter space used for running the wandb sweep in Part-A Q2.

## Files in part_B
- **`B_Q3.py`**: contains code for fine-tuning ResNet50 for the inaturalist_12K dataset as required in Part-B Q3.

## Other Important files
- **`requirements.txt`**: contains all the libraries and dependancies required for running both part A and B.

## Steps to run (follow this order)
1. Clone the repository:
   ```bash
   git clone https://github.com/AkharanCS/CH21B009_DL_Assignment2.git
   ```
2. Download the inaturalist_12K dataset using the following commands in the root directory:
    ```bash
    wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip
    unzip nature_12K.zip
   ```
3. Setup a python virtual environment with the required dependancies using the following commands:
     ```bash
    python -m venv venv
    venv/Scripts/activate
    pip install -r requirements.txt
   ```
4. Run `A_Q1.py` inside Part-A. <br>
5. Save `config.yaml` file inside Part-A. <br>
6. All the other files in Part-A,(`A_Q2.py`,`A_Q4.py`) can be run in any order. <br>
7. There is only one file in Part-B, (`B_Q3.py`) which can be run independently. <br>