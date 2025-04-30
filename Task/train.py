#%%
import os
from openai import OpenAI
import openai
import json
from Preprocessing.feature_extraction import load_eeg_data
#%%
openai.api_key = os.getenv("OPENAI_API_KEY")  # You have set your own environment variable
client = OpenAI()
#%%
def create_file(train_jsonl_dir, val_jsonl_dir):
    """
    Create a file in the OpenAI API
    After creating the file, you should see the file ID(train/test) in the website(platform.openai.com).
    :param train_jsonl_dir: A directory of the train data in jsonl format
    :param val_jsonl_dir: A directory of the test data in jsonl format
    :return:
    """
    # load train data
    client.files.create(
        file=open(train_jsonl_dir, 'rb'),
        purpose='fine-tune'
    )
    print(f"Loaded {train_jsonl_dir}")

    # load validation data
    client.files.create(
        file=open(val_jsonl_dir, 'rb'),
        purpose='fine-tune'
    )
    print(f"Loaded {val_jsonl_dir}")
#%%
base_path = 'your_base_path'
train_jsonl_dir = base_path + 'jsonl/your_train_jsonl.jsonl'
val_jsonl_dir = base_path + 'jsonl/your_val_jsonl.jsonl'
#%%
create_file(train_jsonl_dir, val_jsonl_dir)
#%%
window_size = 1000
selected_columns = [
        [0, [(10, 12), (12, 14)]],  # FCz
        [2, [(20, 22), (22, 24)]],  # C3
        [3, [(8, 10)]],  # Cz
        [4, [(20, 22), (22, 24)]],  # C4
        [5, [(28, 30)]],  # CP3
    ]
#%%
training_file_1 = 'file-your_training_file_1'  # Ensure this should be checked in the website(platform.openai.com)
val_file_1 = 'file-your_val_file_1'  # Ensure this should be checked in the website(platform.openai.com)

training_file_2 = 'file-your_training_file_2'  # Ensure this should be checked in the website(platform.openai.com)
val_file_2 = 'file-your_val_file_2'  # Ensure this should be checked in the website(platform.openai.com)

training_file_3 = 'file-your_training_file_3'  # Ensure this should be checked in the website(platform.openai.com)
val_file_3 = 'file-your_val_file_3'  # Ensure this should be checked in the website(platform.openai.com)

training_file_4 = 'file-your_training_file_4'  # Ensure this should be checked in the website(platform.openai.com)
val_file_4 = 'file-your_val_file_4'  # Ensure this should be checked in the website(platform.openai.com)

model = 'gpt-4o-2024-08-06'
#%% md
# We did a binary classification for each label on the Motor Image EEG data with 4 labels. Because of this, there are a total of 4 different datasets, and we created a total of 4 models. If you want to design a GPT fine-tuning model that performs multi-class classification, you can change the code accordingly.
#%% md
# ### FYI
# <p>The hyperparameters are automatically set by OpenAI during fine tuning without having to set them separately.<br>You can set them yourself if you want.</p>
#%%
def train(training_file, val_file, model):
    """
    Fine-tuning the GPT model.
    After training, you should check the name of the model in the website(platform.openai.com).
    :param training_file: The ID of the training file
    :param val_file: The ID of the validation file
    :param model: anything you want(davinci-002 / gpt-3.5-turbo / and so on)
    """
    # start fine-tuning
    client.fine_tuning.jobs.create(
        training_file=training_file,
        validation_file=val_file,
        model=model
        # default=auto, thus it doesn't need to be specified
        # hyperparameters={
        #     'n_epochs':10,
        #     'batch_size':16,
        #     'learning_rate_multiplier':1e-4
        # }
    )

    print("Fine-tuning started.")
#%% md
# ### FYI
# <p>OpenAI supports up to two GPT fine-tuning at a time.<br>That's the reason that I divided the training cell into two parts.</p>
#%%
train(training_file_1, val_file_1, model)
train(training_file_2, val_file_2, model)
#%%
train(training_file_3, val_file_3, model)
train(training_file_4, val_file_4, model)
#%%
