# To Do
- Retrain RNN to make better
    - Do something about the unbalaced dataset  
        - ~~Mess with the cost function to make missing spam more costly? (don't really wanna build own cost function)~~  
            - ~~Check Pytorch doc if there's a builtin weighted cost func~~  
        - ~~Make each batch have a fixed number of spam (currently 85/15, try like 70/30)~~  
        - ~~More epochs, more aggressive loss weighting? Causes problems with outputs zeroing out~~  
            - ~~Seem to be missing more spam than misclassifying ham as spam, make it more aggressive with guessing spam~~  
    - Fix loss plateaus
        - Implement variable learning rates when loss doesn't decrease a certain amount over some epochs
            - Need to refactor old code to use torch DataLoader and Sampler since LRFinder needs DataLoader

- Look at different criteria for testing (validation loss, precision/recall/f1 score?) and test more often (within the training loop every couple epochs?)  
    - Graph the testing score over time and show with training loss? Could help against overfitting
- Add project to website/resume  
- Allow bot to pm mods or kick spammers  

<br>

# Project Outline
This project contains a recursive neural network trained to differentiate between spam and non-spam text,
as well as a discord bot that uses the RNN to automatically detect spam messages, warn server moderators,
and/or kick/ban spam users.

Use the following link to add the bot to your server:

https://discord.com/oauth2/authorize?client_id=1370818702810939463&permissions=1099511696390&integration_type=0&scope=bot

<br>

# Sections
## Recursive Neural Network
preprocess.py/postprocess.py store useful data processing functions.  
rnn.py has the structure of the RNN.  
dataset.py does some basic data splitting and has the structure of the datasets.  
learn.py allows the bot to be trained/tested on the datasets.  
use.py allows single case use and provides human-readable output.  
my_model stores information about the pretrained model (weights, biases, model dimensions, etc.).  
## Discord Bot
my_bot.py runs the discord bot utilizing the output from use.py.  