# To Do
Retrain RNN to make better, do something about the unbalaced dataset, maybe increase complexity of the model, add to website/resume

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