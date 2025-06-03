# To Do
- Retrain RNN to make better
    - Post-training threshold tuning?
- Check out Long Short Term Memory (LSTM)/GRU/transformers for a future project
    - Don't really wanna switch up the model structure for this one now

- Add project to website  
- Allow bot to pm mods or kick spammers  

<br>

# Project Outline
This project contains a recursive neural network trained to differentiate between spam and non-spam text,
as well as a discord bot that uses the RNN to automatically detect spam messages, warn server moderators,
and/or kick/ban spam users.

Use the following link to add the bot to your server:

https://discord.com/oauth2/authorize?client_id=1370818702810939463&permissions=1099511696390&integration_type=0&scope=bot

<br>

# Cool Features
Since most messages are inherently NOT spam, the dataset available to train on in heavily imbalanced, requiring some 
special techniques to train well. One example is having an increased cost for missing spam, encouraging the network to guess spam more often. Additionally, selecting batches to have a higher proportion of spam messages helps decrease imbalance at the cost of wasting 
some data. Another example is using different metrics for testing, such as precision (out of those the model guessed spam, how many were right), recall (out of all of the spam, how many did the model guess), and f1 score (a mix of both) to get accurate information about how well the model did.

Another problem I encountered was loss plateaus, where the network would reach a certain point and stop improving. One way I found to 
deal with this was by creating a variable learning rate test that looks for the optimal learning rate whenever I encoutered a lack of significant progress over a certain amound of time.

An additional obstacle was when the model would occasionally zero-out all the gradients due to the ReLU layers turning all negative values into 0, leading the model to get stuck if there was an unlucky initialization or it stumbled into a local minimum where it output all 0's. By using leaky ReLU (where negative values are made much smaller, but not fully 0) instead of traditional ReLU, I was able to prevent the disappearing gradients.

# Sections of the Code
## Recursive Neural Network
preprocess.py/postprocess.py store useful data processing functions.  
rnn.py has the structure of the RNN.  
dataset.py does some basic data splitting and has the structure of the datasets.  
learn.py allows the bot to be trained/tested on the datasets.  
use.py allows single case use and provides human-readable output.  
my_model stores information about the pretrained model (weights, biases, model dimensions, etc.).  
## Discord Bot
my_bot.py runs the discord bot utilizing the output from use.py.  