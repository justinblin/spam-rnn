import discord
from dotenv import load_dotenv
import os
import torch
from use import use

load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents = intents)

# SETUP THE RNN
labels_unique = ['ham', 'spam']
rnn = torch.load('./my_model', weights_only = False)

@client.event
async def on_ready():
    print(f'We logged in as {client.user}')

@client.event
async def on_message(message:discord.Message):
    if message.author != client.user:
        print(f'Seen a message: {message.content}')
        guess, guess_index = use(rnn, message.content, labels_unique)
        print(f'This seems like {guess}')
        if guess == 'spam':
            await message.channel.send('This seems like spam!')

client.run(BOT_TOKEN)