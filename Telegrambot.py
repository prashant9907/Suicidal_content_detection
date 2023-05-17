import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model from the Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("Prashant-karwasra/DistilBert-suicidal-content-reviews")
model = AutoModelForSequenceClassification.from_pretrained("Prashant-karwasra/DistilBert-suicidal-content-reviews")

import telebot

bot = telebot.TeleBot('Telegram Bot Token')

def text_analysis(review_text):
    inputs = tokenizer(review_text, return_tensors="pt")
    outputs = model(**inputs)
    _, prediction = torch.max(outputs.logits, dim=1)
    class_names = ['non-suicide', 'suicide']
    print(f'Review text: {review_text}')
    print(f'Sentiment  : {class_names[prediction]}')
    return class_names[prediction]

def print_message(message):
    text = text_analysis(message.text)
    print(text)
    if(text == 'suicide'):
        return True
    else:
        return False

@bot.message_handler(func=print_message)
def send_ans(message):
    bot.reply_to(message, " Help is available Speak with someone today : 9152987821")
    bot.send_message(message.chat.id, " Every storm runs out of rain, just like every dark night turns into day. It's important to hold on and keep fighting")
    bot.send_message(message.chat.id,"You are not alone. In your darkest moments, there are people who care and want to support you. Reach out, and let them help you carry the weight")
    bot.reply_to(message,"Sometimes, the darkest times can bring us to the brightest places. Keep going, because you are stronger than you think")
    # bot.send_sticker(message.chat.id, sticker= "CAACAgIAAxkBAAEBzd5kUqYfyKeTesWjN3x7nga_PoEE2AACWAADUomRI32OzA4HOEuGLwQ")
    bot.send_message(-1001945880068,"or kya haal h bhai")
bot.polling()
