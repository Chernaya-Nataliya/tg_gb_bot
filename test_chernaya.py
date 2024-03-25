# Создание телеграм бота, который может распознавать объекты на изображениях, отправленных пользователями,
#и возвращать результаты обратно пользователю в чате Telegram.

# Импорт необходимых библиотек
import telebot
from telebot import types
import requests
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
import numpy as np

# Установка токена бота
API_TOKEN = '7192901426:AAHvBdPsUceMMw5An52VlvPjhMZtUqnvcFY'
bot = telebot.TeleBot(API_TOKEN)

# Загрузка модели для распознавания объектов: Создается экземпляр модели MobileNetV2 
#с предварительно обученными весами на наборе данных ImageNet для распознавания объектов на изображениях.
model = MobileNetV2(weights='imagenet')

# Функция для обработки изображений, отправленных пользователем. Она загружает изображение, предварительно обрабатывает его для подготовки к анализу моделью, предсказывает объекты на изображении и возвращает топ-3 предсказанных объекта с их уверенностью.
@bot.message_handler(content_types=['photo'])
def handle_image(message):
    try:
        # Получаем информацию о фото
        file_info = bot.get_file(message.photo[-1].file_id)
        file = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(API_TOKEN, file_info.file_path))

        # Сохраняем изображение
        with open("image.jpg", "wb") as img:
            img.write(file.content)

        # Загружаем изображение для обработки
        img_path = "image.jpg"
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Предсказываем объекты на изображении
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Отправляем результат пользователю
        result = "Результат распознавания:\n"
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            result += f"{i+1}: {label} ({score:.2f})\n"
        
        bot.reply_to(message, result)

    except Exception as e:
        bot.reply_to(message, "Что-то пошло не так. Попробуй еще раз.")

# Запуск бота
bot.polling()