from dataclasses import dataclass
import json
from random import choice
import os
import requests
import telegram.ext as tg
import telegram.error as tg_errors
import pandas as pd
import tensorflow as tf
from GoogleNews import GoogleNews

from TelegramBot.pipelines.Pipelines import setup_prediction_pipeline

tf.compat.v1.disable_eager_execution()


@dataclass
class TelegramBot:
    bot_token: str
    weather_api_key: str
    target_column_name: str = "X"
    max_length: int = 10
    label_encoder_path = r"ChatBotModel\label_encoder3.pkl"
    tokenizer_path: str = r"ChatBotModel\tokenizer3.pkl"
    model_path: str = r"ChatBotModel\ChatBotIntentsModel4.h5"
    intents_path: str = r"ChatBotModel\intents4.json"

    def __post_init__(self):
        self.updater = tg.Updater(token=self.bot_token, use_context=True)
        self.disp = self.updater.dispatcher

        self.prediction_pipeline = setup_prediction_pipeline(
            target_column_name=self.target_column_name,
            max_length=10,
            tokenizer_path=self.tokenizer_path,
            label_encoder_path=self.label_encoder_path,
            unique_words_path="",
            model_path=self.model_path,
            intents_path=self.intents_path
        )

    @staticmethod
    def start_command(update, context):
        update.message.reply_text("Hello! I'm awake. Type /help to get more information about my functionality")

    @staticmethod
    def pepe_command(update, context):
        pepes_dir = r".\Images\Pepe"
        pepes = os.listdir(pepes_dir)

        pepe = choice(pepes)
        with open(rf"{pepes_dir}\{pepe}", "rb") as img:
            update.message.reply_photo(img)

    @staticmethod
    def news_command(update, context):
        try:
            default_params = ["Poland", "EN", "PL", 1]
            user_params = default_params.copy()

            for index, param in enumerate(context.args):
                user_params[index] = param

            phrase = user_params[0]
            lang = user_params[1]
            region = user_params[2]
            max_num = int(user_params[3])

            googlenews = GoogleNews(lang=lang, region=region.upper())
            googlenews.get_news(phrase.replace("_", " "))
            res = googlenews.results()
            urls = [article["link"] for article in res[:max_num]]

            space = f"\n{100 * '-'}\n"
            update.message.reply_text(f"{space}".join(urls))
        except tg_errors.BadRequest:
            update.message.reply_text("Message too long :/")
        except IndexError:
            update.message.reply_text("Too much arguments")
        except Exception as e:
            update.message.reply_text(f"Unhandled error: {e}")

    def weather_command(self, update, context):
        try:
            city = context.args[0]
            complete_url = f"https://api.openweathermap.org/data/2.5/weather?q={city.replace('_', ' ')}" \
                           f"&appid={self.weather_api_key}&units=metric"
            request = requests.get(complete_url)

            status_code = request.status_code

            if status_code == 200:
                data = json.loads(request.content)
                weather = data["weather"][0]

                description = weather["description"]
                pressure = data["main"]["pressure"]
                humidity = data["main"]["humidity"]
                temp = data["main"]["temp"]

                update.message.reply_text(f"Temp: {temp}°C, {description}, pressure: {pressure}, humidity: {humidity}")

            elif status_code == 401:
                update.message.reply_text(f"401 (unauthorized), check if  your weather API key is correct")

            elif status_code == 404:
                update.message.reply_text(f"City not Found :/")

        except (TimeoutError, ConnectionError):
            update.message.reply_text(f"There is something wrong with connection :(")

    @staticmethod
    def help_command(update, context):
        update.message.reply_text(
            """
- /pepe -> sends random pepe
- /news [phrase (replace spaces with _)] [language] [region] [number of urls] -> returns urls to articles about
given phras
- /weather [city] -> returns weather data for given city
        """
        )

    def handle_messages(self, update, context):
        df = pd.DataFrame({
            self.target_column_name: [update.message.text]
        })

        pred = self.prediction_pipeline.fit_transform(df)

        update.message.reply_text(pred["Answers"][0])

    def setup_bot(self):
        method_list = [getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")]

        for func in method_list:
            func_name = func.__name__
            if func_name.endswith("command"):
                self.disp.add_handler(tg.CommandHandler(func_name.replace("_command", ""), func))

        self.disp.add_handler(tg.MessageHandler(tg.Filters.text, self.handle_messages))

        self.updater.start_polling()
        self.updater.idle()


if __name__ == '__main__':
    with open("tokens.json") as f:
        keys = json.load(f)

    TOKEN = keys["TelegramToken"]
    WEATHER_API_KEY = keys["WeatherApiKey"]

    tg_bot = TelegramBot(bot_token=TOKEN, weather_api_key=WEATHER_API_KEY)
    print("Bot started...")

    tg_bot.setup_bot()


