from aiogram import types
from aiogram.utils.keyboard import InlineKeyboardBuilder

from search_engine import SearchEngine


def create_meme_keyboard(search_engine: SearchEngine, meme_idx: int, query: str = "") -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    if query:
        query_token = search_engine.store_query_token(query)
        kb.button(text="Еще по этому запросу", callback_data=f"more:{query_token}")
    kb.button(text="В избранное", callback_data=f"fav:{meme_idx}")
    kb.button(text="Случайный мем", callback_data="random")
    kb.button(text="В меню", callback_data="menu")
    kb.adjust(1, 2, 1)
    return kb.as_markup()


def create_main_keyboard() -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Случайный мем", callback_data="random")
    kb.button(text="Мои избранные", callback_data="my_favorites")
    kb.button(text="Поиск мемов", callback_data="search")
    kb.button(text="Помощь", callback_data="help")
    kb.adjust(2, 2)
    return kb.as_markup()


def create_favorites_keyboard(meme_idx: int, current_page: int = 0) -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Удалить из избранного", callback_data=f"unfav:{meme_idx}:{current_page}")
    kb.button(text="Предыдущий", callback_data=f"fav_prev:{current_page - 1}")
    kb.button(text="Следующий", callback_data=f"fav_next:{current_page + 1}")
    kb.button(text="В меню", callback_data="menu")
    kb.adjust(1, 2, 1)
    return kb.as_markup()


def setup_main_menu() -> list[types.BotCommand]:
    return [
        types.BotCommand(command="/start", description="Перезапустить бота"),
        types.BotCommand(command="/random", description="Случайный мем"),
        types.BotCommand(command="/search", description="Поиск мемов"),
        types.BotCommand(command="/addmeme", description="Добавить мем"),
        types.BotCommand(command="/cancel", description="Отменить добавление"),
        types.BotCommand(command="/favorites", description="Мои избранные"),
        types.BotCommand(command="/help", description="Помощь"),
    ]
