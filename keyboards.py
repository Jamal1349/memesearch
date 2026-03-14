from aiogram import types
from aiogram.utils.keyboard import InlineKeyboardBuilder

from search_engine import SearchEngine


def create_meme_keyboard(
    search_engine: SearchEngine,
    meme_idx: int,
    search_session_token: str = "",
) -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    if search_session_token:
        kb.button(text="Р•С‰Рµ РїРѕ СЌС‚РѕРјСѓ Р·Р°РїСЂРѕСЃСѓ", callback_data=f"more:{search_session_token}")
    kb.button(text="Р’ РёР·Р±СЂР°РЅРЅРѕРµ", callback_data=f"fav:{meme_idx}")
    kb.button(text="РЎР»СѓС‡Р°Р№РЅС‹Р№ РјРµРј", callback_data="random")
    kb.button(text="Р’ РјРµРЅСЋ", callback_data="menu")
    kb.adjust(1, 2, 1)
    return kb.as_markup()


def create_main_keyboard() -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="РЎР»СѓС‡Р°Р№РЅС‹Р№ РјРµРј", callback_data="random")
    kb.button(text="РњРѕРё РёР·Р±СЂР°РЅРЅС‹Рµ", callback_data="my_favorites")
    kb.button(text="РџРѕРёСЃРє РјРµРјРѕРІ", callback_data="search")
    kb.button(text="РџРѕРјРѕС‰СЊ", callback_data="help")
    kb.adjust(2, 2)
    return kb.as_markup()


def create_favorites_keyboard(meme_idx: int, current_page: int = 0) -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="РЈРґР°Р»РёС‚СЊ РёР· РёР·Р±СЂР°РЅРЅРѕРіРѕ", callback_data=f"unfav:{meme_idx}:{current_page}")
    kb.button(text="РџСЂРµРґС‹РґСѓС‰РёР№", callback_data=f"fav_prev:{current_page - 1}")
    kb.button(text="РЎР»РµРґСѓСЋС‰РёР№", callback_data=f"fav_next:{current_page + 1}")
    kb.button(text="Р’ РјРµРЅСЋ", callback_data="menu")
    kb.adjust(1, 2, 1)
    return kb.as_markup()


def setup_main_menu() -> list[types.BotCommand]:
    return [
        types.BotCommand(command="/start", description="РџРµСЂРµР·Р°РїСѓСЃС‚РёС‚СЊ Р±РѕС‚Р°"),
        types.BotCommand(command="/random", description="РЎР»СѓС‡Р°Р№РЅС‹Р№ РјРµРј"),
        types.BotCommand(command="/search", description="РџРѕРёСЃРє РјРµРјРѕРІ"),
        types.BotCommand(command="/addmeme", description="Р”РѕР±Р°РІРёС‚СЊ РјРµРј"),
        types.BotCommand(command="/cancel", description="РћС‚РјРµРЅРёС‚СЊ РґРѕР±Р°РІР»РµРЅРёРµ"),
        types.BotCommand(command="/favorites", description="РњРѕРё РёР·Р±СЂР°РЅРЅС‹Рµ"),
        types.BotCommand(command="/help", description="РџРѕРјРѕС‰СЊ"),
    ]
