import asyncio
import html
import os
import uuid
from typing import Optional

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramRetryAfter
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    InlineQueryResultArticle,
    InlineQueryResultCachedPhoto,
    InputTextMessageContent,
)

from app_config import configure_logging, load_config
from keyboards import (
    create_favorites_keyboard,
    create_main_keyboard,
    create_meme_keyboard,
    setup_main_menu,
)
from media_utils import create_input_file
from search_engine import SearchEngine, normalize_query_key
from storage import BotStorage


logger = configure_logging()
config = load_config()
storage = BotStorage(config)
search_engine = SearchEngine(config, logger)

bot = Bot(token=config.token, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()
BOT_USERNAME: Optional[str] = None
warmup_task: Optional[asyncio.Task] = None
pending_add_meme: dict[int, dict[str, str]] = {}


def is_admin(user_id: int) -> bool:
    return (not config.admin_user_ids) or (user_id in config.admin_user_ids)


def is_blocked(user_id: int) -> bool:
    return storage.is_blocked(user_id)


def parse_target_user_id(raw: str) -> int | None:
    text = (raw or "").strip()
    if not text or text.startswith("@"):
        return None
    try:
        return int(text)
    except ValueError:
        return None


async def save_uploaded_photo(photo: types.PhotoSize) -> str:
    os.makedirs(config.local_images_dir, exist_ok=True)
    target_path = os.path.join(config.local_images_dir, f"user_{uuid.uuid4().hex}.jpg")
    file_info = await bot.get_file(photo.file_id)
    await bot.download_file(file_info.file_path, destination=target_path)
    return target_path


async def finalize_add_meme(chat_id: int, user_id: int, description: str, alt: str = "") -> None:
    state = pending_add_meme.get(user_id)
    if not state or not state.get("image_path"):
        await bot.send_message(chat_id, "Сначала отправь картинку через /addmeme")
        return

    image_path = state["image_path"]
    storage.append_local_meme(image_path=image_path, description=description.strip(), alt=alt.strip())
    idx = search_engine.add_local_meme(image_path=image_path, description=description.strip(), alt=alt.strip())
    pending_add_meme.pop(user_id, None)
    await send_meme(chat_id, idx, caption=f"Добавил мем #{idx}")


async def delete_local_meme_by_idx(chat_id: int, meme_idx: int) -> None:
    record = search_engine.delete_local_meme(meme_idx)
    if record is None:
        await bot.send_message(chat_id, "Only local memes can be deleted, and this id was not found.")
        return

    image_info = record.get("image", {})
    image_path = ""
    if isinstance(image_info, dict):
        image_path = str(image_info.get("path") or "")

    if image_path:
        storage.delete_local_meme(image_path)
    storage.purge_meme_references(meme_idx)

    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
        except Exception:
            logger.exception("Failed to delete local meme file: %s", image_path)

    await bot.send_message(chat_id, f"Deleted local meme #{meme_idx}")


async def cache_sent_photo(idx: int, message: types.Message) -> None:
    try:
        storage.set_file_id(idx, message.photo[-1].file_id)
    except Exception:
        logger.exception("Не удалось сохранить file_id для мема #%s", idx)


async def ensure_cached(idx: int) -> Optional[str]:
    cached = storage.get_file_id(idx)
    if cached:
        return cached
    if config.cache_chat_id is None:
        return None

    try:
        row = search_engine.row(idx)
        input_file = create_input_file(row)
        msg = await bot.send_photo(
            chat_id=config.cache_chat_id,
            photo=input_file,
            caption=f"cache:{idx}",
        )
        storage.set_file_id(idx, msg.photo[-1].file_id)
        return msg.photo[-1].file_id
    except Exception:
        logger.exception("Не удалось закешировать мем #%s", idx)
        return None


async def warmup_cache(chat_id_for_updates: int) -> None:
    if config.cache_chat_id is None:
        await bot.send_message(chat_id_for_updates, "CACHE_CHAT_ID не задан")
        return

    state = storage.load_warmup_state()
    next_idx = int(state.get("next_idx", 0))
    ok = int(state.get("ok", 0))
    fail = int(state.get("fail", 0))
    total = search_engine.total()

    await bot.send_message(
        chat_id_for_updates,
        f"Warmup старт: idx={next_idx}, всего мемов: {total}\nКэш-чат: {config.cache_chat_id}",
    )

    sleep_ok = float(os.getenv("WARMUP_SLEEP_OK", "0.8"))
    sleep_fail = float(os.getenv("WARMUP_SLEEP_FAIL", "2.0"))
    save_every = int(os.getenv("WARMUP_SAVE_EVERY", "50"))
    report_every = int(os.getenv("WARMUP_REPORT_EVERY", "200"))

    idx = next_idx
    while idx < total:
        if storage.get_file_id(idx):
            ok += 1
            idx += 1
            state = {"next_idx": idx, "ok": ok, "fail": fail}
            if ok % save_every == 0:
                storage.save_warmup_state(state)
            if ok % report_every == 0:
                await bot.send_message(chat_id_for_updates, f"Прогресс: {ok}/{total}, fail={fail}, idx={idx}")
            continue

        try:
            row = search_engine.row(idx)
            input_file = create_input_file(row)
            msg = await bot.send_photo(
                chat_id=config.cache_chat_id,
                photo=input_file,
                caption=f"cache:{idx}",
            )
            storage.set_file_id(idx, msg.photo[-1].file_id)
            ok += 1
            idx += 1
            state = {"next_idx": idx, "ok": ok, "fail": fail}
            if ok % save_every == 0:
                storage.save_warmup_state(state)
            if ok % report_every == 0:
                await bot.send_message(chat_id_for_updates, f"Прогресс: {ok}/{total}, fail={fail}, idx={idx}")

            await asyncio.sleep(sleep_ok)
        except TelegramRetryAfter as e:
            wait = int(getattr(e, "retry_after", 5)) + 1
            logger.warning("Flood limit on sendPhoto. Wait %ss (idx=%s)", wait, idx)
            await asyncio.sleep(wait)
        except Exception:
            fail += 1
            idx += 1
            storage.save_warmup_state({"next_idx": idx, "ok": ok, "fail": fail})
            logger.exception("Warmup error idx=%s", idx - 1)
            await asyncio.sleep(sleep_fail)

    storage.save_warmup_state({"next_idx": total, "ok": ok, "fail": fail})
    await bot.send_message(chat_id_for_updates, f"Warmup завершен. ok={ok}, fail={fail}, total={total}")


async def send_meme(chat_id: int, idx: int, query: str = "", caption: Optional[str] = None) -> None:
    row = search_engine.row(idx)
    input_file = create_input_file(row)
    sent = await bot.send_photo(
        chat_id=chat_id,
        photo=input_file,
        caption=caption,
        reply_markup=create_meme_keyboard(search_engine, idx, query),
    )
    await cache_sent_photo(idx, sent)


async def show_favorite_meme(chat_id: int, user_id: int, page: int = 0) -> None:
    user_favs = storage.get_favorites(user_id)
    if not user_favs:
        await bot.send_message(chat_id, "У вас пока нет избранных мемов")
        return

    page = max(0, min(page, len(user_favs) - 1))
    idx = user_favs[page]
    row = search_engine.row(idx)
    input_file = create_input_file(row)
    await bot.send_photo(
        chat_id=chat_id,
        photo=input_file,
        caption=f"{page + 1}/{len(user_favs)}",
        reply_markup=create_favorites_keyboard(idx, page),
    )


async def send_random_meme(chat_id: int, user_id: int) -> None:
    try:
        idx = await asyncio.to_thread(search_engine.random_idx)
        await send_meme(chat_id, idx)
    except Exception:
        logger.exception("Ошибка отправки случайного мема")
        await bot.send_message(chat_id, "Ошибка при загрузке мема")


async def perform_search(chat_id: int, query: str, user_id: int) -> None:
    try:
        meme_indices = await asyncio.to_thread(search_engine.search, query, user_id, 5)
        if not meme_indices:
            shown = search_engine.get_shown_results(user_id, normalize_query_key(query))
            if shown:
                await bot.send_message(chat_id, "Больше мемов по этому запросу нет")
            else:
                await bot.send_message(chat_id, f"По запросу «{query}» ничего не найдено")
            return

        idx = meme_indices[0]
        await send_meme(chat_id, idx, query=query)

        if len(meme_indices) > 1:
            await bot.send_message(chat_id, "Нажми «Еще по этому запросу», чтобы увидеть больше мемов")
    except Exception:
        logger.exception("Ошибка поиска")
        await bot.send_message(chat_id, "Ошибка при поиске мемов")


@dp.message(CommandStart())
async def cmd_start(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    args = ""
    if message.text:
        parts = message.text.split(maxsplit=1)
        if len(parts) > 1:
            args = parts[1]

    if args.startswith("send_"):
        try:
            idx = int(args.split("_", maxsplit=1)[1])
            await send_meme(message.chat.id, idx)
            return
        except Exception:
            logger.exception("Ошибка обработки deep-link /start send_")

    welcome_text = "<b>Добро пожаловать в MemeBot!</b>\n\nВыберите действие ниже"
    await message.answer(welcome_text, reply_markup=create_main_keyboard())


@dp.message(Command("search"))
async def cmd_search(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    await message.answer("Введи запрос для поиска мемов")


@dp.message(Command("random"))
async def cmd_random(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    await send_random_meme(message.chat.id, message.from_user.id)
    """_summary_
    """

@dp.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    help_text = (
        "<b>MemeBot - помощь</b>\n\n"
        "<b>Команды:</b>\n"
        "• /start — меню\n"
        "• /random — случайный мем\n"
        "• /search — поиск\n\n"
        "<b>Inline:</b>\n"
        "В любом чате набери: <code>@имябота запрос</code>"
    )
    await message.answer(help_text, reply_markup=create_main_keyboard())


@dp.message(Command("favorites"))
async def cmd_favorites(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    if not storage.get_favorites(message.from_user.id):
        await message.answer("У вас пока нет избранных мемов")
        return
    await show_favorite_meme(message.chat.id, message.from_user.id, 0)


@dp.message(Command("addmeme"))
async def cmd_addmeme(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return

    pending_add_meme.pop(message.from_user.id, None)
    pending_add_meme[message.from_user.id] = {"step": "awaiting_photo"}
    await message.answer("Отправь фото мема. Подпись можно добавить сразу или следующим сообщением.")


@dp.message(Command("cancel"))
async def cmd_cancel(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    if pending_add_meme.pop(message.from_user.id, None) is None:
        await message.answer("Сейчас нечего отменять")
        return
    await message.answer("Добавление мема отменено")


@dp.message(Command("latestmemes"))
async def cmd_latest_memes(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    if not is_admin(message.from_user.id):
        await message.answer("РќРµС‚ РґРѕСЃС‚СѓРїР°")
        return

    parts = (message.text or "").split(maxsplit=1)
    limit = 5
    if len(parts) > 1:
        try:
            limit = max(1, min(20, int(parts[1])))
        except ValueError:
            await message.answer("РСЃРїРѕР»СЊР·СѓР№: /latestmemes 5")
            return

    latest_indices = search_engine.latest_local_indices(limit)
    if not latest_indices:
        await message.answer("Р›РѕРєР°Р»СЊРЅС‹С… РјРµРјРѕРІ РїРѕРєР° РЅРµС‚")
        return

    await message.answer(f"РџРѕСЃР»РµРґРЅРёРµ РґРѕР±Р°РІР»РµРЅРЅС‹Рµ РјРµРјС‹: {len(latest_indices)}")
    for idx in latest_indices:
        await send_meme(message.chat.id, idx, caption=f"Р›РѕРєР°Р»СЊРЅС‹Р№ РјРµРј #{idx}")


@dp.message(Command("deletememe"))
async def cmd_delete_meme(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    if not is_admin(message.from_user.id):
        await message.answer("No access")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2:
        await message.answer("Use: /deletememe <idx>")
        return

    try:
        meme_idx = int(parts[1].strip())
    except ValueError:
        await message.answer("Use: /deletememe <idx>")
        return

    await delete_local_meme_by_idx(message.chat.id, meme_idx)


@dp.message(Command("block"))
async def cmd_block_user(message: types.Message) -> None:
    if not is_admin(message.from_user.id):
        await message.answer("No access")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2:
        await message.answer("Use: /block <user_id>")
        return

    target_user_id = parse_target_user_id(parts[1])
    if target_user_id is None:
        await message.answer("Use numeric user id: /block <user_id>")
        return
    if target_user_id == message.from_user.id:
        await message.answer("You cannot block yourself.")
        return

    if storage.block_user(target_user_id):
        await message.answer(f"Blocked user {target_user_id}")
    else:
        await message.answer(f"User {target_user_id} is already blocked")


@dp.message(Command("unblock"))
async def cmd_unblock_user(message: types.Message) -> None:
    if not is_admin(message.from_user.id):
        await message.answer("No access")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2:
        await message.answer("Use: /unblock <user_id>")
        return

    target_user_id = parse_target_user_id(parts[1])
    if target_user_id is None:
        await message.answer("Use numeric user id: /unblock <user_id>")
        return

    if storage.unblock_user(target_user_id):
        await message.answer(f"Unblocked user {target_user_id}")
    else:
        await message.answer(f"User {target_user_id} is not blocked")


@dp.message(Command("blocked"))
async def cmd_blocked_users(message: types.Message) -> None:
    if not is_admin(message.from_user.id):
        await message.answer("No access")
        return

    blocked = storage.list_blocked_users()
    if not blocked:
        await message.answer("Blocked list is empty")
        return
    await message.answer("Blocked users:\n" + "\n".join(blocked[:200]))


@dp.message(Command("warmup_start"))
async def cmd_warmup_start(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    global warmup_task
    if not is_admin(message.from_user.id):
        await message.answer("Нет доступа")
        return
    if warmup_task and not warmup_task.done():
        await message.answer("Warmup уже запущен. Используй /warmup_status или /warmup_stop")
        return

    warmup_task = asyncio.create_task(warmup_cache(message.chat.id))
    await message.answer("Запустил прогрев. Используй /warmup_status для проверки прогресса.")


@dp.message(Command("warmup_stop"))
async def cmd_warmup_stop(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    global warmup_task
    if not is_admin(message.from_user.id):
        await message.answer("Нет доступа")
        return
    if warmup_task and not warmup_task.done():
        warmup_task.cancel()
        await message.answer("Остановил прогрев. Можно продолжить через /warmup_start")
    else:
        await message.answer("Прогрев не запущен")


@dp.message(Command("warmup_status"))
async def cmd_warmup_status(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    if not is_admin(message.from_user.id):
        await message.answer("Нет доступа")
        return

    state = storage.load_warmup_state()
    await message.answer(
        "Warmup статус:\n"
        f"next_idx: {state.get('next_idx', 0)} / {search_engine.total()}\n"
        f"ok: {state.get('ok', 0)}\n"
        f"fail: {state.get('fail', 0)}\n"
        f"cached file_ids: {len(storage.file_id_cache)}"
    )


@dp.message(lambda message: bool(message.photo))
async def handle_photo(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    user_id = message.from_user.id
    state = pending_add_meme.get(user_id)
    if state is None:
        return
    try:
        image_path = await save_uploaded_photo(message.photo[-1])
    except Exception:
        logger.exception("Ошибка сохранения загруженного фото")
        await message.answer("Не удалось сохранить фото")
        return

    pending_add_meme[user_id] = {"step": "awaiting_description", "image_path": image_path}
    caption = (message.caption or "").strip()
    if caption:
        await finalize_add_meme(message.chat.id, user_id, caption)
        return

    await message.answer("Фото получил. Теперь отправь подпись или описание для поиска.")


@dp.message()
async def handle_text(message: types.Message) -> None:
    if is_blocked(message.from_user.id):
        return
    state = pending_add_meme.get(message.from_user.id)
    if state is not None:

        if state.get("step") == "awaiting_photo":
            await message.answer("Сейчас жду картинку. Отправь фото мема или /cancel.")
            return

        if state.get("step") == "awaiting_description":
            description = (message.text or "").strip()
            if len(description) < 2:
                await message.answer("Описание слишком короткое. Отправь хотя бы 2 символа.")
                return
            await finalize_add_meme(message.chat.id, message.from_user.id, description)
            return

    query = (message.text or "").strip()
    if len(query) < 2:
        await message.answer("Введи минимум 2 символа для поиска")
        return
    await perform_search(message.chat.id, query, message.from_user.id)


@dp.inline_query()
async def inline_search(inline_query: types.InlineQuery) -> None:
    if is_blocked(inline_query.from_user.id):
        await inline_query.answer([], cache_time=5, is_personal=True)
        return
    global BOT_USERNAME

    query = (inline_query.query or "").strip()
    user_id = inline_query.from_user.id
    if len(query) < 1:
        await inline_query.answer([], cache_time=5, is_personal=True)
        return

    if BOT_USERNAME is None:
        try:
            BOT_USERNAME = (await bot.get_me()).username
        except Exception:
            logger.exception("Не удалось получить имя бота")
            BOT_USERNAME = None

    meme_indices = await asyncio.to_thread(search_engine.search, query, user_id, config.inline_limit)
    results = []

    for idx in meme_indices:
        file_id = storage.get_file_id(idx)
        if file_id:
            title = html.escape(search_engine.title_for_idx(idx))[:64]
            results.append(
                InlineQueryResultCachedPhoto(
                    id=f"photo_{idx}",
                    photo_file_id=file_id,
                    title=title,
                )
            )
        else:
            deep_param = f"send_{idx}"
            if BOT_USERNAME:
                deep_link = f"https://t.me/{BOT_USERNAME}?start={deep_param}"
                text = f"Открой бота, чтобы получить мем #{idx}: {deep_link}"
            else:
                text = f"Открой бота, чтобы получить мем #{idx}: /start {deep_param}"

            results.append(
                InlineQueryResultArticle(
                    id=f"open_{idx}",
                    title=f"Открыть мем #{idx} в боте",
                    input_message_content=InputTextMessageContent(message_text=text),
                    description="Если фото не закешировано, откроется через /start",
                )
            )

        if len(results) >= config.inline_limit:
            break

    await inline_query.answer(results, cache_time=config.inline_cache_time, is_personal=True)


@dp.callback_query()
async def handle_callbacks(callback: types.CallbackQuery) -> None:
    if is_blocked(callback.from_user.id):
        await callback.answer()
        return
    data = callback.data or ""
    user_id = callback.from_user.id

    try:
        if data == "random":
            await callback.answer("Ищу случайный мем...")
            await send_random_meme(callback.message.chat.id, user_id)   
        elif data == "search":
            await callback.answer()
            await callback.message.answer("Введи запрос для поиска мемов")
        elif data == "help":
            await callback.answer()
            await cmd_help(callback.message)
        elif data == "menu":
            await callback.answer()
            await cmd_start(callback.message)
        elif data.startswith("more:"):
            query = search_engine.resolve_query_token(data[5:])
            if not query:
                await callback.answer("Запрос устарел, отправь его заново")
                return
            await callback.answer("Ищу еще мемы...")
            await perform_search(callback.message.chat.id, query, user_id)
        elif data.startswith("fav:"):
            meme_idx = int(data[4:])
            if storage.add_favorite(user_id, meme_idx):
                await callback.answer("Добавлено в избранное")
            else:
                await callback.answer("Уже в избранном")
        elif data == "my_favorites":
            await callback.answer()
            await show_favorite_meme(callback.message.chat.id, user_id, 0)
        elif data.startswith("fav_next:"):
            await callback.answer()
            await show_favorite_meme(callback.message.chat.id, user_id, int(data.split(":")[1]))
        elif data.startswith("fav_prev:"):
            await callback.answer()
            await show_favorite_meme(callback.message.chat.id, user_id, int(data.split(":")[1]))
        elif data.startswith("unfav:"):
            parts = data.split(":")
            meme_idx = int(parts[1])
            page = int(parts[2]) if len(parts) > 2 else 0
            if storage.remove_favorite(user_id, meme_idx):
                await callback.answer("Удалено из избранного")
                await callback.message.delete()
                user_favs = storage.get_favorites(user_id)
                if user_favs:
                    await show_favorite_meme(callback.message.chat.id, user_id, min(page, len(user_favs) - 1))
                else:
                    await callback.message.answer("Избранные мемы пусты", reply_markup=create_main_keyboard())
            else:
                await callback.answer("Этого мема нет в избранном")
        else:
            await callback.answer()
    except Exception:
        logger.exception("Ошибка в callback")
        await callback.answer("Ошибка")


async def main() -> None:
    global BOT_USERNAME

    logger.info("Запуск MemeBot...")
    try:
        await bot.set_my_commands(setup_main_menu())

        BOT_USERNAME = (await bot.get_me()).username
        logger.info("Bot username: @%s", BOT_USERNAME)
    except TelegramRetryAfter:
        raise
    except Exception:
        logger.exception("Не удалось инициализировать бота")
        raise
    else:
        if config.cache_chat_id is None:
            logger.warning("CACHE_CHAT_ID не задан, inline будет работать только через deep-link fallback")
        else:
            logger.info("CACHE_CHAT_ID = %s", config.cache_chat_id)

        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())


