import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from app.search.engine import SearchEngine
from app.search.engine import coerce_text_value
from app.shared.config import configure_logging, load_config
from app.shared.media import get_image_bytes
from app.shared.storage import BotStorage


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


cfg = load_config()
log = configure_logging()
storage = BotStorage(cfg)
engine = SearchEngine(cfg, log)
lock = threading.Lock()


def parse_int(value, default=0):
    if value in (None, ""):
        return default
    return int(value)


def image_path_of(row):
    image = row.get("image", {})
    if isinstance(image, dict):
        path = image.get("path")
        if path:
            return str(path)
    return ""


def meta_of(idx):
    row = engine.row(idx)
    return {
        "idx": int(idx),
        "title": engine.title_for_idx(idx),
        "source": str(row.get("source") or ""),
        "image_path": image_path_of(row),
    }


def meme_of(idx):
    item = meta_of(idx)
    row = engine.row(idx)
    item["image_b64"] = base64.b64encode(get_image_bytes(row)).decode("ascii")
    return item


def read_json(handler):
    size = parse_int(handler.headers.get("Content-Length"), 0)
    if size <= 0:
        return {}
    body = handler.rfile.read(size)
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def write_json(handler, status, data):
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def handle_get(path, query):
    if path == "/health":
        return {"ok": True}
    if path == "/stats":
        return {"total": engine.total()}
    if path == "/memes/random":
        return {"meme": meme_of(engine.random_idx())}
    if path == "/memes/local/latest":
        limit = parse_int(first(query, "limit"), 10)
        return {"items": [meta_of(idx) for idx in engine.latest_local_indices(limit)]}
    if path.startswith("/memes/") and path.endswith("/meta"):
        idx = parse_int(path.split("/")[2])
        return {"meme": meta_of(idx)}
    if path.startswith("/memes/"):
        idx = parse_int(path.split("/")[2])
        return {"meme": meme_of(idx)}
    if path.startswith("/search/sessions/"):
        token = path.rsplit("/", 1)[1]
        user_id = parse_int(first(query, "user_id"), -1)
        session = engine.search_sessions.get(token)
        if not session or int(session.get("user_id", -1)) != user_id:
            raise ApiError(404, "search session not found")
        return {
            "token": token,
            "query": str(session.get("query") or ""),
            "has_more": engine.search_session_has_more(token),
        }
    raise ApiError(404, "not found")


def handle_post(path, data):
    if path == "/search":
        query = str(data.get("query") or "")
        limit = parse_int(data.get("limit"), 10)
        items = [meta_of(idx) for idx in engine.search(query, limit=limit)]
        return 200, {"items": items}
    if path == "/search/sessions":
        query = str(data.get("query") or "")
        user_id = parse_int(data.get("user_id"), -1)
        page_size = parse_int(data.get("page_size"), 5)
        token, indices, has_more = engine.start_search_session(query, user_id, page_size)
        return 200, {
            "token": token,
            "query": query,
            "indices": indices,
            "items": [meta_of(idx) for idx in indices],
            "has_more": has_more,
        }
    if path == "/search/sessions/next":
        token = str(data.get("token") or "")
        user_id = parse_int(data.get("user_id"), -1)
        page_size = parse_int(data.get("page_size"), 5)
        indices = engine.next_search_results(token, user_id, page_size)
        return 200, {
            "token": token,
            "query": engine.search_session_query(token),
            "indices": indices,
            "items": [meta_of(idx) for idx in indices],
            "has_more": engine.search_session_has_more(token),
        }
    if path == "/memes/local":
        image_path = str(data.get("image_path") or "").strip()
        description = str(data.get("description") or "").strip()
        alt = str(data.get("alt") or "").strip()
        storage.append_local_meme(image_path=image_path, description=description, alt=alt)
        idx = engine.add_local_meme(image_path=image_path, description=description, alt=alt)
        return 201, {"meme": meme_of(idx)}
    raise ApiError(404, "not found")


def meta_from_record(idx, record, title):
    return {
        "idx": int(idx),
        "title": title,
        "source": str(record.get("source") or ""),
        "image_path": image_path_of(record),
    }


def title_from_record(idx, record):
    return (
        coerce_text_value(record.get("query_phrases"))
        or coerce_text_value(record.get("description"))
        or coerce_text_value(record.get("alt"))
        or f"Мем #{idx}"
    )


def handle_delete(path):
    if not path.startswith("/memes/local/"):
        raise ApiError(404, "not found")
    idx = parse_int(path.rsplit("/", 1)[1])
    record = engine.delete_local_meme(idx)
    if record is None:
        return {"meme": None}
    image_path = image_path_of(record)
    if image_path:
        storage.delete_local_meme(image_path)
    if image_path and os.path.exists(image_path):
        os.remove(image_path)
    return {"meme": meta_from_record(idx, record, title_from_record(idx, record))}


def first(query, key):
    values = query.get(key)
    if not values:
        return ""
    return values[0]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.handle_request("GET")

    def do_POST(self):
        self.handle_request("POST")

    def do_DELETE(self):
        self.handle_request("DELETE")

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)

    def handle_request(self, method):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            with lock:
                if method == "GET":
                    data = handle_get(parsed.path, query)
                    status = 200
                elif method == "POST":
                    status, data = handle_post(parsed.path, read_json(self))
                elif method == "DELETE":
                    data = handle_delete(parsed.path)
                    status = 200
                else:
                    raise ApiError(405, "method not allowed")
        except KeyError as e:
            write_json(self, 404, {"error": str(e)})
            return
        except ValueError as e:
            write_json(self, 400, {"error": str(e)})
            return
        except ApiError as e:
            write_json(self, e.status, {"error": e.message})
            return
        except Exception as e:
            log.exception("api error")
            write_json(self, 500, {"error": str(e)})
            return
        write_json(self, status, data)


def main():
    addr = (cfg.search_api_host, cfg.search_api_port)
    log.info("Search API listening on %s:%s", cfg.search_api_host, cfg.search_api_port)
    server = ThreadingHTTPServer(addr, Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
