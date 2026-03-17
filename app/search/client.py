import json
from urllib import error, parse, request


class SearchApiClient:
    def __init__(self, base_url, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path, params=None):
        url = self.base_url + path
        if not params:
            return url
        query = parse.urlencode({k: v for k, v in params.items() if v is not None})
        if not query:
            return url
        return url + "?" + query

    def _request(self, method, path, data=None, params=None):
        body = None
        headers = {}
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(self._url(path, params), data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except error.HTTPError as e:
            payload = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"search api {e.code}: {payload or e.reason}") from e
        except error.URLError as e:
            raise RuntimeError(f"search api error: {e.reason}") from e
        if not payload:
            return {}
        return json.loads(payload.decode("utf-8"))

    def health(self):
        return self._request("GET", "/health")

    def get_meme(self, idx):
        return self._request("GET", f"/memes/{idx}")["meme"]

    def get_meme_meta(self, idx):
        return self._request("GET", f"/memes/{idx}/meta")["meme"]

    def get_random_meme(self):
        return self._request("GET", "/memes/random")["meme"]

    def total(self):
        return int(self._request("GET", "/stats")["total"])

    def add_local_meme(self, image_path, description, alt=""):
        return self._request(
            "POST",
            "/memes/local",
            {"image_path": image_path, "description": description, "alt": alt},
        )["meme"]

    def delete_local_meme(self, idx):
        data = self._request("DELETE", f"/memes/local/{idx}")
        return data.get("meme")

    def latest_local_memes(self, limit=10):
        return list(self._request("GET", "/memes/local/latest", params={"limit": limit}).get("items", []))

    def start_search_session(self, query, user_id, page_size=5):
        return self._request(
            "POST",
            "/search/sessions",
            {"query": query, "user_id": user_id, "page_size": page_size},
        )

    def next_search_results(self, token, user_id, page_size=5):
        return self._request(
            "POST",
            "/search/sessions/next",
            {"token": token, "user_id": user_id, "page_size": page_size},
        )

    def get_search_session(self, token, user_id):
        return self._request("GET", f"/search/sessions/{token}", params={"user_id": user_id})

    def search(self, query, limit=10):
        return list(self._request("POST", "/search", {"query": query, "limit": limit}).get("items", []))
