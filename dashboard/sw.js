/* MAS dashboard service worker.
   Shell (html/css/js/icons): cache-first, refreshed in the background.
   data.json: network-first with cache fallback, so the home-screen app opens
   instantly offline with the last-published snapshot. Bump VERSION together
   with the ?v= asset query in index.html. */
const VERSION = "mas-v14";
const SHELL = [
  "./",
  "index.html",
  "style.css?v=14",
  "app.js?v=14",
  "manifest.webmanifest",
  "icons/icon-192.png",
  "icons/icon-512.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;

  // data.json — freshest wins, cache is the offline fallback
  if (url.pathname.endsWith("/data.json")) {
    e.respondWith(
      fetch(e.request)
        .then((resp) => {
          const copy = resp.clone();
          caches.open(VERSION).then((c) => c.put("data.json", copy));
          return resp;
        })
        .catch(() => caches.match("data.json"))
    );
    return;
  }

  // shell — cache-first, background refresh
  e.respondWith(
    caches.match(e.request).then((hit) => {
      const refresh = fetch(e.request)
        .then((resp) => {
          if (resp.ok) caches.open(VERSION).then((c) => c.put(e.request, resp.clone()));
          return resp;
        })
        .catch(() => hit);
      return hit || refresh;
    })
  );
});
