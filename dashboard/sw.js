/* MAS dashboard service worker.

   NAVIGATIONS (the HTML document) and data.json: NETWORK-FIRST with cache
   fallback. The document must be network-first — with cache-first, the first
   open after every deploy served the PREVIOUS shell (verified live: pull-to-
   refresh and the pipeline section were absent on load 1 and present on load 2).
   For a home-screen app opened once each morning that means every deploy costs
   one stale session, which is exactly when the user is checking whether the
   deploy worked.

   Sub-resources (css/js/icons): cache-first is correct — they are versioned by
   ?v=, so a fresh document always requests the right ones, and a stale one is
   never served against a new document.

   Bump VERSION together with the ?v= asset query in index.html. */
const VERSION = "mas-v16";
const SHELL = [
  "./",
  "index.html",
  "style.css?v=16",
  "app.js?v=16",
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

  // Document navigations — freshest shell wins, cache is the offline fallback.
  if (e.request.mode === "navigate" || e.request.destination === "document") {
    e.respondWith(
      fetch(e.request)
        .then((resp) => {
          // Only cache a GOOD response. Pages can 404/500 transiently, and
          // caching one would overwrite the working offline shell with an error
          // page that then persists across launches.
          if (resp.ok) {
            const copy = resp.clone();
            e.waitUntil(caches.open(VERSION).then((c) => c.put("index.html", copy)));
          }
          return resp;
        })
        .catch(() => caches.match("index.html").then((hit) => hit || caches.match("./")))
    );
    return;
  }

  // data.json — freshest wins, cache is the offline fallback
  if (url.pathname.endsWith("/data.json")) {
    e.respondWith(
      fetch(e.request)
        .then((resp) => {
          // Same rule as the document: a bad response must not replace the last
          // good snapshot, or the app opens offline showing an error body.
          if (resp.ok) {
            const copy = resp.clone();
            e.waitUntil(caches.open(VERSION).then((c) => c.put("data.json", copy)));
          }
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
          // waitUntil so a cache HIT cannot let the worker terminate before the
          // background refresh finishes writing.
          if (resp.ok) e.waitUntil(caches.open(VERSION).then((c) => c.put(e.request, resp.clone())));
          return resp;
        })
        .catch(() => hit);
      return hit || refresh;
    })
  );
});
